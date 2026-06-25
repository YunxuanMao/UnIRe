from typing import Dict, List, Tuple
import logging
import random

import torch
from torch.nn import Parameter

from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians
from utils.flow_viz import vis_occ_plotly, map_colors
from torch_scatter import scatter

import roma
from pytorch3d.ops.knn import knn_points
logger = logging.getLogger()


class UnIReGS(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._delta_means = torch.zeros(1, 1, 3, device=self.device)
        self.test_set_indices = None

    @property
    def use_delta(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
        
    @property
    def num_instances(self):
        instance_ids = self.point_ids.unique()
        instance_ids = instance_ids[instance_ids != -1]
        return len(instance_ids)
    
    @property
    def num_frames(self):
        return self.instances_fv.shape[1]


    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)

    def _in_test_set(self, frame_id=None):
        if frame_id is None:
            frame_id = self.cur_frame
        if self.test_set_indices is None:
            return self.in_test_set
        else:
            if isinstance(frame_id, torch.Tensor):
                return frame_id.item() in self.test_set_indices
            else:
                return frame_id in self.test_set_indices
            
    def get_pts_valid_mask(self, frame_id=None):
        """
        get the mask for valid points
        """
        if frame_id is None:
            frame_id = self.cur_frame
        return self.instances_fv[self.point_ids, frame_id]
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id

    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)

    def create_from_pcd(self, init_means, init_colors, init_cluster_ids, init_cluster_fv, cluster_trans) -> None:

        # collect all instances
        init_means = init_means.to(self.device).float() # (N, 3)
        init_colors = init_colors.to(self.device).float() # (N, 3)
        self.point_ids = init_cluster_ids.to(self.device).long() # (N, 1)
        self.instances_fv = init_cluster_fv.to(self.device) # (num_instances,num_frames,)
        
        
        instances_trans = cluster_trans.to(self.device).float() # (num_instances, num_frames, 3)
        init_delta_means = instances_trans[self.point_ids] #(N, num_frames)
        
        
        # initialize the means, scales, quats, and colors
        self._means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)
        
        
        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))
        self._delta_means = Parameter(init_delta_means)

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
            self.class_prefix+"delta_means": [self._delta_means]            
        }

    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        return param_groups
    

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():


            

            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                    split_delta_means
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                    dup_delta_means
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self._delta_means = Parameter(torch.cat([self._delta_means.detach(), split_delta_means, dup_delta_means], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)




            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
                print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()

        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._delta_means = Parameter(self._delta_means[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps)
        new_delta_means = self._delta_means[split_mask].repeat(samps, 1, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_ids, new_delta_means

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        dup_delta_means = self._delta_means[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_ids, dup_delta_means
    

    def cull_gaussians_importance(self, culls, optimizer: torch.optim.Optimizer):
        
        n_bef = self.num_points
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._delta_means = Parameter(self._delta_means[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"Importance Cull: {n_bef - self.num_points}")

        param_groups = self.get_gaussian_param_groups()
        remove_from_optim(optimizer, culls, param_groups)

    
    def get_deformation(self, frame_id = None, enable_temporal_smoothing=None) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """
        if frame_id is None:
            frame_id = self.cur_frame
        
        if enable_temporal_smoothing is None:
            enable_temporal_smoothing = self.ctrl_cfg.enable_temporal_smoothing
        
        if self._in_test_set(frame_id):
            if frame_id - 1 > 0 and frame_id + 1 < self.num_frames:
                _prev_delta_means = self._delta_means[:, frame_id - 1]
                _next_delta_means = self._delta_means[:, frame_id + 1]
                _cur_delta_means = self._delta_means[:, frame_id]
                interpolated_trans = (_prev_delta_means + _next_delta_means) * 0.5
                inter_valid_mask = self.instances_fv[self.point_ids, frame_id - 1] & self.instances_fv[self.point_ids, frame_id + 1]
                delta_means = torch.where(
                    inter_valid_mask[:, None], interpolated_trans, _cur_delta_means
                )

            else:
                delta_means = self._delta_means[:, frame_id]

        elif enable_temporal_smoothing and self.training and random.random() < self.ctrl_cfg.smooth_probability and (
                frame_id - 1 > 0 and frame_id + 1 < self.num_frames
            ):
                _prev_delta_means, _,_ = self.get_deformation(frame_id-1, enable_temporal_smoothing=False)
                _next_delta_means,_,_ = self.get_deformation(frame_id+1, enable_temporal_smoothing=False)
                delta_means = (_prev_delta_means + _next_delta_means) * 0.5
        else:
            delta_means = self._delta_means[:, frame_id]
        
        return delta_means


    def get_gaussians(self, cam: dataclass_camera, flow_dir='prev') -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        # filter_mask = self.get_dynamic_mask(0.1)
        self.filter_mask = filter_mask

        # delta_xyz = None
        # if self.use_delta:
        delta_xyz = self.get_deformation()
        if not self.use_delta:
            delta_xyz = delta_xyz.detach()
        

        if self.step < self.ctrl_cfg.stop_optimizing_canonical_xyz_after:
            dynamic_means = self._means + delta_xyz
        else:
            dynamic_means = self._means.data + delta_xyz

        
        world_means = dynamic_means

        world_quats = self.get_quats

        activated_scales = self.get_scaling

        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        valid_mask = self.get_pts_valid_mask()

        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_rotations = self.quat_act(world_quats)
        activated_colors = rgbs

        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=activated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
            instance_id=self.point_ids[filter_mask],
        )


        if flow_dir == 'prev':
            prev_frame_id = self.cur_frame-1
        else:
            prev_frame_id = self.cur_frame+1
        
        prev_delta_xyz = self.get_deformation(prev_frame_id)

        prev_dynamic_means = self._means.data + prev_delta_xyz
        
        prev_world_means = prev_dynamic_means
        gs_dict["_prev_means"] = prev_world_means[filter_mask]

        
        # check nan in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
            "world_means": world_means[filter_mask],
            "delta_means": delta_xyz[filter_mask] if delta_xyz is not None else None,
            "prev_world_means": prev_world_means[filter_mask], 
            "prev_delta_means": prev_delta_xyz[filter_mask],
            "prev_frame_id": prev_frame_id
        }

        return gs_dict
    

    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        
        if self.use_delta:
            delta_xyz = self._gs_cache["delta_means"]
            prev_delta_xyz = self._gs_cache["prev_delta_means"]
            world_means = self._gs_cache["world_means"]
            valid_mask = self.get_pts_valid_mask()

            flow = delta_xyz - prev_delta_xyz
            flow_valid_mask = torch.logical_and(valid_mask, self.get_pts_valid_mask(self._gs_cache["prev_frame_id"]))
            
            flow = flow[flow_valid_mask]

            flow_reg = self.reg_cfg.get("flow_reg", None)
            if flow_reg:
                if (self.cur_radii > 0).sum():
                    K = flow_reg.K
                    
                    dist, nn, _ = knn_points(world_means[flow_valid_mask][None], world_means[flow_valid_mask][None], lengths1=None, lengths2=None, K=K)   
                    knnmask = (dist.squeeze() > 1e-7).float()
                    knnmask = knnmask.reshape(-1, K, 1)

                    nn = nn.reshape(-1)
                    nn_flow = flow[nn].reshape(-1, K, 3)
                    nn_flow_avg = (nn_flow * knnmask).sum(1) / (knnmask.sum(1)+1e-7) # [n, 3]
                    flow_loss = (flow - nn_flow_avg).abs().sum(dim=-1) * (K - 1) / K

                    loss_dict["flow_reg"] = flow_loss.mean() * flow_reg.w
        

            static_reg = self.reg_cfg.get("static_reg", None)
            if static_reg:
                
                local_prev_delta_xyz = prev_delta_xyz - scatter(prev_delta_xyz, self.point_ids, dim=0, reduce='mean')[self.point_ids]
                local_delta_xyz = delta_xyz - scatter(delta_xyz, self.point_ids, dim=0, reduce='mean')[self.point_ids]
                local_flow = local_prev_delta_xyz - local_delta_xyz
                static_loss = local_flow.norm(dim=-1).mean()
                loss_dict["static_reg"] = static_loss * static_reg.w

        
        return loss_dict


    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update({
            "points_ids": self.point_ids,
            "instances_fv": self.instances_fv,
        })
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self.point_ids = state_dict.pop("points_ids")
        self.instances_fv = state_dict.pop("instances_fv")
        self._delta_means = Parameter(torch.zeros((N,) + (self.num_frames, 3), device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg

    
    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        remove instances from the model
        
        Args:
            remove_id_list: list of instance ids to be removed
        """
        for ins_ids in remove_id_list:
            mask = ~(self.point_ids == ins_ids)
            self._means = Parameter(self._means[mask])
            self._scales = Parameter(self._scales[mask])
            self._quats = Parameter(self._quats[mask])
            self._features_dc = Parameter(self._features_dc[mask])
            self._features_rest = Parameter(self._features_rest[mask])
            self._opacities = Parameter(self._opacities[mask])
            self._delta_means = Parameter(self._delta_means[mask])
            self.point_ids = self.point_ids[mask]

    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                instance_raw_dict = {
                    "_means": self._means[self.point_ids == id].detach().cpu(),
                    "_scales": self._scales[self.point_ids == id].detach().cpu(),
                    "_quats": self._quats[self.point_ids == id].detach().cpu(),
                    "_features_dc": self._features_dc[self.point_ids == id].detach().cpu(),
                    "_features_rest": self._features_rest[self.point_ids == id].detach().cpu(),
                    "_opacities": self._opacities[self.point_ids == id].detach().cpu(),
                    "_delta_means": self._delta_means[self.point_ids == id].detach().cpu(),
                    "point_ids": self.point_ids[self.point_ids == id].detach().cpu(),
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict


    def replace_instances(self, replace_dict: Dict[int, int], new_gaussians_dict:  Dict) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
            new_gaussians_dict: {
                ins_id(replace with): gs_dict
                ...
            }
        """
        # new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            for k,v in new_gaussian.items():
                new_gaussian[k] = v.to(self.device)
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0))
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0))
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            self._delta_means = Parameter(torch.cat([self._delta_means, new_gaussian["_delta_means"][:, :self.num_frames]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)

    
    def finer_decompose(self, optimizer, thres=0.1):
        '''
        Change dynamic points to static
        '''
        dynamic_mask = self.get_dynamic_mask_by_flow(thres) # (n_dyn,)
        

        static_means = self._means[~dynamic_mask].data.clone()
        static_scales = self._scales[~dynamic_mask].data.clone()
        static_quats = self._quats[~dynamic_mask].data.clone()
        static_features_dc = self._features_dc[~dynamic_mask].data.clone()
        static_features_rest = self._features_rest[~dynamic_mask].data.clone()
        static_opacities = self._opacities[~dynamic_mask].data.clone()
        static_delta_means = self._delta_means[~dynamic_mask].data.clone()
        static_point_ids = self.point_ids[~dynamic_mask]
        static_point_fv = self.instances_fv[static_point_ids]
        static_means += (static_delta_means * static_point_fv[..., None]).sum(1) / static_point_fv[..., None].sum(1)
        

        self._means = Parameter(self._means[dynamic_mask].detach())
        self._scales = Parameter(self._scales[dynamic_mask].detach())
        self._quats = Parameter(self._quats[dynamic_mask].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[dynamic_mask].detach())
        self._features_rest = Parameter(self._features_rest[dynamic_mask].detach())
        self._opacities = Parameter(self._opacities[dynamic_mask].detach())
        self._delta_means = Parameter(self._delta_means[dynamic_mask].detach())
        self.point_ids = self.point_ids[dynamic_mask]

        print(f"Finer decompose: {(~dynamic_mask).sum()} dynamic points -> static points")

        param_groups = self.get_gaussian_param_groups()
        remove_from_optim(optimizer, ~dynamic_mask, param_groups)

        return {
            "_means": static_means,
            "_scales": static_scales,
            "_quats": static_quats,
            "_features_dc": static_features_dc,
            "_features_rest": static_features_rest,
            "_opacities": static_opacities,
        }
        


    def get_dynamic_mask_by_flow(self, thres):
        flows = self._delta_means[:, 1:] - self._delta_means[:, :-1]
        flow_valid_mask = torch.logical_and(self.instances_fv[self.point_ids, 1:], self.instances_fv[self.point_ids, :-1])
        flows[..., -1] = 0
        flows_means = (flows.norm(dim=-1) * flow_valid_mask).sum(1) / flow_valid_mask.sum(1)
        dynamic_mask = flows_means.squeeze() > thres
        return dynamic_mask