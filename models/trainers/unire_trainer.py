from typing import Dict
import torch
import logging
import time
import torch.nn.functional as F

from models.gaussians.basics import *
from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere
import open3d as o3d
import numpy as np
from utils.init_utils import get_road_label, get_interest_points, vis_init
from pytorch3d.ops.knn import knn_points
from models.renderers.diff_gaussian_renderer import count_render_rasterization_wrapper, get_optical_flow, calculate_v_imp_score
from models.losses import smooth_loss
import random

logger = logging.getLogger()

class UnIReTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "DynamicNodes" in self.model_config:
            self.gaussian_classes["DynamicNodes"] = GSModelType.DynamicNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

        
    
    @property
    def render_flow(self):
        render_flow = False
        if self.training:
            optical_flow_cfg = self.losses_dict.get("optical_flow", None)
            if optical_flow_cfg is not None:
                if self.step > optical_flow_cfg.get("enable_after", 5000):
                    render_flow = True
            optical_flow_smooth_cfg = self.losses_dict.get("optical_flow_smooth", None)
            if optical_flow_smooth_cfg is not None:
                if self.step > optical_flow_smooth_cfg.get("enable_after", 5000):
                    render_flow = True
        else:
            render_flow = True
        return render_flow

    @property
    def render_instance(self):
        render_instance = False
        # if not self.training:
        #     render_instance = True
        
        return render_instance

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # collect models

        if self.model_config.init_flow_path is not None:
            ckpt = torch.load(self.model_config.init_flow_path)
            dynamic_pts_dict = ckpt['cluster_points_dict']
        dynamic_color_dict = ckpt['cluster_color_dict']
        cluster_deform = ckpt['cluster_deform'].to(self.device)
        cluster_fv = ckpt['cluster_fv'].to(self.device)
        static_pts = ckpt['static_means'].to(self.device)
        static_colors = ckpt['static_colors'].to(self.device)

        dynamic_pts = []
        dynamic_color = []
        dynamic_cluster = []
        for k, v in dynamic_pts_dict.items():
            dynamic_pts.append(v)
            dynamic_color.append(dynamic_color_dict[k])
            dynamic_cluster.append(torch.ones([v.shape[0]]) * k)
        dynamic_pts = torch.cat(dynamic_pts).to(self.device)
        dynamic_color = torch.cat(dynamic_color).to(self.device)
        dynamic_cluster = torch.cat(dynamic_cluster).to(self.device)

        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')

                # static points
                # sample points from the lidar point clouds
                sampled_pts = static_pts
                sampled_color = static_colors
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)

                model.create_from_pcd(
                    init_means=sampled_pts.float(), init_colors=sampled_color.float()
                )

            if class_name == 'DynamicNodes':
                init_cfg = model_cfg.pop('init')
                # dynamic points
                num_dynamic_pts = init_cfg.get('dynamic_randoms', 0)
                if num_dynamic_pts > 0:
                    
                    expanded_points = dynamic_pts.repeat_interleave(num_dynamic_pts, dim=0)
                    dynamic_sampled_cluster = dynamic_cluster.repeat_interleave(num_dynamic_pts, dim=0)
                    dynamic_sampled_color = torch.rand(expanded_points.shape, ).to(self.device)
 
                    noise_scale = init_cfg.get("dynamic_trunc", 0.1) 
                    noise = torch.randn_like(expanded_points) * noise_scale
                    dynamic_sampled_pts = expanded_points + noise

                    dynamic_pts = torch.cat([dynamic_pts, dynamic_sampled_pts])
                    dynamic_color = torch.cat([dynamic_color, dynamic_sampled_color])
                    dynamic_cluster = torch.cat([dynamic_cluster, dynamic_sampled_cluster])

                cluster_fv = cluster_fv
                cluster_deform = cluster_deform


                model.create_from_pcd(
                    init_means=dynamic_pts, init_colors=dynamic_color, init_cluster_ids=dynamic_cluster, init_cluster_fv=cluster_fv, cluster_trans=cluster_deform, 
                )
            
            if hasattr(model, 'test_set_indices'):
                model.test_set_indices = self.test_set_indices
                
        logger.info(f"Initialized gaussians from pcd")



    def postprocess_per_train_step(self, step: int,) -> None:
        radii = self.info["radii"]
        if self.render_cfg.absgrad:
            grads = self.info["means2d"].absgrad.clone()
        else:
            grads = self.info["means2d"].grad.clone()
        grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
        grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
        
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            
            self.models[class_name].postprocess_per_train_step(
                step=step,
                optimizer=self.optimizer,
                radii=radii[0, gaussian_mask],
                xys_grad=grads[0, gaussian_mask],
                last_size=max(self.info["width"], self.info["height"])
            )
        
        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)

    def cull_importance(self, step, dataset):

        cull_iteration = self.gaussian_ctrl_general_cfg.get("cull_importance_iteration", [])
        if step in cull_iteration:
            print("Importance culling ...")
            gaussian_list, imp_list = None, None

            for idx in range(len(dataset.split_indices)):
                image_infos, camera_infos = dataset.get_image(idx, 1)
  
                normed_time = image_infos["normed_time"].flatten()[0]
                self.cur_frame = torch.argmin(
                    torch.abs(self.normalized_timestamps - normed_time)
                )
                
                # for evaluation
                for model in self.models.values():
                    if hasattr(model, 'in_test_set'):
                        model.in_test_set = self.in_test_set

                # assigne current frame to gaussian models
                for class_name in self.gaussian_classes.keys():
                    model = self.models[class_name]
                    if hasattr(model, 'set_cur_frame'):
                        model.set_cur_frame(self.cur_frame)
                
                # prapare data
                processed_cam = self.process_camera(
                    camera_infos=camera_infos,
                    image_ids=image_infos["img_idx"].flatten()[0],
                    novel_view=False
                )
                gs = self.collect_gaussians(
                    cam=processed_cam,
                    image_ids=image_infos["img_idx"].flatten()[0]
                )

                render_pkg = self.render_count(gs, processed_cam)

                if gaussian_list is None:
                    gaussian_list, imp_list = (
                        render_pkg["gaussians_count"],
                        render_pkg["important_score"],
                    )
                else:
                    gaussian_list += render_pkg["gaussians_count"].detach()
                    imp_list += render_pkg["important_score"].detach()
            
            v_list = calculate_v_imp_score(gs.scales, imp_list, self.gaussian_ctrl_general_cfg.get("v_pow", 0.1))
            
            i = cull_iteration.index(step)
            cull_decay = self.gaussian_ctrl_general_cfg.get('cull_decay', 0.8)
            cull_percent = self.gaussian_ctrl_general_cfg.get('cull_percent', 0.5)
            percent = (cull_decay**i) * cull_percent
            sorted_tensor, _ = torch.sort(v_list, dim=0)
            index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
            value_nth_percentile = sorted_tensor[index_nth_percentile]
            culls = (v_list <= value_nth_percentile).squeeze()
            

            for class_name in self.gaussian_classes.keys():
                gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                model = self.models[class_name]
                if hasattr(model, 'cull_gaussians_importance'):
                    culls_i = culls[gaussian_mask]
                    model.cull_gaussians_importance(culls_i, self.optimizer)



    def render_count(self,
        gs: dataclass_gs,
        cam: dataclass_camera,):

        return count_render_rasterization_wrapper(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities,
            colors_precomp=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
        )
    

    def finer_decompose(self, step):
        do_finer_decompose = step in self.gaussian_ctrl_general_cfg.get('finer_decompose_at',[])
        if do_finer_decompose:
            static_gaussians = self.models['DynamicNodes'].finer_decompose(self.optimizer, self.gaussian_ctrl_general_cfg.get("finer_decompose_thres", 0.1))
            self.models['Background'].add_gaussians(self.optimizer, static_gaussians)


    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        other_features=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
    
        def render_fn(opaticy_mask=None, return_info=False):
            S_color = gs.rgbs.shape[-1]
            if other_features is None:
                colors = gs.rgbs
                S_other = 0
            else:
                colors = torch.cat([other_features, gs.rgbs], dim=-1)
                S_other = other_features.shape[-1]
                
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=colors,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            assert renders.shape[-1] == 4+S_other, f"Must render feature, rgb, depth and alpha"
            rendered_feature, rendered_rgb, rendered_depth = torch.split(renders, [S_other, S_color, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], rendered_feature
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], rendered_feature, info
        
        # render rgb and opacity
        rgb, depth, opacity, feature, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity,
            "feature": feature
        }
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        return results, render_fn
    
    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor, # leave it here for future use
        flow_dir: str = 'prev'
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "_prev_means": [],
            "instance_id": [],
            "dynamic_mask": [],
            "class_labels": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam, flow_dir)
            if gs is None:
                continue
    
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            if "dynamic_mask" not in gs:
                gs["dynamic_mask"] = torch.zeros((gs["_means"].shape[0],), device=self.device, dtype=torch.bool)

            if "_prev_means" not in gs:
                gs["_prev_means"] = gs["_means"].clone()
            
            if "instance_id" not in gs:
                gs["instance_id"] = torch.zeros((gs["_means"].shape[0]), device=self.device, dtype=torch.int)
            else:
                gs["instance_id"] += 1
            
            
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras={"_prev_means": gs_dict["_prev_means"],
                    "instance_id": gs_dict["instance_id"],
                    "dynamic_mask": gs_dict["dynamic_mask"]}       # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )

        if self.training:
            if self.cur_frame > 0 and random.random() > 0.5:
                flow_dir = 'prev'
            elif self.cur_frame + 1 < self.num_timesteps:
                flow_dir = 'next'
            else:
                flow_dir = 'prev'
        else:
            if self.cur_frame > 0:
                flow_dir = 'prev'
            else:
                flow_dir = 'next'


        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0],
            flow_dir=flow_dir
        )

        optical_flow = None
        instance = None
        other_features = []
        if self.render_instance:
            instance = F.one_hot(gs.extras["instance_id"].squeeze().to(torch.int64))
            other_features.append(instance)
            num_instances = instance.shape[-1]
        if self.render_flow and f"camera_to_world_{flow_dir}" in camera_infos:
            c2w = camera_infos["camera_to_world"]
            c2w_prev = camera_infos[f"camera_to_world_{flow_dir}"]
            means = gs.means
            prev_means = gs.extras["_prev_means"]
            if c2w_prev is not None and prev_means is not None:
                w2c_prev = torch.linalg.inv(c2w_prev)
                w2c = torch.linalg.inv(c2w)
                K = processed_cam.Ks
                optical_flow = get_optical_flow(means, prev_means, w2c, w2c_prev, K)
                other_features.append(optical_flow)

        if len(other_features) > 0:
            other_features = torch.cat(other_features, dim=-1)
        else:
            other_features = None
        

        

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.),
            other_features=other_features
        )

        if instance is not None:
            outputs['instance'] = outputs['feature'][..., :num_instances]
        if optical_flow is not None:
            outputs['flow'] = outputs['feature'][..., -2:]
        
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity, _ = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity, sep_flow = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
                if sep_flow.shape[-1] > 0:
                    outputs["Dynamic_flow"] = sep_flow
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        if "egocar_masks" in image_infos:
        # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])

        optical_flow_cfg = self.losses_dict.get("optical_flow", None)
        if optical_flow_cfg is not None and self.cur_frame > 0:
            if self.step > optical_flow_cfg.get("enable_after", 5000):
                
                gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
                certainty_mask = (image_infos["optical_flow_certainty"] > optical_flow_cfg.certainty_thres).float()
                valid_mask = certainty_mask * gt_occupied_mask
                gt_optical_flow = image_infos["optical_flow"] * valid_mask[..., None]
                predicted_optical_flow = outputs["flow"] * valid_mask[..., None]

                optical_flow_loss = torch.abs(gt_optical_flow - predicted_optical_flow).mean() * optical_flow_cfg.w
                loss_dict.update({"optical_flow_loss": optical_flow_loss})

        optical_flow_smooth_cfg = self.losses_dict.get("optical_flow_smooth", None)
        if optical_flow_smooth_cfg is not None and self.cur_frame > 0:
            if self.step == optical_flow_smooth_cfg.get("enable_after", 5000):
                self.render_dynamic_mask = True
            if self.step > optical_flow_smooth_cfg.get("enable_after", 5000) and "Dynamic_flow" in outputs:
                gt_rgb = image_infos["pixels"]
                predicted_rgb = outputs["rgb"].detach()
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                
                if dynamic_pred_mask.sum() > 0:
                    optical_flow_smooth_loss = smooth_loss(gt_rgb, outputs["Dynamic_flow"]) * optical_flow_smooth_cfg.w
                    # optical_flow_smooth_loss = smooth_loss(predicted_rgb, outputs["Dynamic_flow"]) * optical_flow_smooth_cfg.w
                    loss_dict.update({"optical_flow_smooth_loss": optical_flow_smooth_loss})

        dynamic_opa_reg = self.losses_dict.get("dynamic_opa_reg", None)
        if dynamic_opa_reg is not None:
            start_from = dynamic_opa_reg.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                dynamic_opa_loss = dynamic_opa_reg.w * outputs["Dynamic_opacity"][dynamic_pred_mask].mean()
                loss_dict.update({"dynamic_opa_loss": dynamic_opa_loss})
                

        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict
    
    def get_scene_flow(self, points, frame_id):
        for class_name in self.gaussian_classes.keys():
            if hasattr(self.models[class_name], 'get_scene_flow'):
                return self.models[class_name].get_scene_flow(points, frame_id)
    
    def save_pointcloud(self, aabb, save_dir):
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            results = model.export_gaussians_to_ply()
            positions = results['positions'].detach().cpu().numpy()
            rgb = results['colors'].detach().cpu().numpy()

            rgb = np.maximum(rgb, 0)
            max_rgb = np.max(rgb, axis=1)
            max_rgb = np.maximum(max_rgb, 1)
            rgb = rgb / max_rgb[:, np.newaxis]

            # aabb=dataset.get_aabb().numpy()
            aabb_min, aabb_max = aabb[:3], aabb[3:]
            vis_mask = np.logical_and(positions >= aabb_min, positions < aabb_max).all(-1)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions[vis_mask])
            pcd.colors = o3d.utility.Vector3dVector(rgb[vis_mask])
            o3d.io.write_point_cloud(f'{save_dir[:-4]}_{class_name}.ply', pcd)


