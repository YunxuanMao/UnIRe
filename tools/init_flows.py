
from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset
from utils.init_utils import get_road_label, get_interest_points, vis_init, plane_fitting, query_sdf
from models.flow_model.let_it_flow import run_flow_initialization
from models.flow_model.cluster_utils import run_cluster_init

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    return cfg


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_cfg = cfg.pop('init')

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    dynamic_pts = []
    dynamic_color = []
    static_pts = []
    static_color = []
    
    if init_cfg.get("from_lidar", None) is not None:
        cam_height = init_cfg.get("cam_height", 2)
        # near_trunc = init_cfg.get("near_trunc", 30.)
        init_cam_traj=dataset.train_image_set.datasource.front_camera_trajectory
        for timestep in dataset.lidar_source.timesteps.unique():
            # pts, color, _ = dataset.get_lidar_samples_t(
            #     timestep=timestep, **init_cfg.from_lidar, device=self.device
            # )
            
            
            pts, color, _, road_label = dataset.get_lidar_samples_t(
                timestep=timestep, **init_cfg.from_lidar, device=device
            )
            if not init_cfg.get("use_gt_ground", True) or road_label is None:
                road_label = get_road_label(pts, init_cam_traj, ego_height=cam_height)
            
            # near_label = get_interest_points(pts, init_cam_traj, interest_range=near_trunc)
            # dynamic_label = torch.logical_and(near_label, ~road_label)
            dynamic_label = ~road_label

            dynamic_pts.append(pts[dynamic_label])
            dynamic_color.append(color[dynamic_label])
            static_pts.append(pts[~dynamic_label])
            static_color.append(color[~dynamic_label])
        
        
        static_pts = torch.cat(static_pts)
        static_color = torch.cat(static_color)
        num_static_pts = init_cfg.get('ground_randoms', 100000)
        sampled_idx = torch.randperm(len(static_pts))[:num_static_pts]
        ground_pts = static_pts[sampled_idx]
        ground_colors = static_color[sampled_idx]
        

        points, flows, ins, color = \
                run_flow_initialization(dynamic_pts, dynamic_color, init_cfg.flow_cfg, device=device)


        # static_pts = torch.cat([static_means, ground_pts])
        # static_colors = torch.cat([static_colors, ground_colors])
        static_pts = ground_pts
        static_colors = ground_colors

        

        # vis_init(sampled_pts, flows_all, sampled_pts[:, None] + points_delta, cluster_fv, init_cluster, cfg.log_dir, )

        ckpt = {
            'points_list': points,
            'flow_list': flows,
            'cluster_list': ins,
            'color_list': color,
            'static_means': static_pts.cpu().detach(), 
            'static_colors': static_colors.cpu().detach()
        }

        torch.save(ckpt, f'{cfg.log_dir}/flow.pt')




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Initialize flows")
    parser.add_argument("--config_file", default="./configs/init_flow.yaml", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./output_flow/", help="path to save checkpoints and logs", type=str)
    parser.add_argument("--run_name", type=str, help="wandb run name, also used to enhance log_dir")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    main(args)