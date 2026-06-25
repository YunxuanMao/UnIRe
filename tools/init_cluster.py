import os
import torch
import argparse
import numpy as np
from utils.flow_viz import vis_occ_plotly, map_colors
from models.flow_model.cluster_utils import run_cluster_init, visualize_cluster, transform_canonical, vis_every_cluster
import open3d as o3d


def main():
    parser = argparse.ArgumentParser("4D SuperPoint Clustering")
    parser.add_argument("--flow_dir", required=True, help="directory containing flow.pt from init_flows.py")
    parser.add_argument("--output_dir", default=None, help="output directory (default: <flow_dir>/cluster)")
    parser.add_argument("--static_trunc", type=float, default=0.02, help="threshold for static/dynamic separation")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon for spatiotemporal clustering")
    parser.add_argument("--remove_ids", type=int, nargs="*", default=[], help="cluster IDs to manually remove")
    args = parser.parse_args()

    file_path = os.path.join(args.flow_dir, "flow.pt")
    output_folder = args.output_dir or os.path.join(args.flow_dir, "cluster")
    vis_output_folder = os.path.join(output_folder, "vis_cluster")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(vis_output_folder, exist_ok=True)

    data = torch.load(file_path)
    points = data['points_list']
    flows = data['flow_list']
    ins = data['cluster_list']
    color = data['color_list']
    static_pts = data['static_means']
    static_colors = data['static_colors']

    cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv, new_static_pts, new_static_color = \
        run_cluster_init(points, ins, color, flows, eps=args.eps, static_trunc=args.static_trunc, seperate=True)

    cluster_points_dict = {k: torch.from_numpy(v) for k, v in cluster_points_dict.items()}
    cluster_color_dict = {k: torch.from_numpy(v) for k, v in cluster_color_dict.items()}
    cluster_deform = torch.from_numpy(cluster_deform)
    cluster_fv = torch.from_numpy(cluster_fv)

    if new_static_color is not None:
        new_static_pts, new_static_color = torch.from_numpy(new_static_pts), torch.from_numpy(new_static_color)
        static_pts = torch.cat([static_pts, new_static_pts])
        static_colors = torch.cat([static_colors, new_static_color])

    if args.remove_ids:
        dynamic_mask = torch.ones(len(cluster_fv), dtype=torch.bool)
        dynamic_mask[args.remove_ids] = False
        for i in args.remove_ids:
            cluster_points_dict.pop(i, None)
            cluster_color_dict.pop(i, None)

        cluster_fv = cluster_fv[dynamic_mask]
        cluster_deform = cluster_deform[dynamic_mask]

        i = 0
        new_pts, new_colors = {}, {}
        for k, v in cluster_points_dict.items():
            new_pts[i] = v
            new_colors[i] = cluster_color_dict[k]
            i += 1
        cluster_points_dict = new_pts
        cluster_color_dict = new_colors

    instance_bbox = {}
    for k, v in cluster_points_dict.items():
        instance_bbox[k] = v.abs().max(0)[0]

    ckpt = {
        'cluster_points_dict': cluster_points_dict,
        'cluster_color_dict': cluster_color_dict,
        'cluster_deform': cluster_deform,
        'cluster_fv': cluster_fv,
        'static_means': static_pts,
        'static_colors': static_colors,
        'instance_bbox': instance_bbox
    }

    save_path = os.path.join(output_folder, 'checkpoint.pt')
    torch.save(ckpt, save_path)
    print(f"Saved clustering checkpoint to {save_path}")

    points, ins, color = transform_canonical(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv)
    visualize_cluster(points, ins, color, vis_output_folder, 5)
    vis_every_cluster(cluster_points_dict, cluster_color_dict, vis_output_folder)


if __name__ == "__main__":
    main()
