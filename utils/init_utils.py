from typing import Literal, Union

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from utils.flow_viz import visualize_points3D, vis_occ_plotly, flow_to_rgb, map_colors

from models.flow_model import utils
from torch_scatter import scatter
import open3d as o3d

from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d


# def get_road_label(points: torch.Tensor, cam_traj: torch.Tensor, floor_dim: Literal['x','y','z'] = 'z', ego_height: float = 0.,):
#     floor_dim: int = ['x','y','z'].index(floor_dim)
#     other_dims = [i for i in range(3) if i != floor_dim]
#     ret_min_dis = (points[..., None, other_dims] - cam_traj[None, ..., other_dims]).norm(dim=-1).min(dim=-1)
#     floor_at_in_obj = cam_traj[ret_min_dis.indices][..., floor_dim] - ego_height
#     c = points[..., floor_dim] <= floor_at_in_obj
#     return road_label

def get_road_label(points: torch.Tensor, cam_traj: torch.Tensor, floor_dim: Literal['x','y','z'] = 'z', ego_height: float = 0., thres=0.05):
    plane_model, inliers = plane_fitting(points, distance_threshold=thres)
    plane_model = torch.tensor(plane_model).to(points)
    up_normal = normal_plane2point(plane_model, cam_traj[:1])
    sdf_gt = query_sdf(points, plane_model, up_normal)
    road_label = sdf_gt < 0.2
    return road_label


def get_interest_points(points: torch.Tensor, cam_traj: torch.Tensor, interest_range):
    ret_min_dis = (points[:, None] - cam_traj[None, :]).norm(dim=-1).min(dim=-1)
    mask = ret_min_dis.values < interest_range
    return mask



def distance_point2plane(plane_model, points):
    '''
    Calculate the distance between a plane and points
    Params:
        plane_model: torch.tensor [4]
        points: torch.tensor [N_points, 3]
    Returns:
        distant: torch.tensor [N_points,]
    '''
    points = torch.cat([points, torch.ones_like(points[:, :1])], -1)
    distance = torch.sum(points * plane_model, -1) / torch.norm(plane_model[:3])
    return torch.abs(distance)

def plane_fitting(points, distance_threshold):
    '''
    Fit a plane by point cloud
    '''
    if not torch.is_tensor(points):
        points = torch.tensor(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                        ransac_n=3,
                                        num_iterations=1000)

    return plane_model, inliers

def normal_plane2point(plane_model, points):
    '''
    Get a plane normal from plane to point
    Params:
        plane_model: torch.tensor [4]
        points: torch.tensor [N_points, 3]
    Returns:
        up_normal: torch.tensor [N_points, 3]
    '''
    normal = plane_model[:3]
    z = -plane_model[3] / plane_model[2]
    origin = torch.tensor([[0,0,z]]).to(points)
    v = points - origin
    s = torch.sign(torch.sum(v * normal, -1))
    up_normal = s[..., None] * normal

    return up_normal

def query_sdf(query_pts, plane_model, up_normal, trunc = None):
    sdf_gt = distance_point2plane(plane_model, query_pts)
    normals = normal_plane2point(plane_model, query_pts)
    if trunc is not None:
        trunc_mask = sdf_gt > trunc
        sdf_gt[trunc_mask] = trunc
    s = torch.sign(torch.sum(normals * up_normal, -1))
    sdf_gt = s * sdf_gt
    return sdf_gt
        

def vis_init(
        init_means,
        flows_all,
        points_means,
        cluster_fv,
        init_cluster,
        output_folder,
        vis_every=5
):
    points = []
    ins = []
    flows = []
    ins_colors = []
    flow_colors = []
    ins_global = []
    ins_global_colors = []
    change_points = []

    # new_clusters = init_cluster.clone()
    cluster_unique = init_cluster.unique()
    cluster_max = cluster_unique.max()
    delta_means = torch.cumsum(flows_all, dim=1)
    means = init_means[:, None] + delta_means

    for i in range(0, means.shape[1]-1, vis_every):
        cluster_id_i = torch.where(cluster_fv[:, i])[0]
        valid_mask = torch.logical_or(torch.isin(init_cluster, cluster_id_i), init_cluster == -1)

        points.append(points_means[valid_mask, i].detach().cpu().numpy())
        ins.append(init_cluster[valid_mask].detach().cpu().numpy())
        flows.append(flows_all[valid_mask, i+1].detach().cpu().numpy())
        # change_points.append(means[change_mask].detach().cpu().numpy())
        # ins_global.append(new_clusters[valid_mask].detach().cpu().numpy())

    ins_min = np.min(np.concatenate(ins))
    ins_max = np.max(np.concatenate(ins))
    
    print(ins_max, cluster_max)

    for i in range(len(points)):
        ins_color = map_colors(ins[i], min_value=ins_min, max_value=cluster_max.detach().cpu().numpy())
        # ins_global_color = map_colors(ins_global[i], min=ins_min, max=cluster_max.detach().cpu().numpy())
        flow_color = flow_to_rgb(flows[i], None) / 255.
        ins_colors.append(ins_color)
        flow_colors.append(flow_color)
        # ins_global_colors.append(ins_global_color)

    aabb_max = np.max(np.concatenate(points), 0)
    aabb_min = np.min(np.concatenate(points), 0)
    aabb = np.concatenate([aabb_min, aabb_max])
    aabb_length = aabb_max - aabb_min

    ins_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=points,
            dynamic_colors=ins_colors,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=True,
            title=f"instance",
        )

    flow_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=points,
            dynamic_colors=flow_colors,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=True,
            title=f"flow",
        )

    output_path = f"{output_folder}/pred_instance.html"
    ins_figure.write_html(output_path)

    output_path = f"{output_folder}/pred_flow.html"
    flow_figure.write_html(output_path)

    save_path = f"{output_folder}/canonical_space.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(init_means.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(map_colors(init_cluster.cpu().numpy(), min_value=ins_min, max_value=cluster_max.detach().cpu().numpy()))
    o3d.io.write_point_cloud(save_path, pcd)



