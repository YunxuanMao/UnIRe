import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import os

from pytorch3d.ops.knn import knn_points

from models.flow_model import utils
from utils.flow_viz import flow_to_rgb, vis_occ_plotly, map_colors

def inference_flow(p1, p2, 
                   lr, iters=500, device = 'cuda:0', # training param
                   eps=0.3, min_samples=16, # dbscan param
                    sc_w=1.,K=16, d_thre=0.03, # soft rigid loss param
                    dist_w=2., trunc_dist=0.5, # dist loss param
                   ):

    to_cluster_pc1 = np.concatenate([p1, p2], axis=0)
    
    scaled_cluster_pc1 = to_cluster_pc1 * (1,1, 0.5)   # scale z-axis
    # Spatio-temporal clustering with fixed temporal range
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_cluster_pc1[:,:3])


    c1 = clusters[:len(p1)]
    c2 = clusters[len(p1):]
    
    mask1 = c1 >= 0
    mask2 = c2 >= 0
    p1 = p1[None, mask1]
    p2 = p2[None, mask2]
    c1 = c1[mask1]
    c2 = c2[mask2]

    # p1 = p1[None]
    # p2 = p2[None]
    
    f1 = torch.zeros(p1.shape, device=device, requires_grad=True)
    p1 = torch.tensor(p1, device=device, dtype=torch.float32)
    p2 = torch.tensor(p2, device=device, dtype=torch.float32)
    c1 = torch.tensor(c1, device=device)    # clusters are without batch dim
    c2 = torch.tensor(c2, device=device)
    
    
    optimizer = torch.optim.Adam([f1], lr=lr)

    RigidLoss = utils.SC2_KNN_cluster_aware(p1, K=K, d_thre=d_thre)

    for it in tqdm(range(iters)): 
        loss = 0
        
        # torch.cuda.synchronize()
        # t1 = time.time()
        dist, nn, _ = knn_points(p1 + f1, p2, lengths1=None, lengths2=None, K=1, return_nn=True)   
        dist_b, nn_b, _ = knn_points(p2, p1 + f1, lengths1=None, lengths2=None, K=1, return_nn=True)   
        # torch.cuda.synchronize()
        # t2 = time.time() 
        # print(t2-t1)
        loss += dist_w * (dist[dist < trunc_dist].mean() + dist_b[dist_b < trunc_dist].mean())

        sc_loss = RigidLoss(f1)
        
        if sc_w > 0:
            loss += sc_w * sc_loss
            rigid_loss, center_displacement = utils.center_rigidity_loss(p1, f1, c1[None] + 1)
            loss += sc_w * rigid_loss   # + 1 for noise, works!

        loss += f1[..., 2].norm().mean()
        

        if it % 10 == 0 and it != 0:

            valid_mask = torch.logical_and(dist.squeeze() < 0.1, center_displacement.squeeze().norm(dim=-1) < 0.1)
            c1 = utils.pass_id_clusters(c1, c2, nn, valid_mask)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return p1[0].cpu().detach(), p2[0].cpu().detach(), f1[0].cpu().detach(), c1.cpu(), c2.cpu(), torch.from_numpy(mask1), torch.from_numpy(mask2)


def run_flow_initialization(points_list, color_list, cfg, device = "cuda:0"):
    
    print("Run flow initialization")
    points = []
    flows = []
    ins = []
    color = []

    for i in tqdm(range(len(points_list)-1)):
        p1, p2 = points_list[i].cpu().numpy(), points_list[i+1].cpu().numpy()
        p1, p2, f1, c1, c2, mask1, mask2 = inference_flow(p1, p2, **cfg, device=device)

        points.append(p1)
        flows.append(f1)
        ins.append(c1)
        color.append(color_list[i][mask1].cpu())

    points.append(p2)
    ins.append(c2)
    color.append(color_list[i+1][mask2].cpu())

    return points, flows, ins, color


def visualize_flow(points, flows, ins, color, output_folder, vis_every = 5):
    num_frame = len(points)
    vis_points = []
    vis_flows = []
    vis_ins = []
    vis_color = []
    for i in range(100, num_frame - 1, vis_every):
        vis_points.append(points[i].cpu().numpy())
        vis_ins.append(ins[i].cpu().numpy())
        vis_color.append(color[i].cpu().numpy())
        vis_flows.append(flows[i].cpu().numpy())

    ins_min = np.min(np.concatenate(vis_ins))
    ins_max = np.max(np.concatenate(vis_ins))

    vis_flow_colors = []
    vis_ins_colors = []

    for i in range(len(vis_points)):
        ins_color = map_colors(vis_ins[i], min_value=ins_min, max_value=ins_max)
        # ins_global_color = map_colors(ins_global[i], min=ins_min, max=cluster_max.detach().cpu().numpy())
        flow_color = flow_to_rgb(vis_flows[i], None) / 255.
        vis_ins_colors.append(ins_color)
        vis_flow_colors.append(flow_color)
        # ins_global_colors.append(ins_global_color)

    aabb_max = np.max(np.concatenate(vis_points), 0)
    aabb_min = np.min(np.concatenate(vis_points), 0)
    aabb = np.concatenate([aabb_min, aabb_max])
    aabb_length = aabb_max - aabb_min

    ins_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=vis_points,
            dynamic_colors=vis_ins_colors,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=False,
            title=f"instance",
        )

    flow_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=vis_points,
            dynamic_colors=vis_flow_colors,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=True,
            title=f"flow",
        )
    
    color_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=vis_points,
            dynamic_colors=vis_color,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=True,
            title=f"color",
        )

    output_path = f"{output_folder}/pred_instance.html"
    ins_figure.write_html(output_path)

    output_path = f"{output_folder}/pred_flow.html"
    flow_figure.write_html(output_path)

    output_path = f"{output_folder}/points_color.html"
    color_figure.write_html(output_path)
