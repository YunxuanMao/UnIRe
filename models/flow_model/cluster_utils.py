import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
import torch
from torch_scatter import scatter
from utils.flow_viz import vis_occ_plotly, map_colors
from copy import deepcopy


def compute_position_distance(positions, mask=None):
    num_points, num_frames = positions.shape[0], positions.shape[1]
    distances = np.zeros((num_points, num_points))
    # for i in range(num_points):
    #     for j in range(num_points):
    #         distances[i, j] = torch.norm(trajectories[i] - trajectories[j])
    for t in range(num_frames):
        if mask is not None:
            mask_i = np.matmul(mask[:, t], mask[:, t].T)
            distances += np.linalg.norm(positions[None, :, t, :] - positions[:, None, t, :], axis=-1) * mask_i
        else:
            distances += np.linalg.norm(positions[None, :, t, :] - positions[:, None, t, :], axis=-1)
    if mask is not None:
        counts = mask.sum(axis=1)
        num_counts = np.sqrt(np.matmul(counts, counts.T))
        distances = distances / (num_counts + 1e-6)
    else:
        distances = distances / num_frames
    return distances



# 计算运动方向的相似性（余弦相似度）
def compute_direction_similarity(trajectories, mask=None):
    num_points, num_frames = trajectories.shape[0], trajectories.shape[1]
    total_similarity = np.zeros((num_points, num_points))
    # if mask is not None:
    #     num_counts = torch.zeros((num_points, num_points), dtype=torch.long).to(trajectories.device)

      # 每个点的总位移方向
    directions_normalized = trajectories / (np.linalg.norm(trajectories, axis=-1, keepdims=True) + 1e-6)  # 单位化
    for t in range(num_frames - 1):
        similarities = np.matmul(directions_normalized[:, t], directions_normalized[:, t].T)  # 方向相似性矩阵
        if mask is not None:
            mask_i = np.matmul(mask[:, t], mask[:, t].T)
            total_similarity += (similarities + 1) / 2 * mask_i
        else:
            total_similarity += (similarities + 1) / 2


    if mask is not None:
        counts = mask.sum(axis=1)
        num_counts = np.sqrt(np.matmul(counts, counts.T))
        avg_similarity = total_similarity / (num_counts + 1e-6)
    else:
        avg_similarity = total_similarity / (num_frames - 1)

    return avg_similarity


def cluster_by_similarity(positions, mask=None, eps = 0.2, min_samples = 1, lambda_dir=0.5):
    position_distances = compute_position_distance(positions, mask)
    position_similarity = np.exp(-position_distances)
    direction_similarity = compute_direction_similarity(positions, mask)
    # alpha = 0.5  # 权重参数
    global_similarity =  lambda_dir * direction_similarity + (1-lambda_dir) * position_similarity
    distance_matrix = 1 - global_similarity

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # 使用相似性矩阵
    # dbscan = HDBSCAN(min_cluster_size=5, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)

    return labels


def align_id_clusters(c1, c2, cmax, nn=None, nn_b=None, dist=None, dist_b=None, trunc=0.5, forward=True, backward=True, points_trunc=100):
    # check if c1 seperate in c2
    changed_cid = {}
    if forward:
        assert c1.max() < c2.min()
        assert nn is not None and dist is not None
        valid_mask = dist.squeeze() < trunc
        for clu_id in c1.unique():
            matched = c2[nn[0,valid_mask,0][c1[valid_mask] == clu_id]]
            matched_ids, counts = torch.unique(matched, return_counts = True) 
            if len(matched_ids) == 1:
                continue
            mask = matched_ids != -1
            matched_ids = matched_ids[mask]

            if len(matched_ids) == 0:
                continue
            
            if len(matched_ids) > 1:
                best_match_idx = torch.argmax(counts)
                best_match_id = matched_ids[best_match_idx] 


                c1_valid = c1[valid_mask]
                c1_i = c1_valid[c1_valid == clu_id]
                # not best match -> new ids
                for i in range(len(matched_ids)):
                    if counts[i] > points_trunc and matched_ids[i] != best_match_id:
                        c1_i[matched == matched_ids[i]] = cmax.clone()
                        changed_cid[cmax.clone()] = clu_id
                        cmax += 1
                        c2 += 1
                        
                c1_valid[c1_valid == clu_id] = c1_i
                c1[valid_mask] = c1_valid

    
    # align c1 id to c2
    if backward:
        assert c1.max() < c2.min()
        assert nn_b is not None and dist_b is not None
        valid_mask = dist_b.squeeze() < trunc

        for clu_id in c2.unique():

            # divne, znovu
            # TODO - DISTANCE threshold?
            matched = c1[nn_b[0,valid_mask,0][c2[valid_mask] == clu_id]] # t1时间指定cluster id的点在t2时间对应点的cluster id

            matched_ids, counts = torch.unique(matched, return_counts = True) 
            

            # match only itself
            if len(matched_ids) == 1 and matched_ids[0] == clu_id:
                continue
            mask = matched_ids != -1
            matched_ids = matched_ids[mask]

            if len(matched_ids) == 0:
                continue


            counts = counts[mask]
            best_match_idx = torch.argmax(counts)
            best_match_id = matched_ids[best_match_idx] 
            if len(matched_ids) == 1:
                best_match_counts = counts[best_match_idx]
                c2[c2==clu_id] = best_match_id
            
            elif len(matched_ids) > 1:
                c2_valid = c2[valid_mask]
                c2_i = c2_valid[c2_valid == clu_id]
                for i in range(len(matched_ids)):
                    if counts[i] > points_trunc:
                        c2_i[matched == matched_ids[i]] = matched_ids[i]
                    else:
                        c2_i[matched == matched_ids[i]] = best_match_id
                c2_valid[c2_valid == clu_id] = c2_i
                c2[valid_mask] = c2_valid

    
    return c1, c2, cmax, changed_cid

def align_changed_id(points_list, ins_list, flow_list, changed_ids):
    points_list, ins_list, flow_list = points_list[::-1], ins_list[::-1], flow_list[::-1]
    for new_id,old_id in changed_ids.items():
        for i in range(len(flow_list)):
            if (ins_list[i+1]==old_id).any():
                p1 = torch.cat([points_list[i][ins_list[i]==new_id], points_list[i][ins_list[i]==old_id]])
                p2 = points_list[i+1][ins_list[i+1]==old_id]
                flow = flow_list[i][ins_list[i+1]==old_id]
                new_ins = torch.cat([ins_list[i][ins_list[i]==new_id], ins_list[i][ins_list[i]==old_id]])
                old_ins = ins_list[i+1][ins_list[i+1]==old_id]

                # dist, nn, _ = knn_points(p1[None], p2[None]+flow[None], lengths1=None, lengths2=None, K=1, return_nn=True)
                dist_b, nn_b, _ = knn_points(p2[None]+flow[None], p1[None], lengths1=None, lengths2=None, K=1, return_nn=True)
                mask = new_ins[nn_b.squeeze()] == new_id
                old_ins[mask] = new_id
                ins_list[i+1][ins_list[i+1]==old_id] = old_ins



def associate_clusters(points_list, ins_list, flow_list, cluster_trunc):
    global_points_list = [points_list[0]]
    global_ins_list = [ins_list[0].clone()]
    nn_list = []
    dist_list = []
    nn_b_list = []

    

    for i in tqdm(range(len(flow_list))):
        p1 = points_list[i]
        p2 = points_list[i+1]
        if i == 0:
            c1 = ins_list[i]
            c2 = ins_list[i+1]
            cid = c1.min()
            old_c1 = c1.clone()
            c1_unique = c1.unique()
            for cid_i in c1_unique:
                c1[old_c1==cid_i] = cid
                cid += 1
            cmax = cid

        else:
            c1 = c_next.clone()
            c2 = ins_list[i+1].clone()

        flow = flow_list[i]
        # c2 += cmax

        dist, nn, _ = knn_points(p1[None]+flow[None], p2[None], lengths1=None, lengths2=None, K=1, return_nn=True)
        dist_b, nn_b, _ = knn_points(p2[None], p1[None]+flow[None], lengths1=None, lengths2=None, K=1, return_nn=True)
        
        nn_list.append(nn)
        dist_list.append(dist)

        # exist_mask = dist.squeeze() < cluster_trunc
        align_valid_mask = dist_b.squeeze() < cluster_trunc

        # exist_cluster = c1[exist_mask].unique()
        
        c1, c_next, cmax, changed_id = align_id_clusters(c1, c2 + cmax, cmax, nn, nn_b, dist, dist_b, cluster_trunc)
        new_points_cluster = c_next[~align_valid_mask]
        new_clusters = new_points_cluster[new_points_cluster >= cmax].unique()

        global_ins_list[-1] = c1.clone()
        if len(changed_id) > 0:
            align_changed_id(global_points_list, global_ins_list, flow_list[:i], changed_id)

        if len(new_clusters) > 0:
            # assert len(new_clusters) < 10
            for cid in new_clusters:
                c_next[c_next == cid] = cmax
                cmax += 1

        
        
        global_ins_list.append(c_next)
        global_points_list.append(p2)
    

    return global_ins_list, global_points_list


def associate_clusters_inverse(points_list, ins_list, flow_list, cluster_trunc):
    
    points_list, ins_list, flow_list = points_list[::-1], ins_list[::-1], flow_list[::-1]
    global_points_list = [points_list[0]]
    global_ins_list = [ins_list[0]]
    
    # forward
    for i in tqdm(range(len(flow_list))):
        p1 = points_list[i]
        p2 = points_list[i+1]
        if i == 0:
            c1 = ins_list[i]
            c2 = ins_list[i+1]
            cid = c1.min()
            c1_unique = c1.unique()
            for cid_i in c1_unique:
                c1[c1==cid_i] = cid
                cid += 1
            cmax = cid

        else:
            c1 = c_next
            c2 = ins_list[i+1]
        flow = flow_list[i]
        c2 += cmax

        dist, nn, _ = knn_points(p1[None], p2[None]+flow[None], lengths1=None, lengths2=None, K=1, return_nn=True)
        dist_b, nn_b, _ = knn_points(p2[None]+flow[None], p1[None], lengths1=None, lengths2=None, K=1, return_nn=True)
        
        
        # exist_mask = dist.squeeze() < cluster_trunc
        align_valid_mask  = dist_b.squeeze() < cluster_trunc

        # exist_cluster = c1[exist_mask].unique()

        
        c1, c_next, cmax = align_id_clusters(c1, c2, cmax, nn, nn_b, dist, dist_b, cluster_trunc, forward = False)
        new_points_cluster = c_next[~align_valid_mask]
        new_clusters = new_points_cluster[new_points_cluster >= cmax].unique()
        if len(new_clusters) > 0:
            # assert len(new_clusters) < 10
            for cid in new_clusters:
                c_next[c_next == cid] = cmax
                cmax += 1
        
        
        global_ins_list.append(c_next)
        global_points_list.append(p2)
    

    global_ins_list = global_ins_list[::-1]
    global_points_list = global_points_list[::-1]

    return global_ins_list, global_points_list


def get_canonical(points_list, cluster_list, flow_list, color_list, valid_num_cluster_points=10, valid_fv = 5, device="cuda:0", deform_flow=True):
    num_t = len(points_list)
    cluster_fv = np.zeros([3000, num_t], dtype=np.bool_)
    cluster_means = np.zeros([3000, num_t, 3])
    cluster_flow = np.zeros([3000, num_t, 3])
    num_cluster_points = np.zeros([3000, num_t,])


    for i in range(num_t):
        cluster = np.unique(cluster_list[i])
        cluster_max = cluster.max()
        cluster_torch = torch.from_numpy(cluster_list[i]).to(device)
        num_cluster_points_i = scatter(torch.ones_like(cluster_torch), cluster_torch, dim=0, reduce='sum')
        cluster_means_i = scatter(torch.from_numpy(points_list[i]).to(device), cluster_torch, dim=0, reduce='mean')
        cluster_means[:cluster_max+1, i] = cluster_means_i.cpu().numpy()
        if i < num_t - 1:
            cluster_flow_i = \
                scatter(torch.from_numpy(flow_list[i]).to(device), cluster_torch, dim=0, reduce='mean')
            cluster_flow[:cluster_max+1, i+1] = cluster_flow_i.cpu().numpy()

        cluster_fv[cluster, i] = True
        num_cluster_points[:cluster_max+1, i] = num_cluster_points_i.cpu().numpy()


    num_cluster_points_max = num_cluster_points.max(-1)
    t_cluster_points = num_cluster_points.argmax(-1)
    fv_cluster_points_max = cluster_fv.sum(-1)

    valid_cluster_id = np.where(np.logical_and(num_cluster_points_max > valid_num_cluster_points, fv_cluster_points_max > valid_fv))[0]
    id_map = {id_value: idx for idx, id_value in enumerate(valid_cluster_id)}
    
    cluster_fv = cluster_fv[valid_cluster_id]
    t_cluster_points = t_cluster_points[valid_cluster_id]
    
    for i in tqdm(range(num_t)):
        p = points_list[i]
        c = cluster_list[i]
        valid_mask = np.isin(c, valid_cluster_id)
        c_new = np.zeros_like(c)
        for id_value in id_map:
            if (c==id_value).any():
                c_new[c==id_value] = id_map[id_value]
        cluster_list[i] = c_new[valid_mask]
        points_list[i] = p[valid_mask]
        color_list[i] = color_list[i][valid_mask]
        if i < num_t - 1:
            flow_list[i] = flow_list[i][valid_mask]

    if deform_flow:
        cluster_flow = cluster_flow[valid_cluster_id]
        cluster_deform = np.cumsum(cluster_flow, axis=1)
    else:
        cluster_means = cluster_means[valid_cluster_id]
        # cluster_deform = cluster_means - cluster_means[:, :1]
        cluster_deform = cluster_means
    # get cluster canonical space
    cluster_points_dict = {}
    cluster_color_dict = {}
    canonical_space = []
    canonical_space_ins = []
    canonical_space_color = []
    for cid in valid_cluster_id:
        idx = id_map[cid]
        t = t_cluster_points[idx]
        points_i = points_list[t][cluster_list[t]==idx] - cluster_deform[idx, t]
        color_i = color_list[t][cluster_list[t]==idx]
        cluster_points_dict[idx] = points_i
        cluster_color_dict[idx] = color_i
        canonical_space.append(points_i)
        canonical_space_ins.append(np.ones(len(points_i), dtype=np.int32) * idx)
        canonical_space_color.append(color_i)

    canonical_space = np.concatenate(canonical_space)
    canonical_space_ins = np.concatenate(canonical_space_ins)
    canonical_space_color = np.concatenate(canonical_space_color)

    # points_list, cluster_list, color_list = transform_canonical(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv)

    ret_dict = {
        "cluster_points_dict": cluster_points_dict,
        "cluster_color_dict": cluster_color_dict,
        "cluster_deform": cluster_deform,
        "cluster_fv": cluster_fv, 
        "points_list": points_list, 
        "cluster_list": cluster_list, 
        "color_list": color_list, 
        "flow_list": flow_list,
        "t_cluster_points": t_cluster_points
    }

    return ret_dict


def recluster(cluster_points_dict, cluster_deform, cluster_fv, eps=0.6, ):


    num_instance, num_t = cluster_fv.shape[:2]
    cluster_means = np.zeros([num_instance, num_t, 3])
    for i in range(num_t):
        for idx in np.where(cluster_fv[:, i])[0]:
            p_idx = cluster_points_dict[idx] + cluster_deform[idx, i]
            cluster_means[idx, i] = p_idx.mean(0)

    clusters = cluster_by_similarity(cluster_means * (1., 1., 0.5), cluster_fv, eps=eps, min_samples=1)
    clusters_unique = np.unique(clusters)
    print(f"recluster results: {len(clusters_unique)}")
    
    return clusters


def transform_canonical(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv):
    points_list_numpy = []
    cluster_list_numpy = []
    color_list_numpy = []

    num_instance, num_t = cluster_fv.shape[:2]
    for i in range(num_t):
        p_i = []
        c_i = []
        color_i = []
        for idx in np.where(cluster_fv[:, i])[0]:
            p_idx = cluster_points_dict[idx] + cluster_deform[idx, i]
            p_i.append(p_idx)
            # p_i.append(cluster_points_dict[idx] - cluster_means[idx, t] + cluster_means[idx, i])
            c_i.append(np.ones(len(cluster_points_dict[idx]), dtype=np.int32) * idx)
            color_i.append(cluster_color_dict[idx])
        p_i = np.concatenate(p_i)
        c_i = np.concatenate(c_i)
        color_i = np.concatenate(color_i)
        points_list_numpy.append(p_i)
        cluster_list_numpy.append(c_i)
        color_list_numpy.append(color_i)

    return points_list_numpy, cluster_list_numpy, color_list_numpy


def run_cluster_init(points_list, cluster_list, color_list, flow_list, device="cuda:0", cluster_trunc=0.2, valid_num_cluster_points=100, eps=0.5, recluster_time = 1,static_trunc=0.02, seperate = True):
    num_t = len(points_list)

    for i in range(num_t):
        points_list[i] = points_list[i].to(device)
        cluster_list[i] = cluster_list[i].to(device)
        color_list[i] = color_list[i].to(device)
        if i < num_t - 1:
            flow_list[i] = flow_list[i].to(device)

    print('Run associate clusters...')
    cluster_list, _ = associate_clusters(points_list, cluster_list, flow_list, cluster_trunc)
    print(f"Cluster max: {torch.cat(cluster_list).max().item()}, Cluster num: {len(torch.cat(cluster_list).unique())}")


    print('Get canonical and recluster')

    points_list_numpy = [p_i.cpu().numpy() for p_i in points_list]
    cluster_list_numpy = [c_i.cpu().numpy() for c_i in cluster_list]
    color_list_numpy = [color_i.cpu().numpy() for color_i in color_list]
    flow_list_numpy = [flow_i.cpu().numpy() for flow_i in flow_list]
    new_static_pts, new_static_color = None, None

    # for i in range(recluster_time):
    ret = get_canonical(points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy, valid_num_cluster_points, device = device, deform_flow=True if seperate else False)
    cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv = \
        ret['cluster_points_dict'], ret['cluster_color_dict'], ret['cluster_deform'], ret['cluster_fv']
    points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy = \
        ret["points_list"], ret["cluster_list"], ret["flow_list"], ret["color_list"]

    clusters = recluster(cluster_points_dict, cluster_deform, cluster_fv, eps)

    for j in range(len(cluster_list_numpy)):
        cluster_list_numpy[j] = clusters[cluster_list_numpy[j]]

    ret = get_canonical(points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy, valid_num_cluster_points, device = device, deform_flow=True if seperate else False)

    cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv = \
            ret['cluster_points_dict'], ret['cluster_color_dict'], ret['cluster_deform'], ret['cluster_fv']

    
    if seperate:
        points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy = \
            ret["points_list"], ret["cluster_list"], ret["flow_list"], ret["color_list"]

        new_static_pts, new_static_color, cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv, dynamic_mask = \
            seperate_static(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv, static_trunc=static_trunc)
        
        for i in range(len(points_list_numpy)):
            dynamic_mask_i = dynamic_mask[cluster_list_numpy[i]]
            points_list_numpy[i] = points_list_numpy[i][dynamic_mask_i]
            cluster_list_numpy[i] = cluster_list_numpy[i][dynamic_mask_i]
            color_list_numpy[i] = color_list_numpy[i][dynamic_mask_i]
            if i < len(points_list_numpy) - 1:
                flow_list_numpy[i] = flow_list_numpy[i][dynamic_mask_i]

        
        ret = get_canonical(points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy, valid_num_cluster_points, device = device, deform_flow=False)
        cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv = \
            ret['cluster_points_dict'], ret['cluster_color_dict'], ret['cluster_deform'], ret['cluster_fv']
        points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy = \
            ret["points_list"], ret["cluster_list"], ret["flow_list"], ret["color_list"]

        clusters = recluster(cluster_points_dict, cluster_deform, cluster_fv, eps)

        for j in range(len(cluster_list_numpy)):
            cluster_list_numpy[j] = clusters[cluster_list_numpy[j]]

        ret = get_canonical(points_list_numpy, cluster_list_numpy, flow_list_numpy, color_list_numpy, valid_num_cluster_points, device = device, deform_flow=False)

        cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv = \
                ret['cluster_points_dict'], ret['cluster_color_dict'], ret['cluster_deform'], ret['cluster_fv']
        
    cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv = cluster_in_cluster(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv)

    # points_list_torch = [torch.from_numpy(p_i) for p_i in points_list_numpy]
    # cluster_list_torch = [torch.from_numpy(c_i) for c_i in cluster_list_numpy]
    # color_list_torch = [torch.from_numpy(color_i) for color_i in color_list_numpy]

    
    return cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv, new_static_pts, new_static_color


def cluster_in_cluster(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv):
    max_cluster_id = len(cluster_points_dict)
    cluster_points_dict_old = deepcopy(cluster_points_dict)
    for k, v in cluster_points_dict_old.items():
        dbscan = DBSCAN(eps=0.5, min_samples=16)
        labels = dbscan.fit_predict(v)
        new_cluster = np.unique(labels)
        new_cluster = new_cluster[new_cluster>=0]
        if len(new_cluster > 1):
            color = cluster_color_dict[k]
            for cid in new_cluster:
                if cid == 0:
                    cluster_points_dict[k] = v[labels==0]
                    cluster_color_dict[k] = color[labels==0]
                else:
                    cluster_points_dict[max_cluster_id] = v[labels==cid]
                    cluster_color_dict[max_cluster_id] = color[labels==cid]
                    cluster_deform = np.concatenate((cluster_deform, cluster_deform[k:k+1]))
                    cluster_fv = np.concatenate((cluster_fv, cluster_fv[k:k+1]))
                    max_cluster_id += 1
    return cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv




def visualize_cluster(points, ins, color, output_folder, vis_every = 5):
    

    
    num_frame = len(points)
    vis_points = []
    vis_ins = []
    vis_color = []
    for i in range(0, num_frame - 1, vis_every):
        vis_points.append(points[i])
        vis_ins.append(ins[i])
        vis_color.append(color[i])

    ins_min = np.min(np.concatenate(vis_ins))
    ins_max = np.max(np.concatenate(vis_ins))


    vis_ins_colors = []

    for i in range(len(vis_points)):
        ins_color = map_colors(vis_ins[i], min_value=ins_min, max_value=ins_max)
        vis_ins_colors.append(ins_color)


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

    output_path = f"{output_folder}/points_color.html"
    color_figure.write_html(output_path)

    
def vis_every_cluster(cluster_points_dict, cluster_color_dict, output_folder,):
    points = []
    colors = []

    for k, v in cluster_points_dict.items():
            points.append(v.cpu().numpy())
            colors.append(cluster_color_dict[k].cpu().numpy())

    aabb_max = np.max(np.concatenate(points), 0)
    aabb_min = np.min(np.concatenate(points), 0)
    aabb = np.concatenate([aabb_min, aabb_max])
    aabb_length = aabb_max - aabb_min

    color_figure = vis_occ_plotly(
            vis_aabb=aabb.tolist(),
            dynamic_coords=points,
            dynamic_colors=colors,
            x_ratio=1,
            y_ratio=(aabb_length[1] / aabb_length[0]),
            z_ratio=(aabb_length[2] / aabb_length[0]),
            size=2,
            black_bg=True,
            title=f"color",
        )

    output_path = f"{output_folder}/pred_instance_every.html"
    color_figure.write_html(output_path)


def seperate_static(cluster_points_dict, cluster_color_dict, cluster_deform, cluster_fv, static_trunc=0.02):
    cluster_flow = cluster_deform[:, 1:] - cluster_deform[:, :-1]
    flow_valid_mask = np.logical_and(cluster_fv[:, 1:], cluster_fv[:, :-1])
    cluster_flow[..., -1] = 0
    # flows_means = (cluster_flow.norm(dim = -1) * flow_valid_mask).mean(-1)
    flows_means = (np.linalg.norm(cluster_flow, axis=-1) * flow_valid_mask).sum(-1)/flow_valid_mask.sum(-1)
    static_mask = flows_means < static_trunc
    dynamic_mask = ~static_mask

    new_static_pts = []
    new_static_color = []
    for i in np.where(static_mask)[0].tolist():
        new_static_pts.append(cluster_points_dict[i])
        new_static_color.append(cluster_color_dict[i])
    
    idx = 0
    cluster_points_dict_new = {}
    cluster_color_dict_new = {}
    for i in np.where(dynamic_mask)[0].tolist():
        cluster_points_dict_new[idx] = cluster_points_dict[i]
        cluster_color_dict_new[idx] = cluster_color_dict[i]
        idx += 1
    
    print(f'max dynamic idx: {idx}')
    new_static_pts = np.concatenate(new_static_pts)
    new_static_color = np.concatenate(new_static_color)

    cluster_deform = cluster_deform[dynamic_mask]
    cluster_fv = cluster_fv[dynamic_mask]

    return new_static_pts, new_static_color, cluster_points_dict_new, cluster_color_dict_new, cluster_deform, cluster_fv, dynamic_mask