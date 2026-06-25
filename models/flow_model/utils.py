# Imports and data
import numpy as np
import torch

from sklearn.cluster import DBSCAN, HDBSCAN
from pytorch3d.ops.knn import knn_points
from torch_scatter import scatter

from models.flow_model import sc_utils
    

def pass_id_clusters(c1, c2, nn, valid_mask = None, lock_c1 = False):
    # dist, nn, _ = knn_points(p1 + f1, p2, K=1)

    for clu_id in range(c1.max() + 1):

        # divne, znovu
        # TODO - DISTANCE threshold?
        if valid_mask is not None:
            matched = c2[nn[0,valid_mask,0][c1[valid_mask] == clu_id]] # t1时间指定cluster id的点在t2时间对应点的cluster id

        else:
            matched = c2[nn[0,:,0][c1 == clu_id]] # t1时间指定cluster id的点在t2时间对应点的cluster id

        
        matched_ids = torch.unique(matched)   
        # print(matched_ids)
        
        # TODO - time matching across full sequence?
        # TODO - check it properly

        # match only itself
        if len(matched_ids) == 1 and matched_ids[0] == clu_id:
            continue
        
        matched_ids = matched_ids[matched_ids != -1]

        if len(matched_ids) == 0:
            continue
        
        if not lock_c1:
            if clu_id <= torch.min(matched_ids):
                chosen_id = clu_id
            else:
                chosen_id = torch.min(matched_ids)
            

            c1[c1 == clu_id] = chosen_id

            for m_id in matched_ids:    

                c2[c2 == m_id] = chosen_id
        
        else:
            for m_id in matched_ids:    

                c2[c2 == m_id] = clu_id

    return c1


def align_id_clusters(c1, c2, nn, valid_mask = None,):
    # align c1 id to c2

    # dist, nn, _ = knn_points(p1 + f1, p2, K=1)
    match_dict = {}
    for clu_id in range(c2.max() + 1):

        # divne, znovu
        # TODO - DISTANCE threshold?
        if valid_mask is not None:
            matched = c1[nn[0,valid_mask,0][c2[valid_mask] == clu_id]] # t1时间指定cluster id的点在t2时间对应点的cluster id

        else:
            matched = c1[nn[0,:,0][c2 == clu_id]]
        
        matched_ids, counts = torch.unique(matched, return_counts = True) 
        
        # print(matched_ids)
        
        # TODO - time matching across full sequence?
        # TODO - check it properly

        # match only itself
        if len(matched_ids) == 1 and matched_ids[0] == clu_id:
            continue
        mask = matched_ids != -1
        matched_ids = matched_ids[mask]

        if len(matched_ids) == 0:
            continue


        counts = counts[mask]
        best_match_idx = torch.argmax(counts)
        best_match_counts = counts[best_match_idx]
        best_match_id = matched_ids[best_match_idx]

        # if best_match_counts < 10:
        #     continue
        
        
        c2[c2==clu_id] = best_match_id
        match_dict[clu_id] = best_match_id

    return c2, match_dict


class SC2_KNN_cluster_aware(torch.nn.Module):
    
    ''' Our soft-rigid regularization with neighborhoods
    pc1 : Point cloud
    K : Number of NN for the neighborhood
    use_normals : Whether to use surface estimation for neighborhood construction
    d_thre : constant for working with the displacements as percentual statistics, we use value from https://github.com/ZhiChen902/SC2-PCR
    '''
    def __init__(self, pc1, K=16, d_thre=0.03):
        super().__init__()
        self.d_thre = d_thre
        self.K = K
        dist, self.kNN, _ = knn_points(pc1, pc1, lengths1=None, lengths2=None, K=K, return_nn=True)
    
        
        self.src_keypts = pc1[:, self.kNN[:, :, :]]


    def forward(self, flow, mask = None, ids=None):

        target_keypts = self.src_keypts + flow[:, self.kNN[:, :, :]]
        target_keypts = target_keypts[0, 0]
        src_keypts = self.src_keypts[0, 0]


        target_keypts = src_keypts + flow[:, self.kNN[:, :, :]][0,0]

        if mask is not None:
            src_keypts = src_keypts[mask]
            target_keypts = target_keypts[mask]
        

        src_dist = (src_keypts[:, :, None, :] - src_keypts[:, None, :, :]).norm(dim=-1)
        target_dist = (target_keypts[:, :, None, :] - target_keypts[:, None, :, :]).norm(dim=-1)
        cross_dist = (src_dist - target_dist).abs()
        A = torch.clamp(1.0 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        

        leading_eig = sc_utils.power_iteration(A)

        sc2_rigidity = sc_utils.spatial_consistency_score(A, leading_eig)
        
        # if apply_sc2: # for ablation study
        loss = - torch.log(sc2_rigidity).mean()


        return loss

def center_rigidity_loss(pc1, flow, cluster_ids):
    '''
    For batch size of 1
    :param pc1: 
    :param flow: 
    :param cluster_ids: 
    :return: 
    '''
    pts_centers = scatter(pc1, cluster_ids, dim=1, reduce='mean')
    flow_centers = scatter(pc1 + flow, cluster_ids, dim=1, reduce='mean')
    
    pt_dist_to_center = (pc1 - pts_centers[0, cluster_ids[0]].unsqueeze(0))#.norm(dim=-1, p=1)
    flow_dist_to_center = ((pc1 + flow) - flow_centers[0, cluster_ids[0]].unsqueeze(0))#.norm(dim=-1, p=1)
    
    center_displacement = pt_dist_to_center - flow_dist_to_center
    
    rigidity_loss = center_displacement.norm(dim=-1).mean()

    return rigidity_loss, center_displacement



def initial_clustering(global_list, frame, temporal_range, device, eps=0.3, min_samples=1, z_scale=0.5):

    # init clustering
    to_cluster_pc1 = np.concatenate(global_list[frame:frame+temporal_range], axis=0)
    scaled_cluster_pc1 = to_cluster_pc1[:,:3] * (1,1, z_scale)   # scale z-axis
    # Spatio-temporal clustering with fixed temporal range
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_cluster_pc1[:,:3])


    p1 = to_cluster_pc1[to_cluster_pc1[:,3] == frame][None, :,:3]
    p2 = to_cluster_pc1[to_cluster_pc1[:,3] == frame + 1][None, :,:3]

    c1 = clusters[to_cluster_pc1[:,3] == frame]
    c2 = clusters[to_cluster_pc1[:,3] == frame + 1]
    
    f1 = torch.zeros(p1.shape, device=device, requires_grad=True)
    p1 = torch.tensor(p1, device=device, dtype=torch.float32)
    p2 = torch.tensor(p2, device=device, dtype=torch.float32)
    c1 = torch.tensor(c1, device=device)    # clusters are without batch dim
    c2 = torch.tensor(c2, device=device)

    return p1, p2, c1, c2, f1   # points, ids, flow initialed to 0




def compute_position_distance(positions, mask=None):
    num_points, num_frames = positions.shape[0], positions.shape[1]
    distances = torch.zeros((num_points, num_points)).to(positions)
    # for i in range(num_points):
    #     for j in range(num_points):
    #         distances[i, j] = torch.norm(trajectories[i] - trajectories[j])
    for t in range(num_frames):
        if mask is not None:
            mask_i = torch.mm(mask[:, t], mask[:, t].T)
            distances += torch.norm(positions[None, :, t, :] - positions[:, None, t, :], dim=-1) * mask_i
        else:
            distances += torch.norm(positions[None, :, t, :] - positions[:, None, t, :], dim=-1)
    if mask is not None:
        counts = mask.sum(dim=1)
        num_counts = torch.sqrt(torch.mm(counts, counts.T))
        distances = distances / (num_counts + 1e-6)
    else:
        distances = distances / num_frames
    return distances



# 计算运动方向的相似性（余弦相似度）
def compute_direction_similarity(trajectories, mask=None):
    num_points, num_frames = trajectories.shape[0], trajectories.shape[1]
    total_similarity = torch.zeros((num_points, num_points)).to(trajectories)
    # if mask is not None:
    #     num_counts = torch.zeros((num_points, num_points), dtype=torch.long).to(trajectories.device)

      # 每个点的总位移方向
    directions_normalized = trajectories / (torch.norm(trajectories, dim=-1, keepdim=True) + 1e-6)  # 单位化
    for t in range(num_frames - 1):
        similarities = torch.mm(directions_normalized[:, t], directions_normalized[:, t].T).clamp(-1., 1.)  # 方向相似性矩阵
        if mask is not None:
            mask_i = torch.mm(mask[:, t], mask[:, t].T)
            total_similarity += (similarities + 1) / 2 * mask_i
        else:
            total_similarity += (similarities + 1) / 2


    if mask is not None:
        counts = mask.sum(dim=1)
        num_counts = torch.sqrt(torch.mm(counts, counts.T))
        avg_similarity = total_similarity / (num_counts + 1e-6)
    else:
        avg_similarity = total_similarity / (num_frames - 1)

    return avg_similarity


def cluster_by_similarity(positions, mask=None, eps = 0.3,min_samples = 15):
    position_distances = compute_position_distance(positions, mask)
    position_similarity = torch.exp(-position_distances)
    direction_similarity = compute_direction_similarity(positions, mask)
    # alpha = 0.5  # 权重参数
    global_similarity = (direction_similarity + position_similarity) / 2
    distance_matrix = 1 - global_similarity.cpu().detach().numpy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # 使用相似性矩阵
    # dbscan = HDBSCAN(min_cluster_size=5, metric='precomputed')
    labels = torch.from_numpy(dbscan.fit_predict(distance_matrix)).to(positions.device)

    return labels