
from typing import Dict, Tuple, Optional
from torch import Tensor


import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

def getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def euler2matrix(yaw):
    cos = torch.cos(-yaw)
    sin = torch.sin(-yaw)
    rot = torch.eye(3).float().cuda()
    rot[0,0] = cos
    rot[0,2] = sin
    rot[2,0] = -sin
    rot[2,2] = cos
    return rot

def cat_bgfg(bg, fg, only_dynamic=False, only_xyz=False):
    if only_xyz:
        bg_feats = [bg.get_xyz]
    else:
        bg_feats = [bg.get_xyz, bg.get_opacity, bg.get_scaling, bg.get_rotation, bg.get_features, bg.get_3D_features]
    
    output = []
    for fg_feat, bg_feat in zip(fg, bg_feats):
        if fg_feat is None:
            output.append(bg_feat)
        elif only_dynamic:
            output.append(fg_feat)
        else:
            output.append(torch.cat((bg_feat, fg_feat), dim=0))
    
    return output


def cat_all_fg(all_fg, next_fg):
    output = []
    for feat, next_feat in zip(all_fg, next_fg):
        if feat is None:
            feat = next_feat
        else:
            feat = torch.cat((feat, next_feat), dim=0)
        output.append(feat)
    return output


def proj_uv(xyz, K, w2c):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intr = torch.as_tensor(K[:3, :3]).float().to(device)  # (3, 3)
    w2c = torch.tensor(w2c).float().to(device)[:3, :]  # (3, 4)

    c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    i_xyz = (intr @ c_xyz.mT).mT  # (N, 3)
    uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3) # (N, 2)
    return uv

def get_optical_flow(xyz, xyz_prev, w2c, w2c_prev, K):
    uv = proj_uv(xyz, K, w2c)
    prev_uv = proj_uv(xyz_prev, K, w2c_prev)
    delta_uv = uv - prev_uv
    return delta_uv


def unicycle_b2w(timestamp, model):
    # model = unicycle_models[track_id]['model']
    pred = model(timestamp)
    if pred is None:
        return None
    pred_a, pred_b, pred_v, pred_phi, pred_h = pred
    # r = euler_angles_to_matrix(torch.tensor([0, pred_phi-torch.pi, 0]), 'XYZ')
    rt = torch.eye(4).float().cuda()
    rt[:3,:3] = euler2matrix(pred_phi)
    rt[1, 3], rt[0, 3], rt[2, 3] = pred_h, pred_a, pred_b
    return rt

def calculate_v_imp_score(scale, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(scale, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list



def count_render_rasterization_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N, 1]
    
    
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int, 
    
    shs: Tensor = None,  # [N, D] or [N, K, 3]
    colors_precomp: Tensor = None,

    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    other = [],
    **kwargs,
    ):

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    assert C == 1, "Don't support batchsize > 1"
    device = means.device

    screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device="cuda").unsqueeze(0) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    cid = 0
    FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    world_view_transform = viewmats[cid].transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    background = (
        backgrounds[cid]
        if backgrounds is not None
        else torch.zeros(3, device=device)
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0 if sh_degree is None else sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        f_count=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = screenspace_points[0]



    gaussians_count, important_score, rendered_image, radii = rasterizer(
        means3D = means,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = quats,
        )
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "gaussians_count": gaussians_count,
        "important_score": important_score,
    }
