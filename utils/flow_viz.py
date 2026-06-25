import numpy as np
from collections import namedtuple
import itertools
from typing import Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Union
from matplotlib import cm
from matplotlib.pyplot import get_cmap
import open3d as o3d
import torch
from viser import ViserServer

import plotly.graph_objects as go

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, itertools.accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def color_by_distance(distances: np.ndarray, max_distance: float = 10.0, cmap: str = "viridis"):
    # Use distance to color points, normalized to [0, 1].
    colors = distances.copy()

    # Normalize to [0, 1]
    colors = colors / max_distance
    colors[colors > 1] = 1.0

    colormap = plt.get_cmap(cmap)
    colors = colormap(colors)[:, :3]
    return colors

def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = _make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image

def flow_to_color_2d(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float] = 2.0,
    background: Optional[str] = "bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )
    wheel = _make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional)
        + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs", ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"]
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - np.expand_dims(factors, -1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float32)
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float32)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array = None,
    colors: np.array = None,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
    black_bg: bool = False,
    title: str = None,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    if coords is not None:
        # Add static trace
        static_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=colors,
                symbol="square",
            ),
        )
        fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        if coords is not None:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords) + 1),
                    ],  # Include the static trace
                    label=f"Second {i}",
                )
                step["args"][1][0] = True  # Make the static trace always visible
                step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)
        else:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords)),
                    ],
                    label=f"Second {i}",
                )
                step["args"][1][i] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)

        sliders = [
            dict(
                active=0,
                pad={"t": 1},
                steps=steps,
                font=dict(color="white") if black_bg else {},  # Update for font color
            )
        ]
        fig.update_layout(sliders=sliders)
    title_font_color = "white" if black_bg else "black"
    if not black_bg:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            paper_bgcolor="black",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    eye = np.array([-1, 0, 0.5])
    eye = eye.tolist()
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        ),
    )
    return fig


def map_colors(values: np.ndarray, colormap=cm.gist_rainbow, min_value=None, max_value=None):

    assert callable(colormap) or isinstance(colormap, np.ndarray)
    if min_value is None:
        min_value = values[np.isfinite(values)].min()
    if max_value is None:
        max_value = values[np.isfinite(values)].max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, np.ndarray)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = np.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def visualize_points3D(x, value=None, normals=None, min=None, max=None, colormap=cm.jet, vis=True):
    assert x.ndim == 2
    assert x.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    if value is not None:
        assert len(value) == len(x)
        if isinstance(value, torch.Tensor):
            value = value.float()
            value = value.detach().cpu().numpy()
        assert isinstance(value, np.ndarray)
        value = np.asarray(value, dtype=float)
        if value.ndim == 2:
            assert value.shape[1] == 3
            colors = value
        elif value.ndim == 1:
            colors = map_colors(value, colormap=colormap, min_value=min, max_value=max)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label = None,
    name: str = "tracks",
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    colors = cmap(np.asarray(vis_label))[:, :3]
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="diamond",
        )