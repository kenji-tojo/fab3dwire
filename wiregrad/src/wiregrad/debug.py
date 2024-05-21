from typing import Union, Tuple, List
import torch

from . import _m


def intersect_plane_polyline(
    center: torch.Tensor,
    normal: torch.Tensor,
    nodes: torch.Tensor,
    splits = torch.zeros(0, dtype=torch.int32),
    prims = torch.zeros(0, 2, dtype=torch.int32),
    box_points = torch.zeros(0, 24, dtype=torch.float32)
    ) -> torch.Tensor:

    ip = torch.zeros(len(nodes), dtype=torch.int32)
    count = _m.debug_intersect_plane_polyline(center, normal, nodes, splits, prims, box_points, ip)

    return ip[:count]


def isotoropic_repulsion_kernel(
    r: torch.Tensor,
    d0 = 1.0,
    eps = 1e-1
    ) ->  Tuple[torch.Tensor, torch.Tensor]:

    assert not r.is_cuda
    r = r.view(-1)

    output = torch.zeros_like(r)
    d_r = torch.zeros_like(r)

    _m.debug_isotropic_repulsion_kernel(r, output, d_r, d0, eps)

    return output, d_r


def render_triangles(
    resolution: Union[List[int], Tuple[int, ...]],
    mvp: torch.Tensor,
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    colors = torch.tensor([0.7, 0.3, 0.3], dtype=torch.float32),
    background = torch.ones(3, dtype=torch.float32),
    num_samples = 8,
    num_cpu_threads = 20
    ) -> torch.Tensor:

    width, height = resolution

    mvp = mvp.view(4, 4)
    vertices = vertices.view(-1, 3)
    triangles = triangles.view(-1, 3)
    colors = colors.view(-1, 3)
    background = background.view(3)

    image = torch.zeros(height, width, 3, dtype=torch.float32)

    _m.debug_render_triangles(
        mvp, vertices, triangles,
        colors, background, image,
        num_samples, num_cpu_threads
        )

    return image




