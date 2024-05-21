from typing import Any, List, Tuple, Union
import torch
import math

from . import _m
from . import line_hierarchy
from . import edge_sampling
from . import transform



def checkerboard_pattern(
    num_samples = 8
    ) -> torch.Tensor:

    num_samples = max(1, num_samples)
    samples = torch.zeros(num_samples, 2, dtype=torch.float32)
    num_samples = _m.checkerboard_pattern(num_samples, samples)
    assert num_samples >= 1

    return samples[:num_samples]


def _create_line_hierarchies2d(
    num_nodes: torch.Tensor,
    nodes: torch.Tensor,
    splits = 'auto',
    cyclic = True
    ) -> Tuple[torch.Tensor, ...]:

    assert len(num_nodes) > 0

    assert splits == 'auto' or splits == 'none'
    splits_str = splits

    num_levels, splits, total_boxes, prims, bounds = list(), list(), list(), list(), list()

    node_offset = 0

    for i in range(len(num_nodes)):
        num = num_nodes[i].item()

        spl, prm, bnd = line_hierarchy.create_line_hierarchy2d(
            nodes = nodes[node_offset: node_offset + num],
            splits = splits_str,
            cyclic = cyclic
            )

        num_levels.append(len(spl))
        splits.append(spl)

        total_boxes.append(len(prm))
        prims.append(prm)
        bounds.append(bnd)

        node_offset += num

    num_levels = torch.tensor(num_levels, dtype=torch.int32, device=nodes.device)
    splits = torch.cat(splits, dim=0)

    total_boxes = torch.tensor(total_boxes, dtype=torch.int32, device=nodes.device)
    prims = torch.cat(prims, dim=0)
    bounds = torch.cat(bounds, dim=0)

    return num_levels, splits, total_boxes, prims, bounds


class RenderFilledPolygonsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        width: int,
        height: int,
        num_nodes: torch.Tensor,
        nodes: torch.Tensor,
        color: Tuple[float, ...], # not differentiated
        num_samples = 8,
        num_edge_samples = 8,
        num_cpu_threads = 20,
        background = (1.0, 1.0, 1.0), # not differentiated
        use_hierarchy = True
        ) -> Any:

        color = torch.tensor(color, dtype=torch.float32).view(3)
        background = torch.tensor(background, dtype=torch.float32).view(3)

        num_levels, splits, total_boxes, prims, bounds = _create_line_hierarchies2d(
            num_nodes, nodes,
            splits = 'auto' if use_hierarchy else 'none',
            cyclic = True # cyclic is always true for polygon-fill
            )

        ctx.save_for_backward(
            num_nodes, nodes,
            num_levels, splits, total_boxes, prims, bounds,
            color, background,
            torch.tensor([num_edge_samples, num_cpu_threads], dtype=torch.int32)
            )

        pixel_samples = checkerboard_pattern(num_samples).to(nodes.device)
        weight_sum = torch.zeros(height, width, dtype=torch.float32, device=nodes.device)
        contrib_sum = torch.zeros(height, width, 3, dtype=torch.float32, device=nodes.device)

        _m.render_filled_polygons(
            num_nodes, nodes, num_levels, splits, total_boxes, prims, bounds,
            color, pixel_samples, weight_sum, contrib_sum, num_cpu_threads, background,
            )

        return contrib_sum / weight_sum.unsqueeze(2)


    @staticmethod
    def backward(
        ctx: Any,
        d_image: torch.Tensor
        ) -> Any:

        d_image = d_image.contiguous()

        num_nodes, nodes = ctx.saved_tensors[:2]
        num_levels, splits, total_boxes, prims, bounds = ctx.saved_tensors[2:7]
        color, background = ctx.saved_tensors[7:9]
        options = ctx.saved_tensors[9]

        num_edge_samples = options[0].item()
        num_cpu_threads = options[1].item()

        d_nodes = torch.zeros_like(nodes)

        edges, lengths = edge_sampling.precompute_edges(num_nodes, nodes, cyclic=True)
        length_cdf, length_sum = edge_sampling.create_cdf(lengths)

        samples = torch.arange(num_edge_samples, device=nodes.device) + torch.rand(num_edge_samples, device=nodes.device)
        samples /= num_edge_samples
        node_id, lerp, normal = edge_sampling.sample_edges(nodes, edges, length_cdf, samples)

        _m.d_render_filled_polygons(
            d_image,
            num_nodes, nodes,
            num_levels, splits, total_boxes, prims, bounds,
            color, length_sum, node_id, lerp, normal,
            d_nodes,
            num_cpu_threads, background,
            )

        ret = [ None ] * 3
        ret += [ d_nodes ]
        ret += [ None ] * 6

        return tuple(ret)


def render_filled_polygons(
    resolution: Union[Tuple[int, ...], List[int]],
    mvp: torch.Tensor,
    polygons: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    color = (0.7, 0.3, 0.3), # not differentiated
    num_samples = 8,
    num_edge_samples = 10000,
    num_cpu_threads = 20,
    background = (1.0, 1.0, 1.0), # not differentiated
    use_hierarchy = True
    ) -> torch.Tensor:

    width, height = resolution

    if len(polygons) == 0:
        image = torch.zeros(height, width, 3, dtype=torch.float32)
        image = background[None, None, ...]
        return image

    nodes = torch.cat(polygons, dim=0).view(-1, 3)
    num_nodes = torch.tensor([len(nodes) for nodes in polygons], dtype=torch.int32, device=nodes.device)

    ndc = transform.homography(mvp, nodes)
    ndc[:,1] *= -1.0 # flip vertically
    nodes = transform.viewport((width, height), ndc)

    image = RenderFilledPolygonsFunction.apply(
        width, height, num_nodes, nodes, color,
        num_samples, num_edge_samples, num_cpu_threads,
        background, use_hierarchy
        )

    return image



class RenderPolyLineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        width: int,
        height: int,
        num_nodes: torch.Tensor,
        nodes: torch.Tensor,
        stroke_width: torch.Tensor,
        color: Tuple[float, ...], # not differentiated
        cyclic: bool,
        num_samples = 8,
        num_edge_samples = 8,
        num_cpu_threads = 20,
        background = (1.0, 1.0, 1.0), # not differentiated
        use_hierarchy = True
        ) -> Any:

        color = torch.tensor(color, dtype=torch.float32).view(3)
        background = torch.tensor(background, dtype=torch.float32).view(3)

        num_levels, splits, total_boxes, prims, bounds = _create_line_hierarchies2d(
            num_nodes, nodes,
            splits = 'auto' if use_hierarchy else 'none',
            cyclic = cyclic
            )

        ctx.save_for_backward(
            num_nodes, nodes,
            num_levels, splits, total_boxes, prims, bounds,
            stroke_width, color, background,
            torch.tensor([num_edge_samples, num_cpu_threads, 1 if cyclic else 0], dtype=torch.int32)
            )

        pixel_samples = checkerboard_pattern(num_samples).to(nodes.device)
        weight_sum = torch.zeros(height, width, dtype=torch.float32, device=nodes.device)
        contrib_sum = torch.zeros(height, width, 3, dtype=torch.float32, device=nodes.device)

        _m.render_polylines(
            num_nodes, nodes, num_levels, splits, total_boxes, prims, bounds,
            stroke_width.item(), color, cyclic,
            pixel_samples, weight_sum, contrib_sum, num_cpu_threads, background,
            )

        return contrib_sum / weight_sum.unsqueeze(2)


    @staticmethod
    def backward(
        ctx: Any,
        d_image: torch.Tensor
        ) -> Any:

        d_image = d_image.contiguous()

        num_nodes, nodes = ctx.saved_tensors[:2]
        num_levels, splits, total_boxes, prims, bounds = ctx.saved_tensors[2:7]
        stroke_width, color, background = ctx.saved_tensors[7:10]
        options = ctx.saved_tensors[10]

        num_edge_samples = options[0].item()
        num_cpu_threads = options[1].item()
        cyclic = options[2].item() == 1

        d_nodes = torch.zeros_like(nodes)
        d_storke_width = torch.zeros(0, dtype=torch.float32)

        if stroke_width.requires_grad:
            d_storke_width = torch.zeros(num_edge_samples, dtype=torch.float32, device=nodes.device)

        stroke_width = stroke_width.item()

        edges, lengths = edge_sampling.precompute_edges(num_nodes, nodes, cyclic=cyclic)
        corner_normals, angles = edge_sampling.precompute_corners(num_nodes, nodes, cyclic=cyclic)

        length_cdf, length_sum = edge_sampling.create_cdf(torch.cat((lengths, lengths, 0.5 * stroke_width * angles), dim=0))

        samples = torch.arange(num_edge_samples, device=nodes.device) + torch.rand(num_edge_samples, device=nodes.device)
        samples /= num_edge_samples

        node_id, lerp, normal, position = edge_sampling.sample_edges_and_corners(
            nodes, stroke_width,
            edges, length_cdf, corner_normals, angles,
            samples
            )

        _m.d_render_polylines(
            d_image,
            num_nodes, nodes, num_levels, splits, total_boxes, prims, bounds,
            stroke_width, color, cyclic, length_sum,
            node_id, lerp, normal, position,
            d_nodes, d_storke_width,
            num_cpu_threads, background,
            )

        d_storke_width = None if len(d_storke_width) == 0 else torch.sum(d_storke_width).view(1).cpu()

        ret = [ None ] * 3
        ret += [ d_nodes, d_storke_width ]
        ret += [ None ] * 7

        return tuple(ret)


def render_polylines(
    resolution: Union[Tuple[int, ...], List[int]],
    mvp: torch.Tensor,
    polylines: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    stroke_width: torch.Tensor,
    color = (0.7, 0.3, 0.3), # not differentiated
    cyclic = True,
    num_samples = 8,
    num_edge_samples = 10000,
    num_cpu_threads = 20,
    background = (1.0, 1.0, 1.0), # not differentiated
    use_hierarchy = True
    ) -> torch.Tensor:

    width, height = resolution

    if len(polylines) == 0:
        image = torch.zeros(height, width, 3, dtype=torch.float32)
        image[:,:,0] = background[0]
        image[:,:,1] = background[1]
        image[:,:,2] = background[2]
        return image

    nodes = torch.cat(polylines, dim=0).view(-1, 3)
    num_nodes = torch.tensor([len(nodes) for nodes in polylines], dtype=torch.int32, device=nodes.device)

    stroke_width = stroke_width.view(1).cpu()
    stroke_width = torch.clip(stroke_width, min=1e-3, max=None)

    ndc = transform.homography(mvp, nodes)
    ndc[:,1] *= -1.0 # flip vertically
    nodes = transform.viewport((width, height), ndc)

    image = RenderPolyLineFunction.apply(
        width, height, num_nodes, nodes, stroke_width, color, cyclic,
        num_samples, num_edge_samples, num_cpu_threads,
        background, use_hierarchy
        )

    return image




