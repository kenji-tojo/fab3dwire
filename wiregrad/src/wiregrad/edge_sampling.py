from typing import Tuple
import math
import torch

from .topology import *
from . import _m


def precompute_edges(
    num_nodes: torch.Tensor,
    nodes: torch.Tensor,
    cyclic = True
    ) -> Tuple[torch.Tensor, ...]:

    num_polylines = len(num_nodes)
    total_nodes = torch.sum(num_nodes).item()

    node_offset = 0
    edges = []

    for i in range(num_polylines):
        num = num_nodes[i].item()
        edges.append(polyline_edges(num, cyclic) + node_offset)
        node_offset += num

    edges = torch.cat(edges, dim=0).to(nodes.device)
    num_edges = len(edges)

    if cyclic:
        assert num_edges == total_nodes
    else:
        assert num_edges == total_nodes - num_polylines

    lengths = torch.linalg.vector_norm(nodes[edges[:,1]] - nodes[edges[:,0]], dim=1)

    return edges, lengths


def precompute_corners(
    num_nodes: torch.Tensor,
    nodes: torch.Tensor,
    cyclic: bool
    ) -> Tuple[torch.Tensor, ...]:

    num_polylines = len(num_nodes)

    normals = torch.zeros_like(nodes)
    angles = torch.zeros_like(nodes[:,0])

    def _normalize_edges_and_compute_corners(e: torch.Tensor, normals_out: torch.Tensor, angles_out: torch.Tensor):
        l = torch.linalg.vector_norm(e, dim=1)
        zero_edge = l == 0
        zero_node = torch.logical_and(zero_edge[:-1], zero_edge[1:])

        l[zero_edge] = 1.0
        e /= l.unsqueeze(1)
        e10, e21 = e[:-1], e[1:]

        n = e10 - e21
        n[zero_node] = torch.ones_like(n[:1])
        normals_out.copy_(n / torch.linalg.vector_norm(n, dim=1).unsqueeze(1))

        angles_out.copy_(torch.acos(torch.clip(torch.sum(e10 * e21, dim=1), min=-1.0, max=1.0)))
        angles_out[torch.logical_or(zero_edge[:-1], zero_edge[1:])] = math.pi
        angles_out[zero_node] = 0.0

    node_offset = 0

    for i in range(num_polylines):
        num = num_nodes[i].item()
        bgn = node_offset
        end = node_offset + num

        if cyclic:
            e = torch.zeros(num + 1, 2, dtype=torch.float, device=nodes.device)
            e[1:-1] = nodes[bgn+1:end] - nodes[bgn:end-1]
            e[0] = e[-1] = nodes[bgn] - nodes[end-1]
            _normalize_edges_and_compute_corners(e, normals[bgn:end], angles[bgn:end])
        else:
            e = nodes[bgn+1:end] - nodes[bgn:end-1]
            _normalize_edges_and_compute_corners(e, normals[bgn+1:end-1], angles[bgn+1:end-1])
            normals[bgn], normals[end-1] = -e[0], e[-1]
            angles[bgn] = angles[end-1] = math.pi

        node_offset += num

    return normals, angles


def create_cdf(
    weights: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:

    weights = weights.ravel()

    cdf = torch.zeros(len(weights) + 1, dtype=torch.float32, device=weights.device)
    cdf[1:] = torch.cumsum(weights, dim=0)
    weight_sum = cdf[-1].item()
    cdf /= weight_sum

    return cdf, weight_sum


def sample_edges(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    length_cdf: torch.Tensor,
    samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

    nodes = nodes.view(-1, 2)
    edges = edges.view(-1, 2)
    length_cdf = length_cdf.ravel()
    samples = samples.ravel()

    num_samples = len(samples)
    node_id = torch.zeros(num_samples, 2, dtype=torch.int32, device=nodes.device)
    lerp = torch.zeros(num_samples, dtype=torch.float32, device=nodes.device)
    normal = torch.zeros(num_samples, 2, dtype=torch.float32, device=nodes.device)

    _m.sample_edges(
        nodes, edges, length_cdf,
        samples, node_id, lerp, normal,
        )

    return node_id, lerp, normal


def sample_edges_and_corners(
    nodes: torch.Tensor,
    stroke_width: float,
    edges: torch.Tensor,
    length_cdf: torch.Tensor,
    corner_normals: torch.Tensor,
    angles: torch.Tensor,
    samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

    nodes = nodes.view(-1, 2)
    edges = edges.view(-1, 2)
    length_cdf = length_cdf.ravel()
    corner_normals = corner_normals.view(-1, 2)
    angles = angles.ravel()
    samples = samples.ravel()

    num_samples = len(samples)
    node_id = torch.zeros(num_samples, 2, dtype=torch.int32, device=nodes.device)
    lerp = torch.zeros(num_samples, dtype=torch.float32, device=nodes.device)
    normal = torch.zeros(num_samples, 2, dtype=torch.float32, device=nodes.device)
    position = torch.zeros(num_samples, 2, dtype=torch.float32, device=nodes.device)

    _m.sample_edges_and_corners(
        nodes, stroke_width,
        edges, length_cdf, corner_normals, angles,
        samples, node_id, lerp, normal, position,
        )

    return node_id, lerp, normal, position




