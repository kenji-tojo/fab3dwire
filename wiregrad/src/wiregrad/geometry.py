from typing import Union, Any
import torch

from . import _m
from . import line_hierarchy


class _RepulsionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        nodes: torch.Tensor,
        splits: torch.Tensor,
        prims: torch.Tensor,
        box_points: torch.Tensor,
        cyclic: bool,
        d0: float,
        eps: float,
        num_cpu_threads: int
        ) -> Any:

        energy = torch.zeros_like(nodes[:,0])
        d_nodes = torch.zeros_like(nodes)

        _m.repulsion(
            nodes, splits, prims, box_points,
            energy, d_nodes,
            cyclic, d0, eps, num_cpu_threads
            )

        ctx.save_for_backward(d_nodes)

        return torch.sum(energy)

    @staticmethod
    def backward(ctx: Any, d_energy: Any) -> Any:

        d_nodes = ctx.saved_tensors[0]
        d_nodes *= d_energy

        ret = [ d_nodes ]
        ret += [ None ] * 7

        return tuple(ret) 

def repulsion_loss(
    nodes: torch.Tensor,
    splits: Union[str, Any] = 'auto',
    cyclic = True,
    d0 = 1.0,
    eps = 1e-1,
    num_cpu_threads = 20
    ) -> torch.Tensor:

    splits, prims, box_points = line_hierarchy.create_line_hierarchy3d(nodes, splits, cyclic=cyclic)

    return _RepulsionFunction.apply(
        nodes, splits, prims, box_points, cyclic,
        d0, eps, num_cpu_threads
        )


def uniform_distance_loss(
    nodes: torch.Tensor,
    cyclic = True
    ) -> torch.Tensor:

    nodes = nodes.view(-1,3)

    if len(nodes) < 2:
        raise ValueError(f'nodes.size() must be [>=2, 3]')

    if cyclic:
        l = torch.cat((
            torch.linalg.vector_norm(nodes[1:] - nodes[:-1], dim=1),
            torch.linalg.vector_norm(nodes[:1] - nodes[-1:], dim=1)
            ), dim=0)
    else:
        l = torch.linalg.vector_norm(nodes[1:] - nodes[:-1], dim=1)

    l_tar = torch.mean(l.detach())

    return torch.mean(torch.square(l - l_tar))


def length_loss(
    nodes: torch.Tensor,
    cyclic = True
    ) -> torch.Tensor:

    nodes = nodes.view(-1,3)

    if len(nodes) < 2:
        raise ValueError(f'nodes.size() must be [>=2, 3]')

    if cyclic:
        l_sq = torch.cat((
            torch.sum(torch.square(nodes[1:] - nodes[:-1]), dim=1),
            torch.sum(torch.square(nodes[:1] - nodes[-1:]), dim=1)
            ), dim=0)
    else:
        l_sq = torch.sum(torch.square(nodes[1:] - nodes[:-1]), dim=1)

    return torch.mean(l_sq)


def scale_loss(
    nodes: torch.Tensor
    ) -> torch.Tensor:

    nodes = nodes.view(-1,3)

    return torch.mean(torch.sum(torch.square(nodes), dim=1))


def bending_loss(
    nodes: torch.Tensor,
    cyclic = True
    ) -> torch.Tensor:

    nodes = nodes.view(-1,3)

    if len(nodes) < 3:
        raise ValueError(f'nodes.size() must be [>=3, 3]')

    if cyclic:
        x = torch.cat((nodes, nodes[:2]), dim=0)
    else:
        x = nodes

    e = x[1:] - x[:-1]
    l = torch.linalg.norm(e, dim=1)

    kappa = 2.0 * torch.cross(e[:-1], e[1:], dim=1)
    kappa /= (l[:-1] * l[1:] + torch.sum(e[:-1] * e[1:], dim=1) + 1e-15).unsqueeze(1)

    l_bar = 0.5 * (l[:-1] + l[1:])

    return torch.mean(torch.sum(kappa * kappa, dim=1) / (l_bar.unsqueeze(1) + 1e-15))


def tetrahedron_loss(
    nodes: torch.Tensor,
    cyclic = True
    ) -> torch.Tensor:

    nodes = nodes.view(-1,3)

    if len(nodes) < 4:
        raise ValueError(f'nodes.size() must be [>=4, 3]')

    if cyclic:
        x = torch.cat((nodes, nodes[:3]), dim=0)
    else:
        x = nodes

    x0 = x[0:-3].detach()
    x1 = x[1:-2]
    x2 = x[2:-1]
    x3 = x[3:  ].detach()

    y1 = x1 - x0
    y2 = x2 - x0
    y3 = x3 - x0

    tau = torch.abs(torch.sum(y1 * torch.cross(y2, y3, dim=1), dim=1))

    return torch.sum(tau)



