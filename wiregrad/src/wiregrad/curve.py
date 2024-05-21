from typing import Union, Any
import torch

from . import _m


class _CubicBasisSplineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        knots: torch.Tensor,
        cyclic: bool,
        num_cpu_threads: int
        ) -> Any:

        nodes = torch.zeros(len(knots), points.size()[-1], dtype=torch.float32, device=points.device)

        options = torch.tensor([1 if cyclic else 0, num_cpu_threads], dtype=torch.int32)

        ctx.save_for_backward(knots, points, options)

        _m.cubic_basis_spline(points, knots, nodes, cyclic)

        return nodes

    @staticmethod
    def backward(
        ctx: Any,
        d_nodes: torch.Tensor
        ) -> Any:

        d_nodes = d_nodes.contiguous()

        knots, points, options = ctx.saved_tensors

        d_points = torch.zeros_like(points)

        cyclic = True if options[0].item() == 1 else False
        num_cpu_threads = options[1].item()

        _m.d_cubic_basis_spline(d_nodes, knots, d_points, cyclic, num_cpu_threads)

        ret = [ d_points ]
        ret += [ None ] * 3

        return tuple(ret)


def cubic_basis_spline(
    points: torch.Tensor,
    knots: Union[int, torch.Tensor],
    cyclic = True,
    num_cpu_threads = 20
    ) -> torch.Tensor:

    if len(points.size()) == 1:
        points = points.view(-1, 1)
    else:
        points = points.view(-1, points.size()[-1])

    assert len(points) >= 2

    if isinstance(knots, int):
        n = max(len(points), knots)
        if cyclic:
            knots = len(points) * torch.arange(n, dtype=torch.float32, device=points.device) / n
        else:
            knots = (len(points) - 3) * torch.arange(n, dtype=torch.float32, device=points.device) / (n - 1)
    else:
        assert isinstance(knots, torch.Tensor) and len(knots) >= len(points)
        knots = knots.to(points.device)

    return _CubicBasisSplineFunction.apply(points, knots, cyclic, num_cpu_threads)






