import torch


def polyline_edges(
    num: int,
    cyclic = False
    ) -> torch.Tensor:

    if cyclic:
        edges = torch.zeros(num, 2, dtype=torch.int32)
        edges[:, 0] = torch.arange(len(edges))
        edges[:, 1] = torch.arange(len(edges)) + 1
        edges[-1, 1] = 0
    else:
        edges = torch.zeros(num - 1, 2, dtype=torch.int32)
        edges[:, 0] = torch.arange(len(edges))
        edges[:, 1] = torch.arange(len(edges)) + 1

    return edges


