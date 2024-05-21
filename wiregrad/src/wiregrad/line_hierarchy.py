from typing import Union, Tuple, List, Any
import numpy as np
import torch
import math

from . import _m


def create_line_hierarchy3d(
    nodes: torch.Tensor,
    splits: Union[str, Any] = 'auto',
    cyclic = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    nodes = nodes.view(-1,3)
    num_prims = len(nodes) if cyclic else len(nodes) - 1


    if isinstance(splits, str):
        if splits == 'auto':
            if num_prims < 64:
                splits = []
            else:
                splits = [ 16 ]
                splits += [ 2 ] * max(0, int(math.log2(num_prims // 16)) - 1)
        elif splits == 'none':
            splits = []
        else:
            raise ValueError(f'if splits is str, it must be either \'auto\' or \'none\', but is {splits}')

    if isinstance(splits, List) or isinstance(splits, Tuple):
        splits = torch.tensor(splits, dtype=torch.int32)

    if isinstance(splits, np.ndarray):
        splits = torch.from_numpy(splits).type(torch.int32)

    if isinstance(splits, torch.Tensor):
        assert splits.dtype == torch.int32
        splits = splits.view(-1).to(nodes.device)
    else:
        raise TypeError(f'splits must be str, List, Tuple, numpy.ndarray, or torch.Tensor, but is {type(splits)}')

    assert len(splits) == 0 or torch.prod(splits) < num_prims


    boxes = torch.cumprod(splits, dim=0)
    total_boxes = torch.sum(boxes).item()
    prims = torch.zeros(total_boxes, 2, dtype=torch.int32, device=nodes.device)
    box_points = torch.zeros(total_boxes, 24, dtype=torch.float32, device=nodes.device)

    _m.create_line_hierarchy3d(nodes, boxes, prims, box_points, cyclic)

    return splits, prims, box_points





def create_line_hierarchy2d(
    nodes: torch.Tensor,
    splits: Union[str, Any] = 'auto',
    cyclic = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    nodes = nodes.view(-1, 2)
    num_prims = len(nodes) if cyclic else len(nodes) - 1


    if isinstance(splits, str):
        if splits == 'auto':
            if num_prims < 32:
                splits = []
            else:
                splits = [ 16 ]
                splits += [ 2 ] * max(0, int(math.log2(num_prims // 16)))
        elif splits == 'none':
            splits = []
        else:
            raise ValueError(f'if splits is str, it must be either \'auto\' or \'none\', but is {splits}')

    if isinstance(splits, List) or isinstance(splits, Tuple):
        splits = torch.tensor(splits, dtype=torch.int32)

    if isinstance(splits, np.ndarray):
        splits = torch.from_numpy(splits).type(torch.int32)

    if isinstance(splits, torch.Tensor):
        assert splits.dtype == torch.int32
        splits = splits.view(-1).to(nodes.device)
    else:
        raise TypeError(f'splits must be str, List, Tuple, numpy.ndarray, or torch.Tensor, but is {type(splits)}')

    assert len(splits) == 0 or torch.prod(splits) < num_prims


    boxes = torch.cumprod(splits, dim=0)
    total_boxes = torch.sum(boxes).item()
    prims = torch.zeros(total_boxes, 2, dtype=torch.int32, device=nodes.device)
    bounds = torch.zeros(total_boxes, 4, dtype=torch.float32, device=nodes.device)

    _m.create_line_hierarchy2d(nodes, boxes, prims, bounds, cyclic)

    return splits, prims, bounds





