from typing import Any
import json
from easydict import EasyDict as edict
import math
import numpy as np
import torch
import igl

import wiregrad as wg


def trefoil(num: int) -> torch.Tensor:

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)

    return torch.cat((x,y,z), dim=1)


def parse_config(path: str) -> Any:
    with open(path, 'r') as f:
        config = edict(json.loads(f.read()))

    return config


def save_polyline(
    points: torch.Tensor,
    path: str
    ):

    points = points.detach().cpu()
    points = torch.matmul(points, wg.rotation(torch.tensor([-90.0, 0, 0])).t())

    ## Hacking libigl trimesh saver by using a dummy triangle.
    ## We're only interested in the vertices and their ordering, which is preserved in the output OBJ file.
    igl.write_obj(path, points.numpy(), np.array([[0,1,2]], dtype=np.int32))


def save_polyline(
    nodes: torch.Tensor,
    path: str,
    rotate = True
    ):

    nodes = nodes.detach().cpu()

    if rotate:
        nodes = torch.matmul(nodes, wg.rotation(torch.tensor([-90.0, 0, 0])).t())

    nodes = nodes.numpy()
    content = '\n'.join([ f'v {v[0]} {v[1]} {v[2]}' for v in nodes ]) + '\n'

    with open(path, 'w') as f:
        f.write(content)


def load_polyline(
    path: str
    ) -> torch.Tensor:

    nodes = []

    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('v'):
                v = [ float(x) for x in line.split(' ')[1:] ]
                assert len(v) == 3
                nodes.append(v)

    return torch.tensor(nodes, dtype=torch.float32)



