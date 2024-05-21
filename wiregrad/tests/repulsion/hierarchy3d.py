import torch
import numpy as np
import math

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=2000, help='number of nodes')
    args = parser.parse_args()


    num = args.num

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)


    splits, prims, box_points = wg.line_hierarchy.create_line_hierarchy3d(nodes, splits='auto')
    splits = splits.detach().cpu().numpy()
    print('     splits =', splits)
    print('# of leaves =', np.prod(splits))
    print('# of prims  =', len(nodes))


    if torch.cuda.is_available() and wg.cuda.is_available():
        print('testing CUDA hierarchy construction')
        nodes = nodes.to('cuda')
        _, prims_cuda, box_points_cuda = wg.line_hierarchy.create_line_hierarchy3d(nodes, splits=splits)
        nodes = nodes.cpu()
        prims_cuda = prims_cuda.cpu()
        box_points_cuda = box_points_cuda.cpu()

        assert torch.max(torch.abs(prims - prims_cuda)) == 0

        float_diff = torch.max(torch.abs(box_points - box_points_cuda))
        print(f'floatting point diff. = {float_diff}')
        assert float_diff < 1e-6




    nodes = nodes.detach().cpu().numpy()

    edges = wg.polyline_edges(len(nodes), cyclic=True)
    edges = edges.detach().cpu().numpy()

    import polyscope as ps
    ps.init()
    curve = ps.register_curve_network('curve', nodes, edges, radius=1.9e-3)

    offset = 0

    box_edges_template = np.array([
        [0, 1], [0, 2], [2, 3], [1, 3],
        [4, 5], [4, 6], [6, 7], [5, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
        ], dtype=np.int32)

    for i in range(len(splits)):
        num_boxes = np.prod(splits[0:i+1])

        prm = prims[offset:offset+num_boxes]
        box = box_points[offset:offset+num_boxes]

        box_nodes, box_edges = [], []

        for ht,b in zip(prm, box):
            if ht[0] < ht[1]:
                box_edges.append(box_edges_template.copy() + len(box_nodes) * 8)
                box_nodes.append(b.view(8,3).detach().cpu().numpy())

        box_nodes = np.concatenate(box_nodes, axis=0)
        box_edges = np.concatenate(box_edges, axis=0)
        _ = ps.register_curve_network(f'split_{i}', box_nodes, box_edges, radius=2e-3, enabled=i<3)

        offset += num_boxes

    ps.show()


