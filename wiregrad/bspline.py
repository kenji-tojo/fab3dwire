import numpy as np
import torch
import igl
import os

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('path', help='path to the input obj file or the directory containing obj files')
    parser.add_argument('--discretize', help='save discretized polyline data')
    parser.add_argument('--num_knots', type=int, default=-1, help='number of knots, which is basically the resolution of the discretized polyline')
    args = parser.parse_args()



    controls = []


    if os.path.isfile(args.path):

        input_dir = os.path.dirname(args.path)

        vtx, _ = igl.read_triangle_mesh(args.path)

        controls.append(torch.from_numpy(vtx).type(torch.float32))

    elif os.path.isdir(args.path):
        inputs = []
        for file in sorted(os.listdir(args.path)):
            if file.startswith('controls'):
                inputs.append(file)

        for file in inputs:
            vtx, _ = igl.read_triangle_mesh(os.path.join(args.path, file))

            controls.append(torch.from_numpy(vtx).type(torch.float32))

    else:
        assert False



    if not args.discretize:

        import polyscope as ps
        ps.init()

        for i,points in enumerate(controls):

            num_knots = len(points) * 8 if args.num_knots == -1 else args.num_knots

            nodes = wg.cubic_basis_spline(points, knots=num_knots).detach().cpu().numpy()
            edges = wg.polyline_edges(len(nodes), cyclic=True).detach().cpu().numpy()

            _ = ps.register_point_cloud(f'controls_{i}', points.detach().cpu().numpy(), enabled=False)
            _ = ps.register_curve_network(f'polyilne_{i}', nodes, edges)

        ps.show()

    else:
        pass
