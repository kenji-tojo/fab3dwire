import numpy as np
import torch
import igl
import os

import wiregrad as wg

from utils import save_polyline, load_polyline


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('path', help='path to the input obj file or the directory containing obj files')
    parser.add_argument('--discretize', action='store_true', help='save discretized polyline data')
    parser.add_argument('--num_knots', type=int, default=-1, help='number of knots, which is basically the resolution of the discretized polyline')
    args = parser.parse_args()



    input_dir = None
    controls = []


    if os.path.isfile(args.path):

        input_dir = os.path.dirname(args.path)

        controls.append(load_polyline(args.path))

    elif os.path.isdir(args.path):

        input_dir = args.path

        inputs = []
        for file in sorted(os.listdir(args.path)):
            if file.startswith('controls'):
                inputs.append(file)

        for file in inputs:
            controls.append(load_polyline(os.path.join(args.path, file)))

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

        for i,points in enumerate(controls):

            assert input_dir is not None

            num_knots = len(points) * 8 if args.num_knots == -1 else args.num_knots

            save_polyline(
                nodes = wg.cubic_basis_spline(points, knots=num_knots),
                path = os.path.join(input_dir, f'polyline_{i}.obj')
                )


