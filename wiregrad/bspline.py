import numpy as np
import torch
import igl
import os

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='path to the input obj file')
    parser.add_argument('-d', '--dir', help='path to the directory containing input obj files')
    args =parser.parse_args()

    assert args.file or args.dir, 'either --file or --dir must be specified'


    import polyscope as ps
    ps.init()

    if args.file:
        vtx, _ = igl.read_triangle_mesh(args.file)

        points = torch.from_numpy(vtx).type(torch.float32)
        nodes = wg.cubic_basis_spline(points, knots=len(points) * 10)


        points = points.detach().cpu().numpy()
        nodes = nodes.detach().cpu().numpy()
        edges = wg.polyline_edges(len(nodes), cyclic=True).detach().cpu().numpy()


        pnt = ps.register_point_cloud('control_points', points, enabled=False)
        net = ps.register_curve_network('curve', nodes, edges)

    elif args.dir:
        controls = []
        for file in sorted(os.listdir(args.dir)):
            if file.startswith('controls'):
                controls.append(file)

        for i,file in enumerate(controls):
            vtx, _ = igl.read_triangle_mesh(os.path.join(args.dir, file))

            points = torch.from_numpy(vtx).type(torch.float32)
            nodes = wg.cubic_basis_spline(points, knots=len(points) * 10)


            points = points.detach().cpu().numpy()
            nodes = nodes.detach().cpu().numpy()
            edges = wg.polyline_edges(len(nodes), cyclic=True).detach().cpu().numpy()

            pnt = ps.register_point_cloud(f'control_points_{i}', points, enabled=False)
            net = ps.register_curve_network(f'curve_{i}', nodes, edges)


    ps.show()

