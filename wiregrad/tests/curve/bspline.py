import numpy as np
import torch
import math

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=50, help='number of control points')
    args = parser.parse_args()


    num = args.num

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    points = torch.cat((x,y,z), dim=1)


    if torch.cuda.is_available() and wg.cuda.is_available():
        points = points.to('cuda')

    print('points.device =', points.device)


    is_closed = True

    num_knots = 8 * num
    nodes = wg.cubic_basis_spline(points, knots=num_knots, cyclic=is_closed)
    nodes = torch.cat((nodes, nodes[-1:]), dim=0)
    edges = wg.polyline_edges(len(nodes), cyclic=False)


    import polyscope as ps
    import polyscope.imgui as psim
    ps.init()

    net = ps.register_curve_network(
        'curve',
        nodes.detach().cpu().numpy(),
        edges.detach().cpu().numpy(),
        radius=2e-3
        )

    pnt = ps.register_point_cloud(
        'points',
        points=points.detach().cpu().numpy()
        )

    def callback():
        global is_closed
        changed, is_closed = psim.Checkbox("closed", is_closed)

        if not changed:
            return

        nodes = wg.cubic_basis_spline(points, knots=num_knots, cyclic=is_closed)

        if is_closed:
            nodes = torch.cat((nodes, nodes[:1]), dim=0)
        else:
            nodes = torch.cat((nodes, nodes[-1:]), dim=0)

        net.update_node_positions(nodes=nodes.detach().cpu().numpy())

    ps.set_user_callback(callback)

    ps.show()



