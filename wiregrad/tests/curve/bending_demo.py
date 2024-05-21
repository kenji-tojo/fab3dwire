import numpy as np
import torch
import math

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=50, help='number of control points')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
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


    if not args.cpu and torch.cuda.is_available() and wg.cuda.is_available():
        points = points.to('cuda')

    print('points.device =', points.device)


    points.requires_grad_()

    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=True),
        step_size = 3e-3,
        reparam_lambda = 0.05
        )


    num_knots = 8 * num
    nodes = wg.cubic_basis_spline(points.detach(), knots=num_knots, cyclic=True)
    edges = wg.polyline_edges(len(nodes), cyclic=True)


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

    # dummy points to adjust the initial camera settings
    _ = ps.register_point_cloud('_dummy', points=5.0*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]), enabled=False)


    stop = False
    iter = 0

    def callback():
        global stop
        global iter

        if iter == 1000:
            stop = True

        psim.Text(f'iter = {iter}')

        _, stop = psim.Checkbox("stop", stop)

        if stop:
            return

        iter += 1

        optimizer.zero_grad()


        nodes = wg.cubic_basis_spline(points, knots=num_knots, cyclic=True)

        # main bending loss
        loss = wg.bending_loss(nodes, cyclic=True)

        # bending loss tends to elongate the curve, so regularize length for a balance.
        loss += 1e-2 * wg.length_loss(points, cyclic=True)

        # uniform spacing of control points
        loss += 1e-1 * wg.uniform_distance_loss(points, cyclic=True)

        loss.backward()

        optimizer.step()

        net.update_node_positions(nodes=nodes.detach().cpu().numpy())
        pnt.update_point_positions(points=points.detach().cpu().numpy())

    ps.set_user_callback(callback)

    ps.show()


