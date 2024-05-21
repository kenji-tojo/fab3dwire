import numpy as np
import torch
import math

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=100, help='number of control points')
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




    num_knots = 10 * num
    nodes = wg.cubic_basis_spline(points.detach(), knots=num_knots, cyclic=True)
    edges = wg.polyline_edges(len(nodes), cyclic=True)


    points.requires_grad_()

    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=True),
        step_size = 5e-5,
        reparam_lambda = 1e-3
        )


    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    # ps.set_ground_plane_mode("none")

    curve = ps.register_curve_network(
        'curve',
        nodes.detach().cpu().numpy(),
        edges.detach().cpu().numpy(),
        radius=2e-3
        )

    # dummy points to adjust the initial camera settings
    _ = ps.register_point_cloud('_dummy', points=6.0*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]), enabled=False)


    initial_length = torch.sum(torch.linalg.vector_norm(nodes[1:]-nodes[:-1], dim=1)).detach()

    stop = False
    iter = 0

    def callback():
        global stop
        global iter

        if iter == 2000:
            stop = True

        psim.Text(f'iter = {iter}')

        _, stop = psim.Checkbox("stop", stop)

        if stop:
            return

        iter += 1

        optimizer.zero_grad()

        nodes = wg.cubic_basis_spline(points, knots=num_knots, cyclic=True)

        # main torsion loss to encourage piecewise-planar bending parts
        loss = 1e1 * wg.tetrahedron_loss(nodes, cyclic=True)

        # roughly preserve the curve length
        length = torch.sum(torch.linalg.vector_norm(nodes[1:] - nodes[:-1], dim=1))
        loss += 1e-1 * torch.square(initial_length - length)

        # fixing the curve center at the origin
        center = torch.mean(points, dim=0)
        loss += 1e-1 * torch.sum(torch.square(center))

        # uniform spacing of control points
        loss += 1e-1 * wg.uniform_distance_loss(points, cyclic=True)

        loss.backward()

        optimizer.step()

        curve.update_node_positions(nodes=nodes.detach().cpu().numpy())

    ps.set_user_callback(callback)

    ps.show()

