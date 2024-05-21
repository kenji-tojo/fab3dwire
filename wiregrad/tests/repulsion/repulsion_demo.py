import torch
import numpy as np
import math

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--wo', action='store_true', help='running without repulsion')
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    parser.add_argument('-n', '--num', type=int, default=1000, help='number of polyline nodes')
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


    d0 = 10.0


    if torch.cuda.is_available() and wg.cuda.is_available():
        points = points.to('cuda')

    print('points.device =', points.device)



    num_knots = 10 * num
    nodes = wg.cubic_basis_spline(points.detach(), knots=num_knots, cyclic=True)
    edges = wg.polyline_edges(len(nodes), cyclic=True)


    points.requires_grad_()

    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=True),
        step_size = 2e-3,
        reparam_lambda = 0.1
        )


    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_ground_plane_mode("none")

    curve = ps.register_curve_network(
        'curve',
        nodes.detach().cpu().numpy(),
        edges.detach().cpu().numpy(),
        radius=2e-3
        )


    initial_length = torch.sum(torch.linalg.vector_norm(nodes[1:]-nodes[:-1], dim=1)).detach()
    target_length = 10.0 * initial_length


    stop = False

    def callback():
        global stop

        io = psim.GetIO()
        psim.Text(f'Running at {io.Framerate:.1f} FPS')

        _, stop = psim.Checkbox("stop", stop)

        if stop:
            return

        optimizer.zero_grad()

        nodes = wg.cubic_basis_spline(points, knots=num_knots, cyclic=True)

        loss = 0.0

        if not args.wo:
            # enforcing repulsion
            loss += 1e4 * wg.repulsion_loss(nodes, splits='auto', d0=0.3)

        # elongaging the curve
        loss += torch.square(target_length - torch.sum(torch.linalg.vector_norm(nodes[1:] - nodes[:-1], dim=1)))

        # containing it in a sphere
        loss += 1e4 * torch.sum(torch.square(torch.clip(torch.linalg.vector_norm(nodes, dim=1) - 2.0, min=0.0, max=None)))

        # uniform spacing of control points
        loss += wg.uniform_distance_loss(points, cyclic=True)

        loss.backward()

        optimizer.step()

        curve.update_node_positions(nodes=nodes.detach().cpu().numpy())

    ps.set_user_callback(callback)

    ps.show()


