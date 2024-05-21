import numpy as np
import torch
import math

import wiregrad as wg


if __name__ == '__main__':

    num = 20

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



    num_knots = 8 * num
    nodes = wg.cubic_basis_spline(points, knots=num_knots)
    edges = wg.polyline_edges(len(nodes))


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

    org = ps.register_point_cloud(
        'origin',
        points=np.zeros((1,3), dtype=np.float32),
        radius=0.01
        )

    org.add_vector_quantity('X', np.array([[1.0, 0, 0]]), enabled=True, radius=5e-3, length=0.1, color=[0.9,0,0])
    org.add_vector_quantity('Y', np.array([[0, 1.0, 0]]), enabled=True, radius=5e-3, length=0.1, color=[0,0.9,0])
    org.add_vector_quantity('Z', np.array([[0, 0, 1.0]]), enabled=True, radius=5e-3, length=0.1, color=[0,0,0.9])


    theta_x = 0.0
    theta_y = 0.0
    theta_z = 0.0

    def callback():
        global theta_x
        global theta_y
        global theta_z

        def update():
            rot = wg.rotation(torch.tensor([theta_x, theta_y, theta_z], dtype=torch.float32)).to(points.device)
            points_rot = torch.matmul(points, rot.transpose(1, 0))
            nodes_rot = wg.cubic_basis_spline(points_rot, knots=num_knots)
            net.update_node_positions(nodes_rot.detach().cpu().numpy())
            pnt.update_point_positions(points_rot.detach().cpu().numpy())

        changed_x, theta_x = psim.SliderFloat("theta_x", theta_x, v_min=0, v_max=360)
        changed_y, theta_y = psim.SliderFloat("theta_y", theta_y, v_min=0, v_max=360)
        changed_z, theta_z = psim.SliderFloat("theta_z", theta_z, v_min=0, v_max=360)

        if changed_x or changed_y or changed_z:
            update()

    ps.set_user_callback(callback)

    ps.show()



