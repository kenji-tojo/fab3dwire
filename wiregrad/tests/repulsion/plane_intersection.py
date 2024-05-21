import torch
import numpy as np
import math

import wiregrad as wg


if __name__ == '__main__':
    ## This code visually explains the plane-polyline intersection test
    ## that is used for computing our simplified repulsion energy.
    ##
    ## "intersections" are the points that may exert repulsive force against the point "primitive."
    ## "plane" is orthogonal to the curve "tangent" at the "primitive."
    ##
    ## Note that, as our kernel is finite-supported, intersection points that are far enough will be ignored.
    ##


    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--prim_id', type=int, default=1000, help='ID of a line segment to visualize')
    args = parser.parse_args()



    num = 2000
    prim_id = max(0, args.prim_id) % num

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)

    plane_center = 0.5 * (nodes[prim_id+1] + nodes[prim_id])
    plane_normal = nodes[prim_id+1] - nodes[prim_id]
    plane_normal /= torch.linalg.norm(plane_normal)


    ip_brute_force = wg.debug.intersect_plane_polyline(plane_center, plane_normal, nodes)

    splits, prims, box_points = wg.line_hierarchy.create_line_hierarchy3d(nodes, splits='auto')
    ip_accelerated = wg.debug.intersect_plane_polyline(plane_center, plane_normal, nodes, splits, prims, box_points)

    ip_brute_force = ip_brute_force.detach().cpu().numpy()
    ip_accelerated = ip_accelerated.detach().cpu().numpy()
    np.sort(ip_brute_force)
    np.sort(ip_accelerated)

    assert len(ip_brute_force) == len(ip_accelerated)
    assert np.all(ip_brute_force == ip_accelerated)


    ip = ip_accelerated
    ip = np.array(list(set(ip) - {prim_id}))

    nodes = nodes.detach().cpu().numpy()

    edges = wg.polyline_edges(len(nodes), cyclic=True)
    edges = edges.detach().cpu().numpy()

    import polyscope as ps
    ps.init()

    curve = ps.register_curve_network('curve', nodes, edges, radius=2e-3)

    prim = ps.register_point_cloud('primitive', nodes[prim_id][None], radius=1e-2)
    prim.add_vector_quantity('tangent', plane_normal.detach().cpu().numpy()[None], radius=0.008, length=0.07, enabled=True)

    _ = ps.register_point_cloud('intersections', nodes[ip], radius=1e-2)

    ## visualizing the plane
    cen = plane_center.detach().cpu().numpy()
    ez = plane_normal.detach().cpu().numpy()
    ex = np.array([-ez[1], ez[0], 0], dtype=np.float32)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)

    radius = 5.0
    c0 = cen + radius * (-ex - ey)
    c1 = cen + radius * ( ex - ey)
    c2 = cen + radius * (-ex + ey)
    c3 = cen + radius * ( ex + ey)

    pln_vtx = np.concatenate((c0, c1, c2, c3), axis=0).reshape(4,3)
    pln_tri = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    pln = ps.register_surface_mesh('plane', pln_vtx, pln_tri)
    pln.set_transparency(0.5)

    ps.show()



