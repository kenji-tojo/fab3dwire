import torch
import numpy as np
import math
import os
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import plotter


if __name__ == '__main__':
    ## 
    ## (Internal) This code visualizes the output of the edge-and-corner sampling function used for
    ##   differentiating the line rendering.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=6, help='number of nodes')
    parser.add_argument('-s', '--samples', type=int, default=1000, help='number of edge samples')
    parser.add_argument('--cpu', action='store_true', help='use CUP')
    parser.add_argument('--open', action='store_true', help='use open polylines')
    parser.add_argument('--jitter', action='store_true', help='use jittered samples')
    args = parser.parse_args()


    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)


    stroke_width = 4.0

    polygons = []

    num = args.num
    cyclic = not args.open

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi)
    y = torch.sin(phi)
    z = torch.zeros(len(phi))
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)

    shift = 0.5 * torch.tensor([[-1.0, 0, 1.0]])
    nodes += shift

    polygons.append(nodes)

    nodes = nodes.detach().clone()
    nodes -= shift
    nodes = torch.matmul(nodes, wg.rotation(torch.tensor([0, -30.0, 0])).t())
    nodes -= shift

    polygons.append(nodes)


    if not args.cpu and torch.cuda.is_available() and wg.cuda.is_available():
        polygons = [ nodes.to('cuda') for nodes in polygons ]

    print('nodes.device =', polygons[0].device)


    width, height = 128, 128

    camera = wg.Camera()
    camera.aspect = width / height

    matrix = camera.look_at(
        eye = 3.0 * torch.tensor([1.0, 0.0, 1.0]),
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )


    screen = []

    for nodes in polygons:
        ndc = wg.homography(matrix, nodes)
        ndc[:,1] *= -1.0 # flip vertically
        screen.append(wg.viewport((width, height), ndc))


    nodes = torch.cat(screen, dim=0)
    num_nodes = torch.tensor([ len(nodes) for nodes in screen ], device=nodes.device).type(torch.int32)

    edges, lengths = wg.edge_sampling.precompute_edges(num_nodes, nodes, cyclic=cyclic)
    corner_normals, angles = wg.edge_sampling.precompute_corners(num_nodes, nodes, cyclic=cyclic)


    length_cdf, length_sum = wg.edge_sampling.create_cdf(torch.cat((lengths, lengths, 0.5 * stroke_width * angles), dim=0))

    num_samples = args.samples

    if args.jitter:
        torch.manual_seed(0)
        samples = (torch.arange(num_samples) + torch.rand(num_samples)) / num_samples
    else:
        samples = (torch.arange(num_samples) + 0.5) / num_samples

    samples = samples.to(nodes.device)

    node_id, lerp, normal, position = wg.edge_sampling.sample_edges_and_corners(
        nodes, stroke_width,
        edges, length_cdf, corner_normals, angles,
        samples
        )


    lerp = lerp.unsqueeze(1)

    p = position.detach().cpu().numpy()
    q = (1.0 - lerp) * nodes[node_id[:,0]] + lerp * nodes[node_id[:,1]]
    q = q.detach().cpu().numpy()


    plotter.init()

    fig, ax = plotter.create_figure(510, 510)
    ax.set_title(f'Sample points and their normal directions (n_samples = {len(samples)})')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks(width * np.arange(5) / 4)
    ax.set_yticks(height * np.arange(5) / 4)

    ax.scatter(p[:,0], p[:,1], s=2.0)
    ax.scatter(q[:,0], q[:,1], s=2.0)

    nrm = normal.detach().cpu().numpy()

    for i in range(num_samples):
        ax.arrow(p[i,0], p[i,1], nrm[i,0], nrm[i,1], lw=1.0)

    for sc in screen:
        x = sc[:,0].detach().cpu().tolist()
        y = sc[:,1].detach().cpu().tolist()
        if cyclic:
            x += x[:1]
            y += y[:1]
        ax.plot(x, y, lw=0.5)


    fname = os.path.join(output_dir, 'sample_corners_and_edges.png')
    print('Saving result to', fname)

    fig.savefig(fname, dpi=300)





