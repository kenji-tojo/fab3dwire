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
    ## (Internal) This code visualizes the output of the edge sampling function used for
    ##   differentiating the polygon-filled rendering.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=100, help='number of nodes')
    parser.add_argument('-s', '--samples', type=int, default=400, help='number of edge samples')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--open', action='store_true', help='use open polyline (not actually used in polygon-fill)')
    parser.add_argument('--jitter', action='store_true', help='use jittered samples')
    args = parser.parse_args()


    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)


    polygons = []

    num = args.num
    cyclic = not args.open

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)

    shift = 3.0
    nodes[:,0] += shift

    polygons.append(nodes)

    nodes = nodes.detach().clone()
    nodes = torch.matmul(nodes, wg.rotation(torch.tensor([90.0, 0, 0])).t())

    nodes[:,0] -= 2.0 * shift
    polygons.append(nodes)


    if not args.cpu and torch.cuda.is_available() and wg.cuda.is_available():
        polygons = [ nodes.to('cuda') for nodes in polygons ]

    print('nodes.device =', polygons[0].device)


    width, height = 128, 128

    camera = wg.Camera()
    camera.aspect = width / height

    matrix = camera.look_at(
        eye = 20.0 * torch.tensor([0.0, 0.0, 1.0]),
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
    length_cdf, length_sum = wg.edge_sampling.create_cdf(lengths)

    num_samples = args.samples

    if args.jitter:
        torch.manual_seed(0)
        samples = (torch.arange(num_samples) + torch.rand(num_samples)) / num_samples
    else:
        samples = (torch.arange(num_samples) + 0.5) / num_samples

    samples = samples.to(nodes.device)
    node_id, lerp, normal = wg.edge_sampling.sample_edges(nodes, edges, length_cdf, samples)


    lerp = lerp.unsqueeze(1)

    p = (1.0 - lerp) * nodes[node_id[:,0]] + lerp * nodes[node_id[:,1]]
    p = p.detach().cpu().numpy()


    plotter.init()

    fig, ax = plotter.create_figure(510, 510)
    ax.set_title(f'Sample points and their normal directions (n_samples = {num_samples})')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks(width * np.arange(5) / 4)
    ax.set_yticks(height * np.arange(5) / 4)

    ax.scatter(p[:,0], p[:,1], s=2.0)

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

    fname = os.path.join(output_dir, 'sample_edges.png')
    print('Saving result to', fname)

    fig.savefig(fname, dpi=300)





