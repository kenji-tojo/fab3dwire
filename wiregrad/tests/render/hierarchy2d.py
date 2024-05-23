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
    ## (Internal) This code visualizes the 2D bounding box hierarchy used for rendering 2D curves.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    args = parser.parse_args()


    num = 20

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



    num_knots = 8 * num
    nodes = wg.cubic_basis_spline(points, knots=num_knots)
    edges = wg.polyline_edges(len(nodes))


    width, height = 200, 200

    camera = wg.Camera()
    camera.aspect = width / height

    matrix = camera.look_at(
        eye = 6.0 * torch.tensor([1.0, 1.0, 1.0]),
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )

    ndc = wg.homography(matrix, nodes)
    sc = wg.viewport((width, height), ndc)

    splits, prims, bounds = wg.line_hierarchy.create_line_hierarchy2d(sc, splits='auto', cyclic=True)

    print('splits =', splits.tolist())

    plotter.init()

    sc = sc.detach().cpu().numpy()

    fig, ax = plotter.create_figure(300, 300)
    ax.set_title('2D line hierarchy')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(width * np.arange(5) / 4)
    ax.set_yticks(height * np.arange(5) / 4)
    ax.plot(sc[:,0], sc[:,1], lw=0.5)

    splits = splits.detach().cpu().numpy()
    bounds = bounds.detach().cpu().numpy()

    color = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    offset = 0

    lw = 1.0

    for i in range(len(splits)):
        num_boxes = np.prod(splits[0:i+1])

        for bd in bounds[offset:offset + num_boxes]:
            x = [ bd[0], bd[1], bd[1], bd[0], bd[0] ]
            y = [ bd[2], bd[2], bd[3], bd[3], bd[2] ]
            # ax.plot(x, y, lw=lw, c=color[i % len(color)])
            ax.plot(x, y, lw=lw)

        lw *= 0.8
        offset += num_boxes

    import os
    os.makedirs('./output', exist_ok=True)

    fname = os.path.join('./output', 'hierarchy2d.png')
    print('Saving the result to', fname)

    fig.savefig(fname, dpi=300)


