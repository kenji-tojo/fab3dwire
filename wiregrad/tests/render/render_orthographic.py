import torch
import numpy as np
import math
import time
import os
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import imsave


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=16, help='number of nodes')
    parser.add_argument('-s', '--samples', type=int, default=8, help='number of samples per pixel')
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available() and wg.cuda.is_available()

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)


    num = args.num

    phi = 2.0 * math.pi * torch.arange(num) / num
    radius = 5.0

    x = radius * torch.sin(phi)
    y = radius * torch.cos(phi)
    z = torch.zeros(len(phi))
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)

    nodes = torch.cat((x,y,z), dim=1)
    nodes[torch.arange(len(phi)//2, dtype=torch.int64) * 2 + 1] *= 0.4

    if use_cuda:
        nodes = nodes.cuda()

    print('nodes.device =', nodes.device)



    width, height = 256, 256

    mvp = wg.orthographic(
        left = -radius,
        right = radius,
        bottom = -radius,
        top = radius,
        near = -1.0,
        far = 1.0
        )


    img = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ nodes ],
        num_samples = args.samples,
        num_cpu_threads = args.threads,
        use_hierarchy = True
        )


    imsave.save_image(img, os.path.join(output_dir, 'orthographic.png'))




