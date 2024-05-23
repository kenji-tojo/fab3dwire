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
    ##
    ## Polygon-fill rendering
    ##


    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=100, help='number of nodes')
    parser.add_argument('-s', '--samples', type=int, default=8, help='number of samples per pixel')
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available() and wg.cuda.is_available()

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)


    num = args.num

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)




    width, height = 320, 270

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.look_at(
        eye = 6.0 * torch.tensor([1.0, 1.0, 1.0]),
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )


    img = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ nodes ],
        num_samples = args.samples,
        num_cpu_threads = args.threads
        ).detach()


    imsave.save_image(img, os.path.join(output_dir, 'trefoil_fill.png'))




