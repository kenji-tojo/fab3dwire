import numpy as np
import torch
from torchvision.transforms.v2 import GaussianBlur
import math
import os, shutil
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import imsave


if __name__ == '__main__':
    ##
    ## Optimization of open curves
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=240, help='number of itereations')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    args = parser.parse_args()


    use_cuda = not args.cpu and torch.cuda.is_available() and wg.cuda.is_available()


    output_dir = './output/line_open'
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


    num = 500
    x = 2.0 * (torch.arange(num) + 0.5) / num
    x = x.unsqueeze(1)
    y = 0.8 * torch.sin(math.pi * x)
    z = torch.zeros(num).unsqueeze(1)
    points = torch.cat((x, y, z), dim=1)


    if use_cuda:
        print(f'use {torch.cuda.get_device_name()}')
        points = points.cuda()

    print('points.device =', points.device)



    stroke_width = torch.tensor([8.0])


    width, height = 224, 224

    mvp = wg.orthographic(
        left = -0.2,
        right = 2.2,
        bottom = -1.2,
        top = 1.2,
        near = -1.0,
        far = 1.0
        )

    target = wg.render_polylines(
        (width, height),
        mvp = mvp,
        polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 10, cyclic=False) ],
        stroke_width = stroke_width,
        num_samples = 8,
        cyclic = False
        ).detach()

    imsave.save_image(
        img = target,
        path = os.path.join(output_dir, 'target.png')
        )



    num = 50
    x = 2.4 * (torch.arange(num) + 0.5) / num
    x = x.unsqueeze(1) - 0.2
    y = torch.zeros(num).unsqueeze(1)
    z = torch.zeros(num).unsqueeze(1)
    points = torch.cat((x, y, z), dim=1)


    if use_cuda:
        points = points.cuda()


    points.requires_grad_()


    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=False),
        step_size = 1e-3,
        reparam_lambda = 0.05
        )


    from tqdm import tqdm
    iterations = tqdm(range(args.iter), 'gradient descent')

    ini = None


    for iter in iterations:

        optimizer.zero_grad()

        img = wg.render_polylines(
            (width, height),
            mvp = mvp,
            polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 8, cyclic=False) ],
            stroke_width = stroke_width,
            num_samples = 8,
            num_edge_samples = 10000,
            cyclic = False
            )

        diff = target - img

        loss = torch.mean(torch.abs(diff))
        loss += wg.uniform_distance_loss(points, cyclic=False)
        loss.backward()

        optimizer.step()

        if iter == 0:
            ini = img.detach().clone().cpu()

        if iter % 2 == 0:
            imsave.save_image(img.detach(), os.path.join(tmp_dir, f'img_{iter//2:04d}.png'), logging=False)
            imsave.save_signed_image(torch.sum(diff.detach(), dim=-1), os.path.join(tmp_dir, f'diff_{iter//2:04d}.png'), v_max=3.0, logging=False)


    import subprocess

    subprocess.call(['ffmpeg', '-framerate', '60', '-y', '-i',
                    str(os.path.join(tmp_dir, f'img_%4d.png')), '-vb', '20M',
                    '-vcodec', 'libx264',
                    str(os.path.join(output_dir, 'color.mp4'))])

    subprocess.call(['ffmpeg', '-framerate', '60', '-y', '-i',
                    str(os.path.join(tmp_dir, f'diff_%4d.png')), '-vb', '20M',
                    '-vcodec', 'libx264',
                    str(os.path.join(output_dir, 'diff.mp4'))])

    shutil.rmtree(tmp_dir)


    if ini is not None:
        imsave.save_image(ini.detach(), os.path.join(output_dir, f'initial.png'), logging=False)

    img = wg.render_polylines(
        (width, height),
        mvp = mvp,
        polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 8, cyclic=False) ],
        stroke_width = stroke_width,
        num_samples = 8,
        cyclic = False
        ).detach()

    imsave.save_image(img, os.path.join(output_dir, f'final.png'))



