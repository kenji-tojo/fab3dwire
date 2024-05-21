import numpy as np
import torch
import math
import os, shutil
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import imsave


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=240, help='number of itereations')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    args = parser.parse_args()


    use_cuda = not args.cpu and torch.cuda.is_available() and wg.cuda.is_available()


    output_dir = './output/fill_multiple'
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


    num = 10
    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.sin(phi).unsqueeze(1)
    y = torch.cos(phi).unsqueeze(1)
    z = torch.zeros(len(phi)).unsqueeze(1)
    points = torch.cat((x, y, z), dim=1)
    points[torch.arange(len(phi)//2, dtype=torch.int64) * 2 + 1] *= 0.3

    points = torch.cat((
        points + torch.Tensor([-1.0, 0, 0]),
        points + torch.Tensor([ 1.0, 0, 0]),
        ), dim=0)


    if use_cuda:
        print(f'use {torch.cuda.get_device_name()}')
        points = points.cuda()

    print('points.device =', points.device)


    width, height = 448, 224

    mvp = wg.orthographic(
        left = -2.0,
        right = 2.0,
        bottom = -1.0,
        top = 1.0,
        near = -1.0,
        far = 1.0
        )

    target = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ wg.cubic_basis_spline(points[i*num:(i+1)*num], knots=num * 50) for i in range(2) ],
        num_samples = 8
        ).detach()

    imsave.save_image(
        img = target,
        path = os.path.join(output_dir, 'target.png')
        )




    num = 50
    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.sin(phi).unsqueeze(1)
    y = torch.cos(phi).unsqueeze(1)
    z = torch.zeros(num).unsqueeze(1)
    points = torch.cat((x, y, z), dim=1) * 0.2

    points = torch.cat((
        points + torch.Tensor([-1.0, 0, 0]),
        points + torch.Tensor([ 1.0, 0, 0]),
        ), dim=0)

    unique_edges = torch.cat((
        wg.polyline_edges(num, cyclic=True),
        wg.polyline_edges(num, cyclic=True) + num,
        ), dim=0)


    if use_cuda:
        points = points.cuda()


    points.requires_grad_()

    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = unique_edges,
        step_size = 1e-3,
        reparam_lambda = 0.05
        )



    from tqdm import tqdm
    iterations = tqdm(range(args.iter), 'gradient descent')

    ini = None


    for iter in iterations:

        optimizer.zero_grad()

        img = wg.render_filled_polygons(
            (width, height),
            mvp = mvp,
            polygons = [ wg.cubic_basis_spline(points[i*num:(i+1)*num], knots=num * 8) for i in range(2) ],
            num_samples = 8,
            num_edge_samples = 10000
            )

        diff = target - img

        loss = torch.mean(torch.abs(diff))
        loss += wg.uniform_distance_loss(points[:num], cyclic=True)
        loss += wg.uniform_distance_loss(points[num:], cyclic=True)
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
        imsave.save_image(ini.detach(), os.path.join(output_dir, f'initial.png'))

    img = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ wg.cubic_basis_spline(points[i*num:(i+1)*num], knots=num * 8) for i in range(2) ],
        num_samples = 8
        ).detach()

    imsave.save_image(img, os.path.join(output_dir, f'final.png'))



