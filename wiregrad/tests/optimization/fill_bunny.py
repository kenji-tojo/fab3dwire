import torch
import numpy as np
import math
import os, shutil
import igl
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


    output_dir = './output/fill_bunny'
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)



    vtx, tri = igl.read_triangle_mesh('../../data/assets/bunny_8k.obj')
    vtx = torch.from_numpy(vtx).type(torch.float32)
    tri = torch.from_numpy(tri).type(torch.int32)
    vtx = torch.matmul(vtx, wg.rotation(angle=torch.tensor([90.0, 0, 0])).t())


    width, height = 224, 224

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.rotate(
        eye = torch.tensor([-0.2, -4, 1.5]),
        angle = torch.tensor([80.0, 0, 0])
        )

    target = wg.debug.render_triangles(
        (width, height),
        mvp, vtx, tri,
        colors = torch.tensor([0.7, 0.3, 0.3]),
        num_samples = 8,
        ).detach()

    imsave.save_image(
        img = target,
        path = os.path.join(output_dir, 'target.png')
        )




    num = 150
    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.sin(phi).unsqueeze(1)
    y = torch.cos(phi).unsqueeze(1)
    z = torch.zeros(num).unsqueeze(1)
    points = torch.cat((x, y, z), dim=1) * 0.5

    mvp = camera.look_at(
        eye = 3 * torch.tensor([0.0, 0.0, 1.0]),
        center = torch.tensor([0.0, 0.0, 0.0]),
        up = torch.tensor([0.0, 1.0, 0.0])
        )

    if use_cuda:
        print(f'use {torch.cuda.get_device_name()}')
        target = target.cuda()
        points = points.cuda()

    print('points.device =', points.device)


    points.requires_grad_()

    optimizer = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=True),
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
            polygons = [ wg.cubic_basis_spline(points, knots=len(points) * 4) ],
            num_samples = 8,
            num_edge_samples = 10000
            )

        diff = target - img

        loss = torch.mean(torch.abs(diff))
        loss += wg.uniform_distance_loss(points, cyclic=True)
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
        polygons = [ wg.cubic_basis_spline(points, knots=len(points) * 8) ],
        num_samples = 8
        ).detach()

    imsave.save_image(img, os.path.join(output_dir, f'final.png'))



