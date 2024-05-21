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
    ## You can also optimize camera prameters, such as the camera's eye position.
    ## 

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=240, help='number of itereations')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    args = parser.parse_args()


    use_cuda = not args.cpu and torch.cuda.is_available() and wg.cuda.is_available()



    output_dir = './output/line_eye'
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


    phi = 2.0 * math.pi * torch.arange(50) / 50
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    points = torch.cat((x,y,z), dim=1)

    stroke_width = torch.tensor([8.0])

    eye = 7.0 * torch.tensor([1.0, 1.0, 1.0])


    if use_cuda:
        print(f'use {torch.cuda.get_device_name()}')
        points = points.cuda()

    print('points.device =', points.device)




    width, height = 224, 224

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.look_at(
        eye = eye,
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )

    target = wg.render_polylines(
        (width, height),
        mvp = mvp,
        polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 8) ],
        stroke_width = stroke_width,
        num_samples = 8
        ).detach()

    imsave.save_image(
        img = target,
        path = os.path.join(output_dir, 'target.png')
        )



    eye = 9.0 * torch.tensor([[0.5, 0.5, 1.5]])
    eye.requires_grad_()


    optimizer = wg.VectorAdam(
        points = eye,
        step_size = 1e-1
        )

    kernel_size = 31
    blur_op = GaussianBlur(kernel_size=kernel_size, sigma=(kernel_size-1)/2,)

    def multi_level_diff(diff: torch.Tensor):
        diff = diff.permute(2, 0, 1)
        diff = 0.5 * (diff + blur_op(diff))
        return diff.permute(1, 2, 0)


    from tqdm import tqdm
    iterations = tqdm(range(args.iter), 'gradient descent')

    ini = None


    for iter in iterations:

        optimizer.zero_grad()

        mvp = camera.look_at(
            eye = eye,
            center = torch.zeros(3),
            up = torch.tensor([0, 1.0, 0])
            )

        img = wg.render_polylines(
            (width, height),
            mvp = mvp,
            polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 8) ],
            stroke_width = stroke_width,
            num_samples = 8,
            num_edge_samples = 10000
            )

        diff = target - img

        loss = torch.mean(torch.abs(multi_level_diff(diff))) ## simple multi-level loss
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

    mvp = camera.look_at(
        eye = eye,
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )

    img = wg.render_polylines(
        (width, height),
        mvp = mvp,
        polylines = [ wg.cubic_basis_spline(points, knots=len(points) * 8) ],
        stroke_width = stroke_width,
        num_samples = 8
        ).detach()

    imsave.save_image(img, os.path.join(output_dir, f'final.png'))



