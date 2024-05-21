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
    ## This code computes the forward and gradient images to show
    ##   the backward pass of our polygon-filled rendering is correct.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=50, help='number of control points')
    parser.add_argument('-s', '--samples', type=int, default=2048, help='number of samples per pixel')
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    args = parser.parse_args()


    output_dir = './output/fill'
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = not args.cpu and torch.cuda.is_available() and wg.cuda.is_available()


    torch.manual_seed(0)


    num = args.num
    num_knots = 8 * num


    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    points = torch.cat((x,y,z), dim=1)


    if use_cuda:
        points = points.to('cuda')

    print('points.device =', points.device)



    width, height = 80, 72

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.look_at(
        eye = 6.0 * torch.tensor([1.0, 1.0, 1.0]),
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )


    points.requires_grad_()


    start = time.time()
    img = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ wg.cubic_basis_spline(points, knots=num_knots) ],
        num_samples = args.samples,
        num_cpu_threads = args.threads,
        use_hierarchy = True
        )
    time_forward = time.time() - start


    start = time.time()
    torch.sum(img).backward()
    time_backward = time.time() - start


    print('forward  time =', time_forward)
    print('backward time =', time_backward)
    print('gradient sum =', torch.sum(points.grad).item())

    img = img.detach()



    grad_image = torch.zeros(height, width, dtype=torch.float32)

    points.grad = None

    tmp = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ wg.cubic_basis_spline(points, knots=num_knots) ],
        num_samples = 1,
        num_cpu_threads = args.threads,
        use_hierarchy = True
        )

    from tqdm import tqdm

    for ih in tqdm(range(height), 'brute-force rendering of a gradient image'):
        for iw in range(width):
            torch.sum(tmp[ih, iw]).backward(retain_graph=True)
            grad_image[ih, iw] = torch.sum(points.grad[:,0]).detach().item()
            points.grad.zero_()


    ## rendering of a finite-difference image
    delta = 1e-2
    points = points.detach()
    points[:,0] += delta

    img_delta = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ wg.cubic_basis_spline(points, knots=num_knots) ],
        num_samples = args.samples,
        num_cpu_threads = args.threads,
        use_hierarchy = True
        )


    FD_image = torch.sum(img_delta - img, dim=-1) / delta




    imsave.save_image(img, os.path.join(output_dir, 'forward.png'))

    v_max = 40.0

    imsave.save_signed_image(
        img = grad_image,
        path = os.path.join(output_dir, 'backward.png'),
        v_max = v_max
        )

    ## Finite difference image should (exactly) match the backward image.
    ## In reality, there's slight diff. due to the inaccuracy of the finite difference method
    imsave.save_signed_image(
        img = FD_image,
        path = os.path.join(output_dir, 'finite_difference.png'),
        v_max = v_max
        )

