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
    ## (Internal) This code tests the hierarchical acceleration of polygon-fill rendering
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=500, help='number of nodes')
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




    width, height = 80, 72

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.look_at(
        eye = 6.0 * torch.tensor([1.0, 1.0, 1.0]),
        center = torch.zeros(3),
        up = torch.tensor([0, 1.0, 0])
        )


    start = time.time()
    img_bruteforce = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ nodes ],
        num_samples = args.samples,
        num_cpu_threads = 1,
        use_hierarchy = False
        )
    time_bruteforce = time.time() - start


    start = time.time()
    img_accel_single = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ nodes ],
        num_samples = args.samples,
        num_cpu_threads = 1,
        use_hierarchy = True
        )
    time_accel_single = time.time() - start


    start = time.time()
    img_accel_multi = wg.render_filled_polygons(
        (width, height),
        mvp = mvp,
        polygons = [ nodes ],
        num_samples = args.samples,
        num_cpu_threads = args.threads,
        use_hierarchy = True
        )
    time_accel_multi = time.time() - start


    if use_cuda:
        print('testing CUDA')

        start = time.time()
        img_cuda = wg.render_filled_polygons(
            (width, height),
            mvp = mvp,
            polygons = [ nodes.cuda() ],
            num_samples = args.samples,
            use_hierarchy = True
            )
        time_cuda = time.time() - start


    print(f'brute-force            time = {time_bruteforce * 1000.0:.3f} ms')
    print(f'accel. (single-thread) time = {time_accel_single * 1000.0:.3f} ms ({100.0 * time_accel_single / time_bruteforce:.2f}% of brute-force)')
    print(f'accel.  (multi-thread) time = {time_accel_multi * 1000.0:.3f} ms ({100.0 * time_accel_multi / time_bruteforce:.2f}% of brute-force)')

    if use_cuda:
        print(f'CUDA                   time = {time_cuda * 1000.0:.3f} ms ({100.0 * time_cuda / time_bruteforce:.2f}% of brute-force)')


    torelance = 1e-5

    diff = torch.max(torch.abs(img_bruteforce - img_accel_single)).item()
    print('diff (accel. single) =', diff)
    assert diff < torelance

    diff = torch.max(torch.abs(img_bruteforce - img_accel_multi)).item()
    print('diff (accel.  multi) =', diff)
    assert diff < torelance

    if use_cuda:
        diff = torch.max(torch.abs(img_bruteforce - img_cuda.detach().cpu())).item()
        print('diff (cuda)          =', diff)

        if diff >= torelance:
            print(f'WARNING: CUDA diff exceeds the torelance (= {torelance:.1E})')



    imsave.save_image(img_accel_multi, os.path.join(output_dir, 'trefoil_fill.png'))



