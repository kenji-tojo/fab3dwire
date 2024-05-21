import numpy as np
import torch
import igl
import os
import sys
import os

import wiregrad as wg

sys.path.append('../..')
from utils import imsave


if __name__ == '__main__':
    ##
    ## This code shows an example of the triangle-mesh rendering using wiregrad.
    ## Note that the backward pass of the mesh rendering is currently not supported.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--samples', type=int, default=8, help='number of samples')
    parser.add_argument('-t', '--threads', type=int, default=32, help='number of cpu threads')
    args =parser.parse_args()

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)


    vtx, tri = igl.read_triangle_mesh('../../data/assets/bunny_8k.obj')
    nrm = igl.per_vertex_normals(vtx, tri)

    vtx = torch.from_numpy(vtx).type(torch.float32)
    nrm = torch.from_numpy(nrm).type(torch.float32)
    tri = torch.from_numpy(tri).type(torch.int32)

    lo, hi = 0.3, 0.9
    rgb = (hi - lo) * 0.5 * (nrm + 1.0) + lo

    rot = wg.rotation(angle=torch.tensor([90.0, 0, 0]))
    vtx = torch.matmul(vtx, rot.transpose(1, 0))


    width, height = 320, 270

    camera = wg.Camera()
    camera.aspect = width / height

    mvp = camera.rotate(
        eye = torch.tensor([-0.2, -4, 1.5]),
        angle = torch.tensor([80.0, 0, 0])
        )

    img = wg.debug.render_triangles(
        (width, height),
        mvp, vtx, tri,
        colors = rgb,
        num_samples = args.samples,
        num_cpu_threads = args.threads
        ).detach().cpu().numpy()


    imsave.save_image(img, os.path.join(output_dir, 'bunny.png'))


