import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


from tqdm import tqdm
import os, shutil
import igl

import wiregrad as wg

from visual_loss import *
from utils import imsave, trefoil, parse_config, save_polyline


if __name__ == '__main__':
    ##
    ## A wire-shape optimization example using a 3D mesh input.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', default='./data/config/bunny.json', help='path to the input config file')
    parser.add_argument('--no_video', action='store_true', help='not creating video')
    parser.add_argument('--cpu', action='store_true', help='force to use CPU')
    args = parser.parse_args()


    ## Target images specified in ./data/config/bunny.json were rendered using Blender.
    ## The eye position and rotation was the same as ones used in Blender.
    ## The default focal length 50mm of Blender shows a good match with fov 40deg,
    ##   which we use in this code to compute the mvp matrix.


    config_name = os.path.splitext(os.path.basename(args.config))[0]
    print('config_name =', config_name)


    output_dir = os.path.join('./output/mesh', config_name)
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


    use_cuda = not args.cpu and torch.cuda.is_available() and wg.cuda.is_available()

    if use_cuda:
        print('use', torch.cuda.get_device_name())
        device = 'cuda'
    else:
        device = 'cpu'



    torch.manual_seed(0)
    np.random.seed(0)


    config = parse_config(args.config)


    # a color used for silhouette and line renderings
    color = (0.1, 0.1, 0.1)

    # a constant stroke width to render lines
    stroke_width = torch.tensor([1.6])




    ## loading target mesh
    msh_vtx, msh_tri = igl.read_triangle_mesh(config.mesh.path)
    msh_vtx = torch.from_numpy(msh_vtx).type(torch.float32)
    msh_tri = torch.from_numpy(msh_tri).type(torch.int32)

    msh_vtx = torch.matmul(msh_vtx, wg.rotation(torch.tensor(config.mesh.rotation, dtype=torch.float32)).t())
    msh_cen = 0.5 * (msh_vtx.max(dim=0).values + msh_vtx.min(dim=0).values)


    kernel_size = config.guidance.kernel_size
    sigma = config.guidance.sigma

    cam = wg.Camera()
    cam.aspect = 1.0
    cam.fovy = 40.0
    cam.near = 1e-2
    cam.far = 100.0



    views = []

    for i,view in enumerate(config.views):

        data = dict()

        eye = torch.tensor(view.eye).view(3)
        angle = torch.tensor(view.rotation).view(3)
        data['mvp'] = cam.rotate(eye, angle)

        target = get_target(os.path.join(view.target.path), device=device)
        data['target'] = target.detach()

        imsave.save_image(target.squeeze(0).permute(1, 2, 0), os.path.join(output_dir, f'target_{i}.png'))


        target_mask = wg.debug.render_triangles(
            config.guidance.resolution,
            mvp = data['mvp'],
            vertices = msh_vtx,
            triangles = msh_tri,
            colors = torch.tensor((color), dtype=torch.float32)
            ).permute(2,0,1).unsqueeze(0).to(device)

        data['silhouette_loss'] = SilhouetteLoss(target_mask, config.guidance.kernel_size, config.guidance.sigma)

        imsave.save_image(data['silhouette_loss'].target_mask.squeeze(0).permute(1, 2, 0), os.path.join(output_dir, f'target_mask_{i}.png'))

        views.append(data)



    points = trefoil(config.curve.num_controls)

    ## rotate to reproduce the coordinates in our exepriments
    points = torch.matmul(points, wg.rotation(torch.tensor([0, 90.0, 0])).t())

    points = float(config.curve.radius) * points + msh_cen.unsqueeze(0)

    points = points.to(device)
    points.requires_grad_()

    num_knots = config.curve.num_nodes


    resolution = (224, 224)

    for i,data in enumerate(views):

        mvp = data['mvp']

        img = wg.render_polylines(
            resolution,
            mvp = mvp,
            polylines = [ wg.cubic_basis_spline(points, knots=num_knots) ],
            color = color,
            stroke_width = stroke_width
            )

        imsave.save_image(img, os.path.join(output_dir, f'initial_{i}.png'))




    optimier = wg.ReparamVectorAdam(
        points = points,
        unique_edges = wg.polyline_edges(len(points), cyclic=True),
        step_size = 1e-4,
        reparam_lambda = 0.05
        )


    iterations = config.iterations
    print('Iteration count =', iterations)
    print('Temporary and final results are saved to', output_dir)


    ## semantic loss for image inputs
    semantic_loss = CLIPConvLoss(device=device, clip_conv_layer=3)

    video_frame_id = 0


    for iter in tqdm(range(iterations), 'gradient descent'):


        ## Silhouette loss
        nodes = wg.cubic_basis_spline(points, knots=num_knots)
        loss = 0.0

        for i,view in enumerate(views):

            img = wg.render_filled_polygons(
                config.guidance.resolution,
                mvp = view['mvp'],
                polygons = [ nodes ],
                color = color,
                num_edge_samples = 10000
                ).permute(2,0,1).unsqueeze(0)

            ## Please text-search 'silhouette_loss' in this file for the definition
            loss += view['silhouette_loss'](img)

        loss *= (1.0 / len(views))
        loss.backward() # flush loss and free internal buffers


        ## Semantic loss
        nodes = wg.cubic_basis_spline(points, knots=num_knots)
        loss = 0

        for i, view in enumerate(views):

            img = wg.render_polylines(
                resolution,
                mvp = view['mvp'],
                polylines = [ nodes ],
                color = color,
                stroke_width = stroke_width,
                num_edge_samples = 10000
                ).permute(2,0,1).unsqueeze(0)

            loss_dict = semantic_loss(img, view['target'])

            for key in loss_dict:
                loss += 1e1 * loss_dict[key]

            if iter % 10 == 0 or iter == iterations - 1:
                img = img.detach().squeeze(0).permute(1,2,0)
                imsave.save_image(img, os.path.join(output_dir, f'current_{i}.png'),logging=False)

                save_polyline(points = points.detach(), path = os.path.join(output_dir, f'controls.obj'))

                if not args.no_video and iter % 10 == 0:
                    imsave.save_image(img, os.path.join(tmp_dir, f'img_{video_frame_id:04d}_{i}.png'),logging=False)
                    if i == len(views) - 1:
                        video_frame_id += 1

        loss *= (1.0 / len(views))
        loss.backward() # flush loss and free internal buffers


        # Fabrication and geometric losses
        nodes = wg.cubic_basis_spline(points, knots=num_knots)
        loss = 0

        loss += 1e-3 * wg.bending_loss(nodes, cyclic=True)
        loss += 3e2 * wg.tetrahedron_loss(nodes, cyclic=True)
        loss += wg.repulsion_loss(nodes, d0=0.2, cyclic=True) ## d0 is the size of repulsion support

        loss += 1e1 * wg.uniform_distance_loss(points, cyclic=True)
        loss += wg.length_loss(nodes, cyclic=True)
        # loss += wg.scale_loss(nodes)

        loss.backward()


        ## Gradients occasionally contain NaNs when the spacing of nodes gets very small.
        ## I suspect they are from the torch.sqrt function in length computation,
        ##   which produces inf in the backward when the forward pass outputs 0.
        ## Here's a simple remedy:
        with torch.no_grad():
            points.grad = torch.nan_to_num(points.grad)

        optimier.step()
        optimier.zero_grad()

        torch.cuda.empty_cache()




    if not args.no_video:

        import subprocess

        for i in range(len(views)):

            subprocess.call(['ffmpeg', '-framerate', '60', '-y', '-i',
                            str(os.path.join(tmp_dir, f'img_%4d_{i}.png')), '-vb', '20M',
                            '-vcodec', 'libx264',
                            str(os.path.join(output_dir, f'view_{i}.mp4'))])


    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)



