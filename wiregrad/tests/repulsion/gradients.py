import torch
import numpy as np
import math
import time

import wiregrad as wg


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--show', action='store_true', help='launch viewer')
    parser.add_argument('-t', '--threads', type=int, default=20, help='number of cpu threads')
    args = parser.parse_args()


    num = 2000

    phi = 2.0 * math.pi * torch.arange(num) / num
    x = torch.cos(phi) + 2.0 * torch.cos(2.0 * phi)
    y = torch.sin(phi) - 2.0 * torch.sin(2.0 * phi)
    z = 2.0 * torch.sin(3.0 * phi)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    nodes = torch.cat((x,y,z), dim=1)


    d0 = 10.0

    nodes.requires_grad_()



    # brute-force
    start = time.time()

    energy = wg.repulsion_loss(nodes, splits='none', cyclic=True, d0=d0, num_cpu_threads=args.threads)
    energy.backward()

    energy = energy.detach()
    grad = nodes.grad.detach().clone()

    time_bruteforce = time.time() - start

    print('    energy =', energy.item())
    print('grad.sum() =', torch.sum(torch.abs(grad)).item())



    nodes.grad = None


    # accel. single-thread
    start = time.time()

    energy_accel = wg.repulsion_loss(nodes, splits='auto', cyclic=True, d0=d0, num_cpu_threads=1)
    energy_accel.backward()

    time_accel_st = time.time() - start

    energy_accel = energy_accel.detach()
    grad_accel = nodes.grad.detach()

    energy_diff = (torch.abs(energy - energy_accel) / torch.abs(energy)).item()
    grad_diff = (torch.max(torch.abs(grad - grad_accel)) / torch.max(torch.abs(grad))).item()

    print('energy_diff (accel. single) =', energy_diff)
    print('  grad_diff (accel. single) =', grad_diff)
    assert energy_diff < 1e-5
    assert grad_diff < 1e-5




    nodes.grad = None


    # accel. multi-thread
    start = time.time()

    energy_accel = wg.repulsion_loss(nodes, splits='auto', cyclic=True, d0=d0, num_cpu_threads=args.threads)
    energy_accel.backward()

    time_accel_mt = time.time() - start

    energy_accel = energy_accel.detach()
    grad_accel = nodes.grad.detach()

    energy_diff = (torch.abs(energy - energy_accel) / torch.abs(energy)).item()
    grad_diff = (torch.max(torch.abs(grad - grad_accel)) / torch.max(torch.abs(grad))).item()

    print(f'energy_diff (accel. {args.threads} threads) =', energy_diff)
    print(f'  grad_diff (accel. {args.threads} threads) =', grad_diff)
    assert energy_diff < 1e-5
    assert grad_diff < 1e-5



    if torch.cuda.is_available() and wg.cuda.is_available():
        print('testing CUDA computation')
        nodes = nodes.detach().cuda()
        nodes.requires_grad_()
        nodes.grad = None

        # CUDA
        start = time.time()

        energy_cuda = wg.repulsion_loss(nodes, splits='auto', cyclic=True, d0=d0)
        energy_cuda.backward()

        time_cuda = time.time() - start

        energy_cuda = energy_cuda.detach().cpu()
        grad_cuda = nodes.grad.detach().cpu()

        energy_diff = (torch.abs(energy - energy_cuda) / torch.abs(energy)).item()
        grad_diff = (torch.max(torch.abs(grad - grad_cuda)) / torch.max(torch.abs(grad))).item()
        print('energy_diff (cuda) =', energy_diff)
        print('  grad_diff (cuda) =', grad_diff)

        if  energy_diff >= 1e-5:
            print(f'WARNING: energy_diff exceeds 1e-5')
        if  grad_diff >= 1e-5:
            print(f'WARNING: grad_diff exceeds 1e-5')





    print(f'brute-force     elapsed = {time_bruteforce * 1000.0:.3f} ms')
    print(f'accel. (single) elapsed = {time_accel_st * 1000.0:.3f} ms ({100.0 * time_accel_st / time_bruteforce:.2f}% of brute-force)')
    print(f'accel. (multi)  elapsed = {time_accel_mt * 1000.0:.3f} ms ({100.0 * time_accel_mt / time_bruteforce:.2f}% of brute-force)')
    print(f'cuda            elapsed = {time_cuda * 1000.0:.3f} ms ({100.0 * time_cuda / time_bruteforce:.2f}% of brute-force)')


    if args.show:
        nodes = nodes.detach().cpu().numpy()
        d_nodes = -1.0 * grad.detach().cpu().numpy()

        edges = wg.polyline_edges(len(nodes), cyclic=True)
        edges = edges.detach().cpu().numpy()

        import polyscope as ps
        ps.init()
        curve = ps.register_curve_network('curve', nodes, edges, radius=2e-3)
        curve.add_vector_quantity('d_nodes', d_nodes, length=0.05, enabled = True)
        ps.show()


