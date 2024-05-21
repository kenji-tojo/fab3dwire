import numpy as np
import torch
import os
import sys

import wiregrad as wg

sys.path.append('../..')
from utils import plotter


if __name__ == '__main__':
    ##
    ## (Internal) This code visualizes a checkerboard sampling pattern that is used for antialiasing.
    ##

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--samples', type=int, default=8, help='number of samples')
    args =parser.parse_args()

    samples = wg.checkerboard_pattern(num_samples=args.samples)

    plotter.init()

    xy = samples.transpose(1, 0).detach().cpu().numpy()

    fig, ax = plotter.create_figure(300, 300)

    ax.set_title(f'# of samples per pixel = {len(xy[0])}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(9) / 8)
    ax.set_yticks(np.arange(9) / 8)
    ax.scatter(xy[0], xy[1])

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    fname = os.path.join(output_dir, 'checkerboard.png')
    print('Saving the result to', fname)
    fig.savefig(fname, dpi=300)

