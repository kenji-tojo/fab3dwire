from typing import Union
import numpy as np
import torch
import imageio


def save_image(
    img: Union[torch.Tensor, np.ndarray],
    path: str,
    logging = True
    ):

    if logging:
        print('saving image to', path)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.rint(img * 255).clip(0, 255).astype(np.uint8)

    imageio.imsave(path, img)


def save_signed_image(
    img: Union[torch.Tensor, np.ndarray],
    path: str,
    v_max = 50.0,
    logging = True
    ):

    if logging:
        print('saving image to', path)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    from matplotlib import cm
    img = np.clip((img + v_max) / (2.0 * v_max), a_min=0, a_max=1.0)
    img = cm.seismic(img)[:,:,:3]
    img = np.rint(img * 255).clip(0, 255).astype(np.uint8)

    imageio.imsave(path, img)


