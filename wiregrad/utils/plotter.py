from typing import Any, Tuple
import matplotlib.pyplot as plt


def init(usetex = False):
    if usetex:
        plt.rcParams["font.family"] = "Times"
        plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 9})

def clf():
    plt.clf()

def _init_ax(ax: Any, labelsize=7.0):
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True)

    ax.tick_params(
        length = 2.0,
        width = 0.5,
        labelsize = labelsize,
        pad = 1.5,
        grid_color = (1.0, 1.0, 1.0),
        grid_linewidth = 0.7
        )

def create_figure(
    width_in_points: float,
    height_in_points: float,
    labelsize = 7.0,
    alpha = 1.0,
    nrows = 1,
    ncols = 1
    ) -> Tuple[plt.Figure, plt.Axes]:

    fig, axes = plt.subplots(nrows, ncols)
    fig.set_size_inches(w=width_in_points/72.0, h=height_in_points/72.0)
    fig.patch.set_alpha(alpha)

    if nrows == 1 and ncols == 1:
        _init_ax(axes, labelsize=labelsize)

    elif nrows > 1 and ncols > 1:
        for row in axes:
            for ax in row:
                _init_ax(ax, labelsize=labelsize)

    else:
        assert nrows == 1 or ncols == 1
        assert nrows > 1 or ncols > 1
        for ax in axes:
            _init_ax(ax, labelsize=labelsize)

    return fig, axes



