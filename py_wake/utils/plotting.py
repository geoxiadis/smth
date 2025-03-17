import matplotlib.pyplot as plt


def setup_plot(ax=None, title=None, ylabel=None, xlabel=None, xlim=None, ylim=None, grid=True, axis=None, figsize=None):
    ax = ax or plt.gca()
    if figsize:
        ax.figure.set_size_inches(figsize)
    if axis:
        ax.axis(axis)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid()
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    plt.tight_layout()
