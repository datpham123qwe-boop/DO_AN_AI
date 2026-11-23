"""Export qualitative panels and row-normalized confusion matrices."""

import numpy as np

from src.evaluation.error_maps import ERROR_NAMES, ERROR_PALETTE, colorize_error_map, make_error_map
from src.spraying.spray_mask import make_spray_mask
from src.visualization.visualize import PALETTE


def colorize_labels(mask):
    rgb = np.full((*mask.shape, 3), 160, dtype=np.uint8)
    valid = mask != 255
    rgb[valid] = PALETTE[mask[valid]]
    return rgb


def save_qualitative_figure(path, image, pred, target=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = [("RGB", image)]
    if target is not None:
        panels.append(("Ground truth", colorize_labels(target)))
    panels.append(("Prediction", colorize_labels(pred)))
    if target is not None:
        panels.append(("Error map", colorize_error_map(make_error_map(pred, target))))
    spray = make_spray_mask(pred, target)
    panels.append(("Spray mask", spray.astype(np.uint8) * 255))
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.7))
    for axis, (name, panel) in zip(np.atleast_1d(axes), panels):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    if target is not None:
        handles = [Patch(color=color / 255, label=name) for name, color in zip(ERROR_NAMES, ERROR_PALETTE)]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_confusion_figure(path, matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.datasets.label_mapping import CLASS_NAMES

    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Prediction", ylabel="Ground truth")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:,}\n{normalized[row, col]:.1%}",
                    ha="center", va="center", color="white" if normalized[row, col] > 0.5 else "black", fontsize=8)
    fig.colorbar(plot, ax=ax, label="Fraction of GT class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
