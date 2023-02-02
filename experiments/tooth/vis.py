import os

import matplotlib.pyplot as plt
import numpy as np

fig_dir = '/workspace/medicaldetectiontoolkit/tmp'
fig_pth = os.path.join(fig_dir, 'tmp.png')

os.makedirs(fig_dir, exist_ok=True)


def save_img_lbl(img, lbl, slice_idx, num_classes, axis_off=True, alpha=0.9, fig_size=(20, 10)):
    cmap = 'viridis'

    plt.figure("check", fig_size)

    plt.subplot(1, 2, 1)
    plt.title(f"image (slice: {slice_idx})")
    plt.imshow(img, cmap="gray")
    if axis_off:
        plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"image & label (slice: {slice_idx})")
    plt.imshow(img, cmap="gray")
    im_masked = np.ma.masked_where(lbl == 0, lbl)
    plt.imshow(
        im_masked,
        cmap,
        interpolation='none',
        alpha=alpha,
        vmin=1,
        vmax=num_classes
    )
    if axis_off:
        plt.axis('off')

    plt.tight_layout()

    plt.savefig(fig_pth)
