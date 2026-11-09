import numpy as np
import matplotlib.pyplot as plt

PALETTE = np.array([[101, 67, 33],    # Background: nâu
                    [0, 200, 0],      # Crop: xanh lá
                    [220, 0, 0]],     # Weed: đỏ
                   dtype=np.uint8)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def colorize_mask(mask):
    return PALETTE[mask]            # mask giá trị 0/1/2 -> [H,W,3]

def overlay(image, mask, alpha=0.5):
    color = colorize_mask(mask)
    return (image * (1 - alpha) + color * alpha).astype(np.uint8)

def denormalize(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def plot_samples(dataset, n=4):
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        s = dataset[i]
        img = denormalize(s["image"])
        msk = s["mask"].numpy()
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(colorize_mask(msk))
        ax[i, 2].imshow(overlay(img, msk))
        for a in ax[i]:
            a.axis("off")
    plt.tight_layout()
    plt.show()