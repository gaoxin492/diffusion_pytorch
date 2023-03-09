import torch
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from train import q_sample

def make_transform(image_size):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),
    ])
    return transform

def reverse_transform(x):
    reverse_t = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    return reverse_t(x)

def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)
    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(250,50), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig("/root/autodl-tmp/diffusion/figures/12.png")
    #plt.show()

if __name__=="__main__":

    image = Image.open("/root/autodl-tmp/diffusion/figures/cat.jpg")
    image_size = 128
    transform = make_transform(image_size)
    x_start = transform(image).unsqueeze(0)
    print(x_start.shape)

    # use seed for reproducability
    torch.manual_seed(0)
    plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

