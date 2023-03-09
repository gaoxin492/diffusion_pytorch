import matplotlib.pyplot as plt
# sample 64 images
import torch
from tqdm import tqdm
from params import sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance, timesteps
from UNet import Unet

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

if __name__=='__main__':

    model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(torch.load("/root/autodl-tmp/diffusion/diffusion1.pth"))

    model.eval()
    with torch.no_grad():
        samples = sample(model, image_size=28, batch_size=64, channels=1)
        # show a random one
        random_index = 5
        plt.imshow(samples[-1][random_index].reshape(28, 28, 1), cmap="gray")
        plt.savefig("/root/autodl-tmp/diffusion/figures/inference.png")