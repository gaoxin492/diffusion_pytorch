import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from inference import sample,extract
from params import sqrt_one_minus_alphas_cumprod, timesteps, sqrt_alphas_cumprod
from UNet import Unet
from torchvision.utils import save_image
import torch.nn.functional as F


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    # 先采样噪声
    if noise is None:
        noise = torch.randn_like(x_start)

    # 用采样得到的噪声去加噪图片
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    # 根据加噪了的图片去预测采样的噪声
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def num_to_groups(num, divisor): #4,128
    groups = num // divisor    #0
    remainder = num % divisor  #4
    arr = [divisor] * groups   #[128,128,128,128]
    if remainder > 0:
        arr.append(remainder)
    return arr

def train(dataloader, results_folder, epochs, lr):
    results_folder.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("—————————第{}轮训练开始啦————————".format(epoch+1))
        for step, (batch, _) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            if step>0 and step % 100 == 0:
                print("epoch{},step{},Loss:{}".format(epoch + 1, step + 1, loss.item()))

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "/root/autodl-tmp/diffusion/diffusion1.pth", _use_new_zipfile_serialization=True)


if __name__=='__main__':

    transform = Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    results_folder = Path("results")
    train(dataloader, results_folder, 50, 3e-4)

