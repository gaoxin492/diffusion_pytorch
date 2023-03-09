# Diffusion PyTorch
This repository contains the PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM). The code is based on the original implementation in Jupyter Notebook, which can be found in the references.

## Requirements
· Python 3.7.15

· PyTorch

· NumPy

· Matplotlib

· tqdm

## Files
**data/:** Contains the MNIST dataset.

**figures/:** Contains some test images and generated images.

**params.py:** Calculates some of the basic parameters used in the model. Please refer to the original paper for the details.

**utils.py:** Includes several helper functions.

**test.py:** Applies noise to test images and displays the process.

**UNet.py:** A network used for predicting noise.

**train.py:** Includes the forward process of the model and training for the noise prediction network. We only trained for 50 epochs and did not carefully tune the parameters.

**inference.py:** Calls the trained model from train.py, generates random noise with the same size as the image, and generates images from the approximate original data distribution through the backward process.

## Usage
To train the model, simply run train.py. To generate images using the trained model, run inference.py. You can also modify the parameters and hyperparameters in params.py to experiment with different settings.

## References
《Denoising Diffusion Probabilistic Models》
https://zhuanlan.zhihu.com/p/572161541

## Acknowledgements
We would like to thank the original authors for their implementation in Jupyter Notebook, which provided valuable guidance for this implementation.).
