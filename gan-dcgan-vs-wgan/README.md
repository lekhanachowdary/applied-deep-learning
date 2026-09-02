# DCGAN vs. WGAN on CIFAR-10

A side-by-side comparison of two generative adversarial network formulations
trained on CIFAR-10:

- **DCGAN**, the standard convolutional GAN with a binary cross-entropy
  discriminator.
- **WGAN**, which replaces the discriminator with a critic that estimates the
  Wasserstein distance and uses weight clipping for the Lipschitz constraint.

Both models are trained for 50 epochs. For each epoch the run saves a grid of
generated samples, the generator and discriminator losses, and the Frechet
Inception Distance between generated and real images.

## Contents

| Path | What it holds |
| --- | --- |
| `dcgan_vs_wgan.ipynb` | Loads the saved metrics and renders the comparison plots (FID over epochs, loss curves, real vs. generated grids) |
| `Output/WGAN_Fake`, `Output/WGAN_Real` | Per-epoch WGAN sample grids |
| `Results/DCGAN_FAKE`, `Results/DCGAN_REAL` | Per-epoch DCGAN sample grids |
| `Results/FID_score/*.npy` | FID per epoch for each model |
| `Results/Loss_Data/*.npy` | Generator and discriminator loss per step for each model |
| `Results/DCGAN_Real_vs_Fake_Comparison.jpg` | Final DCGAN real vs. generated comparison |

## Setup

1. CIFAR-10 downloads automatically via `torchvision`, or fetch it from
   https://www.cs.toronto.edu/~kriz/cifar.html
2. The trained generators are too large for the repo. Download them if you want
   to sample without retraining:
   - DCGAN: https://drive.google.com/file/d/1kFqFG7IhIfdAv3u8ShYd1FPe0qA5WSxW/view
   - WGAN: https://drive.google.com/file/d/13VRl2Fa-Q3VSMg4HFcksL-htnqkd-TDR/view
3. Open `dcgan_vs_wgan.ipynb` to regenerate the plots from the saved `.npy`
   metrics.
