# MNIST Autoencoder Study

#### Disclaimer
This project was developed as a learning and research exercise to understand better how neural networks work. Feel free to use the code as you wish.

## Introduction
This study is designed to search the best configurations for a set of AE architectures in the MNIST dataset. In order to add some difficulty, a transform will be applied to the whole dataset which will change the rotation, translation and scale of the images:
```python
transforms.RandomAffine(
    degrees=10,
    translate=(0.04, 0.04),
    scale=(0.96, 1.04),
    interpolation=InterpolationMode.BILINEAR
)
```

In this study first an optimal configuration for the architecture has to be found and analysed. This is made via experimentation with `autoencoder_training_lab.ipynb`, which is especially designed to simplify and automate this task. Once found an optimal configuration and having checked the statistics of the resulting model, it can be benchmarked.

To ensure the reliability of the results, each parameter was tested with three different loss functions (MSE, L1 and SSIM) and the losses were averaged over 10 trainings to reduce variation. It is important to keep in mind that L1 and SSIM both offer far better results than MSE in image reconstruction.


## Parts Of The Study

- [x] Traditional autoencoder (image compresion).
- [ ] Sparse autoencoder (feature extraction).
- [ ] Variatonal autoencoder (image generation).
- [ ] Denoising Autoencoder (image modifying).

## TODO
- [ ] VGG Loss
- [ ] Clean and unify code

## Traditional Autoencoder

This part of the study is based in the traditional AE architecture which's main usage is compression of the image's data. The encoder uses a convolutional architecture to simplify the training and increase the performance.

#### Activation Function:
Three different activation functions were tested: ReLU, Leaky ReLU and SiLU. As expected, SiLU gave the best results because of its smoother curve and the capacity to mantain negative numbers.

<p align="center">
    <img src="/images/ae/activation_functions.png" width="800">
</p>

#### Latent Activation Function:

<p align="center">
    <img src="/images/ae/latent_activation_functions.png" width="800">
</p>

#### Dropout Rate:
In this case the best result was 0.0 dropout, as we are using a RandomAffine and this acts as a regulator. Adding dropout only lowers the quality of the model.

<p align="center">
    <img src="/images/ae/dropout_training.png" width="800">
</p>

Dropout evaluation:

<p align="center">
    <img src="/images/ae/dropout.png" width="800">
</p>

#### Latent Dimension:
A log scale was used to search for the best quality/compression ratio, the tested values are [2, 4, 8, 16, 32, 64]. The best result appeared to be dim=32 for this case (this depends on the dataset used, and the preferences of the problem) but for this problem it didn't had a big difference to dim=64, outperformed dim=16 and mantained a low parameter count.

#### Loss Function:
The following loss functions were tested:

- MSE
- L1
- SSIM
- SSIM + L1

The optimal solution appeared to be SSIM + L1 loss. The combination of structural similarity (SSIM) and pixel precision (L1) gave the best results as loss function. The optimal results used ```alpha=0.4``` (L1 * 04 + SSIM * 06).

<p align="center">
    <img src="/images/ae/loss_functions.png" width="800">
</p>

#### Optimal Configuration:

```python
{
    "id": "Optimal",
    "act_function": nn.SiLU,
    "latent_act_function": nn.Tanh,
    "dropout_rate": 0.0,
    "latent_dim": 32,
    "loss_function": SSIML1Loss(0.4),
    "epochs": 19
}
```

- Original image size: 28 $\times$ 28 = 784
- Latent vector size: 32
- Compression ratio: $\frac{784}{32}$ = 24.5
- Reduction: (1 - $\frac{32}{784}$) $\times$ 100 $\approx$ 95.92%

Reconstructed images using the optimal configuration:

<p align="center">
    <img src="/images/ae/reconstructed.png" width="800">
</p>

UMAP proyection of the latent space:

<p align="center">
    <img src="/images/ae/umap.png" width="600">
</p>

## Sparse Autoencoder (SAE)

This part of the study differs significantly from the previous one in both the objetive of the model and the approach taken. A SAE doesn't persue to compress the information into a reduced latent space, instead it's intention is to project the information into a larger latent space so it can recognise individual characteristics. The same convolutional architecture of the previous experiment, as well as some of the optimal results seen, are used in this section.

To avoid dead neurons (common in this architecture) we will add some of the complete reconstruction loss to the sparse one. This method is commonly used to avoid gradient inactivity. 

```python
loss = criterion(reconstructed_topk, images) + t * criterion(reconstructed_dense, images)
```


#### Fixed parameters:

The following configurations are mantained from the previous results as the SAE does not require changes and this parameters have shown to be optimal for this architecture.

```python
{
    "act_function": nn.SiLU,
    "dropout_rate": 0.0,
    "loss_function": SSIML1Loss(0.4),
    "epochs": 19
}
```

#### Latent Activation Function:

In a SAE we need to force sparsity, this means turning off many of the avaliable neurons. Because of this, we need to use `nn.ReLU` as the latent activation function so the negative neurons get turned off.


#### Top-k And Latent Dimension (Sparsity):
There are many ways to force the sparsity in a AE, the more popular ones are:
- L1 Reguralization
- KL divergence
- Top-K
- ReLU/Jump ReLU

For this project I have choosen Top-K + ReLU, I have tried L1 too but it didn't force a correct sparsity and tended to reduce all neurons at once. On the other hand Top-K offers a good control over the number of characteristics and creates great sparsity.


Top-K work's by forcing only the k neurons with greater activation values to stay active. This method requires to use ReLU as negatives values can become ambiguous in both training and analysis. This approach only has a remarcable issue, <ins>**dead neurons**</ins>. Top-K turns off neurons, reducing their gradients to 0, which added to ReLU can cause heavy issues. It's very important to select appropiate values for both k and the latent dimension.

Objectives:
- K value: as low as posible, low k forces sparsity and disentanglement in the latent space.
- Latent dimension: as high as posible, higher dimensions offer finer characteristics. 

Limitations:
- K value
    - The lower it goes the lower the quality of the reconstruction
    - If it's not consistent with the latent dimension it can explode the number of death neurons.
- Latent dimension:
    - There's a point from where more dimension doesn't mean a great impact in reconstruction quality (but it greatly reduces performance).

To start discarding configurations a width spectrum test is started:

<p align="center">
    <img src="/images/sae/train_losses/train_losses_1024-4096.png" width="800">
    <img src="/images/sae/losses_1024-4096.png" width="800">
</p>


With this results it is obvious that k=16 and dim=4096 can be easily descarded. We can extract some early conclusions from this results, it appears that the MNIST dataset (with our custom transform applied) need a minimun of 32-64 individual characteristics to be correctly reconstructed. It also appears that dim=2048 hits the peak of feature extracting, adding more dimensions have shown not only to not increase the quality but worse it. This can happen for many reasons but the main one is that the features become such fine and abstract that the number of neurons needed to reconstruct a digit explodes.

Then, some more specific tests are run to explore the greatest configurations:

<p align="center">
    <img src="/images/sae/train_losses/train_losses_1024-2048.png" width="800">
    <img src="/images/sae/losses_1024-2048.png" width="800">
    <img src="/images/sae/dead_neurons/dead_neurons_k=32.png" width="410">
    <img src="/images/sae/dead_neurons/dead_neurons_k=64.png" width="410">
</p>

A normal dead neurons rate it's between 4-6% in SAE architectures, the only architecture which mantains a low error and keeps the dead neurons count low is the combination `{k=32, dim=1024}`.

#### Dead Neurons:
As said at the start of the section, to control the dead neuron count a fraction of the normal reconstruction loss is added to the sparse one. This adding is controled by a parammeter `t` which default value has been `t=0.15`. A test to itter the `0.05`, `0.1`,  `0.2` values will be run to study the optimal value.

<p align="center">
  <b>t = 0.05</b><br>
  <img src="/images/sae/dead_neurons/0.05-dead-neurons.png" width="600">
</p>

<p align="center">
  <b>t = 0.10</b><br>
  <img src="/images/sae/dead_neurons/0.1-dead-neurons.png" width="600">
</p>

<p align="center">
  <b>t = 0.15</b><br>
  <img src="/images/sae/dead_neurons/0.15-dead-neurons.png" width="600">
</p>

<p align="center">
  <b>t = 0.20</b><br>
  <img src="/images/sae/dead_neurons/0.2-dead-neurons.png" width="600">
</p>
