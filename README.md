# MNIST Autoencoder Study
---
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

![Activation functions](/images/ae/activation_functions.png)

#### Latent Activation Function:

![Latent activation functions](/images/ae/latent_activation_functions.png)

#### Dropout Rate:
In this case the best result was 0.0 dropout, as we are using a RandomAffine and this acts as a regulator. Adding dropout only lowers the quality of the model.

![Dropout losses while training](/images/ae/dropout_training.png)

Dropout evaluation:
![Dropout evaluation](/images/ae/dropout.png)

#### Latent Dimension:
A log scale was used to search for the best quality/compression ratio, the tested values are [2, 4, 8, 16, 32, 64]. The best result appeared to be dim=32 for this case (this depends on the dataset used, and the preferences of the problem) but for this problem it didn't had a big difference to dim=64, outperformed dim=16 and mantained a low parameter count.

#### Loss Function:
The following loss functions were tested:

- MSE
- L1
- SSIM
- SSIM + L1

The optimal solution appeared to be SSIM + L1 loss. The combination of structural similarity (SSIM) and pixel precision (L1) gave the best results as loss function. The optimal results used ```alpha=0.4``` (L1 * 04 + SSIM * 06).

![Loss functions](/images/ae/loss_functions.png)

#### L1 Regularization
In this part of the study L1 regularization isn't needed so a low value was set (1e-10).

#### Optimal Configuration

`{"id": "Optimal", "act_function": nn.SiLU, "latent_act_function": nn.Tanh, "dropout_rate": 0.0, "latent_dim": 32, "loss_function": SSIML1Loss(0.4), "lambda_l1": 1e-10, "epochs": 19},`

- Original image size: 28 $\times$ 28 = 784
- Latent vector size: 32
- Compression ratio: $\frac{784}{32}$ = 24.5
- Reduction: (1 - $\frac{32}{784}$) $\times$ 100 $\approx$ 95.92%

Reconstructed images using the optimal configuration:

![Reconstructed images](/images/ae/reconstructed.png)

UMAP proyection of the latent space:

![UMAP proyection](/images/ae/umap.png)




