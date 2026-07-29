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

#### Latent Activation Function:

#### Dropout Rate:
In this case the best result was 0.0 dropout, as we are using a RandomAffine and this acts as a regulator. Adding dropout only lowers the quality of the model.

#### Latent Dimension:
A log scale was used to search for the best quality/compression ratio, the tested values are [2, 4, 8, 16, 32, 64]. The best result appeared to be dim=32 for this case (this depends on the dataset used, and the preferences of the problem) but for this problem it didn't had a big difference to dim=64, outperformed dim=16 and mantained a low parameter count.

#### Loss Function:
The following loss functions were tested:

- MSE
- L1
- SSIM
- SSIM + L1

### L1 Regularization
In this part of the study L1 regularization isn't needed so a low value was set (1e-10).

`{"id": "32", "act_function": nn.SiLU, "latent_act_function": nn.Tanh, "dropout_rate": 0.0, "latent_dim": 32, "loss_function": SSIML1Loss(0.4), "lambda_l1": 1e-10, "epochs": 19},`


