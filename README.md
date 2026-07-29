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
- [ ] Same scale in the loss graphs

## Traditional Autoencoder

This part of the study is based in the traditional AE architecture which's main usage is compression of the image's data. The encoder uses a convolutional architecture to simplify the training and increase the performance.


