#### A denoising project based on the autoencoder architecture trained on the mnist dataset
## Architecture
![image](https://user-images.githubusercontent.com/80089456/177017307-f07b8966-5afa-448e-9229-757304d25fe4.png)

The encoder and decoder each has three layers for compressing input and reconstructing the input from latent space.
The use of ReLu is to provide non-linearity and tanh is to map the values to -1 to 1.
The sigmoid in the end is used to force the nodes to 0 to 1.

## Learning Curve
![Denoiser Training Error, Loss_0 012422](https://user-images.githubusercontent.com/80089456/177017325-33722629-b667-4e3c-a9a5-5647453fd6a6.png)

## Results
![deoiser sample New 2](https://user-images.githubusercontent.com/80089456/177017356-81bd004b-b1d8-4c22-b6e4-62905783a8f9.png)
