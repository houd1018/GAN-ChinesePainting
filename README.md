# GAN-ChinesePainting
## DCGAN
Files in the models folder are models that trained in 3000 epochs
(The training ratio of generator to discriminator is 5:1).
## WGAN-GP
Hyperparameter c_lambda need to be tuned, according to the critic's loss
## High Resolution
Image size: 256 * 256  
add one more conv2D: 3 -> 64 -> 128 -> 256 -> 512 -> 1024
