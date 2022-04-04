# GAN-ChinesePainting

## DCGAN

Files in the models folder are models that trained in 3000 epochs
(The training ratio of generator to discriminator is 5:1).

## WGAN-GP

Hyperparameter c_lambda need to be tuned, according to the critic's loss

## High Resolution

Image size: 256 * 256  
add one more conv2D: 3 -> 64 -> 128 -> 256 -> 512 -> 1024

## CycleGAN

**Photo2ChinesePainting**

```
python test.py --dataroot datasets/<dataset_name>/ --cuda
```

This command will take the images under the *dataroot/test* directory and run test in all images in the folder. Output image will be saved in *output* directory.

```
python test_solo.py --cuda
```

This command will let user enter the Chinese Painting and Photo image path, and run test on single image. Output image will be saved in *output_solo* directory.
