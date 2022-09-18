import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
       
        # ConV & BatchNorm & LeakyReLU
        # beacasue we use batchNorm2D -> bias = False
        # padding_mode="reflect" -> supposedly use artifacts
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

# 64 -> 128 -> 256 -> 512
# 256 -> 30x30 (depends on your image size)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # initial block: no BatchNorm (hard to explain the reason)
        self.initial = nn.Sequential(
            # why in_channel * 2: it gets the input image x and output image y, instead of normal GAN only a number presenting FAKE or REAL
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]

        
        # Build layers except for  first layer
        for feature in features[1:]:
            layers.append(
                # stride = 2 in the last 512 layer
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        # add one more layer -> [1, 1, 26, 26], each grid with 0-1 to represent Real / fake
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        # put this layers list into nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # input x & label y are concanated in the first dimentsion        
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    # batch , channel, width, length
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()