import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Default input: 3 channels colored imgs
        # Default output: 3 channels colored imgs

        # Initial Convolution Block
        model = [nn.ReflectionPad2d(3),
            nn.Conv2d(3,64,7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)]

        #Downsampling
        features = 64
        for i in range(2):
            model += [nn.Conv2d(features, features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True)]
            features = features*2

        # Residual blocks
        # Default 9 layes
        for i in range(9):
            model += [ResidualBlock(features)]

        # Upsampling
        for i in range(2):
            model += [nn.ConvTranspose2d(features, features//2, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(features//2),
                        nn.ReLU(inplace=True)]
            features = features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 3, 7),
                    nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Default input: 3 channels colored imgs

        self.model = nn.Sequential(
                    nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(512, 1, 4, padding=1),
                )

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x,1)
        return x
