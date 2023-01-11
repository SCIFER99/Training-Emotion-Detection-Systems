# By: Tim Tarver

# Emotion Net file contributing to training the real-
# time emotion detection system.

import torch.nn as nn
import torch.nn.functional as functional

# We begin to create the EmotionNet Class creating the foundation
# of the emotion detection system!

class EmotionNet(nn.Module):

    network_configuration = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    # We want to develop its features through a specific number of
    # channels and classes throughout the system with a fixed probability.

    def __init__(self, number_of_channels, number_of_classes):

        super(EmotionNet, self).__init__()
        self.features = self.make_layers(number_of_channels,
                                          self.network_configuration)
        self.classifier = nn.Sequential(nn.Linear(6*6*128, 64),
                                        nn.ELU(True),
                                        nn.Dropout(probability=0.5),
                                        nn.Linear(64, number_of_classes))

    # Now, we create a function to instruct Pytorch how to execute the defined
    # layers in the network.

    def forward(self, x):

        out = self.features(x)
        out1 = out.view(out.size(0), -1)
        out2 = functional.dropout(out1, probability1 = 0.5, training=True)
        out3 = self.classifier(out2)
        return out3

    # Generate the Convolutional Layers within the Network with the make_layers
    # function to receive the in_channel and network configuration.

    def make_layers(self, in_channels, configuration):

        layers = []
        
        for x in configuration:

            if x == 'M':

                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]

            else:

                layers += [nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                                     nn.BatchNorm2d(x),
                                     nn.ELU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)                   
            
