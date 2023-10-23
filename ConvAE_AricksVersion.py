# Script to hold a simple implementation of a convolutional autoencoder useable for classification
# and reconstruction

import torch.nn as nn

class ConvAE_WithBuiltInClassification(nn.Module):
    def __init__(self, inputShape):
        super(ConvAE_WithBuiltInClassification, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(64, 32, 7)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Classifier
        self.dense1 = nn.Linear(in_features=?)

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_embedded = self.relu(self.conv3(x))

        # Decoder
        x = self.relu(self.deconv1(x_embedded))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x
