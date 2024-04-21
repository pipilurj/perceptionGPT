import torch
import torch.nn as nn
import torchvision.models as models

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Image extractor using ResNet
        self.image_extractor = models.resnet18(pretrained=True)
        self.image_extractor.fc = nn.Identity()  # Remove the fully connected layer

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode the input image
        features = self.image_extractor(x)

        # Reshape the features
        features = features.view(features.size(0), -1, 1, 1)

        # Decode the features
        reconstructed_image = self.decoder(features)

        return features, reconstructed_image
