# Import Modules
import torch
from torch import nn


# VAE Model


class ConvolutionVariationalAutoEncoder(nn.Module):

    def __init__(self, z_dim=64):
        super().__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # flattening
        self.flattten = nn.Flatten()

        # de convolution
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # upsample
        self.upsample = nn.Upsample(size=(60, 42), mode='bilinear', align_corners=False)

        # encoder
        self.conv3_2mu = nn.Linear(128*8*6, z_dim)
        self.conv3_2log_var = nn.Linear(128*8*6, z_dim)

        # decoder
        self.z_2flattened = nn.Linear(z_dim, 128*8*6)
        self.deflatten = lambda x : x.view(-1, 128, 6, 8)

        # relu
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.flattten(h)

        mu, log_var = self.conv3_2mu(h), self.conv3_2log_var(h)
        return mu, log_var

    def decode(self,z):
        x = self.relu(self.z_2flattened(z))
        x = self.deflatten(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        x = self.upsample(x)

        return x

    def forward(self,x):
        # print("f1",x.shape)
        mu, log_var = self.encode(x)
        std = torch.exp(0.5*log_var)
       
        epsilon = torch.randn_like(std)
        z_reparametrized = mu + std*epsilon
        x_reconstructed = self.decode(z_reparametrized)

        return x_reconstructed, mu, log_var



