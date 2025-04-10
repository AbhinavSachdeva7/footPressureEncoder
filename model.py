# Import Modules
import torch
from torch import nn

# VAE Model
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, h_dim=512, z_dim=32): # Default h_dim updated based on train script
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2imd = nn.Linear(h_dim, h_dim)
        self.imd_2mu = nn.Linear(h_dim, z_dim)
        self.imd_2log_var = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2imd = nn.Linear(z_dim, h_dim)
        self.imd_2hid = nn.Linear(h_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        h = self.relu(self.hid_2imd(h))
        mu, log_var = self.imd_2mu(h), self.imd_2log_var(h)
        return mu, log_var

    def decode(self, z):
        h = self.relu(self.z_2imd(z))
        h = self.relu(self.imd_2hid(h))
        return self.hid_2img(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        z_reparametrized = mu + std*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, log_var 