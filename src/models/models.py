import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor = self.scale_factor)
        return x
    
class DSVDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(128 * 4 * 4, 128, bias=False)
        self.rep_dim = 128

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
    
        h = self.fc(out)

        return h

class VAE(nn.Module):
    def __init__(self, rep_dim, encoding_dim, L=10):
        super().__init__()
        self.L = L  # the number of reparameterization
        self.rep_dim = rep_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(int(128 / (4 * 4)), 128, 5, padding=2),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            Interpolate(2),
            nn.ConvTranspose2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            Interpolate(2),
            nn.ConvTranspose2d(64, 32, 5, padding=2),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(negative_slope=0.1),
            Interpolate(2),
            nn.ConvTranspose2d(32, 3, 5, padding=2),
            nn.Sigmoid()
        )

        self.fc1_mu = nn.Linear(encoding_dim, self.rep_dim)
        self.fc1_logvar = nn.Linear(encoding_dim, self.rep_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1_mu(x)
        logvar = self.fc1_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = z.view(z.size(0), int(self.rep_dim / 16), 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        recon_list = []
        for l in range(self.L):
            z = self.reparameterize(mu, logvar)
            recon_list.append(self.decode(z))
        return recon_list, mu, logvar
    

def get_vae(L=10):
    return VAE(rep_dim = 128, encoding_dim=128 * 4 * 4, L=L)


def get_dsvdd():
    return DSVDD(rep_dim=128, encoding_dim=128 * 4 * 4)
