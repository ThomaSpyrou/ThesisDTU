# modules
import torch
from skimage.util import random_noise


# TODO 
# how should i store when it comes to real prod env

def gaussian_noise(loader):
    noise_img = []

    for data in loader:
        img, _ = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
        noise_img.append(gauss_img)

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(noise_img)), batch_size=4, shuffle=True, num_workers=4)


def salt_paper_noise(loader):
    noise_img = []

    for data in loader:
        img, _ = data[0], data[1]
        salt_img = torch.tensor(random_noise(img, mode='salt', amount=0.05))
        noise_img.append(salt_img)

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(noise_img)), batch_size=4, shuffle=True)


def speckle_noise(loader):
    noise_img = []

    for data in loader:
        img, _ = data[0], data[1]
        speckle_img = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.05, clip=True))
        noise_img.append(speckle_img)

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(noise_img)), batch_size=4, shuffle=True)
