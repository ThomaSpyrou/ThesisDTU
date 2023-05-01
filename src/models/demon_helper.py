# modules
import sys
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import ks_2samp
from torchvision import models
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler, Subset

sys.path.append('../')
from utils.utilities import *


def ks_test_f_score(init_features_batch, curr_features_batch):
    """
    return: p_value and ks statistics for feature level batches
    """
    ks_static, p_value = ks_2samp(init_features_batch, curr_features_batch)

    return ks_static, p_value


def get_features(img_batch, model, device):
    """
    return: batch of extracted img features 
    """
    batch_features = []
    batch_score = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, _ in img_batch:
            images = images.to(device)
            features = model(images)
            embeddings = torch.flatten(features, start_dim=1)

            prob = torch.softmax(features, dim=1)
            score, _ = torch.max(prob, dim=1)

            batch_score.append(score.cpu().numpy().flatten())
            batch_features.append(embeddings.cpu().numpy().flatten())

        batch_features = np.concatenate(batch_features)
        batch_score = np.concatenate(batch_score)

    return batch_features, batch_score


def extract_batch_img_features(img_batch):
    """
    return: features/characteristics of imgs in batches brightness, sharpness, contrast
            output lenght should be the same as n_rounds for each list
    """
    batch_brightness, batch_contrast, batch_sharpness = [], [], []

    for images, _ in img_batch:
        for img in images:
            brightness, contrast, sharpness = extract_img_features(img)

            batch_brightness.append(brightness.cpu().numpy().flatten())
            batch_contrast.append(contrast.cpu().numpy().flatten())
            batch_sharpness.append(sharpness.cpu().numpy().flatten())

    batch_brightness = np.concatenate(batch_brightness)
    batch_contrast = np.concatenate(batch_contrast)
    batch_sharpness = np.concatenate(batch_sharpness)

    return batch_brightness, batch_contrast, batch_sharpness


def get_data():
    train_dataset, test_dataset = load_dataset()

    # for now combine the dataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    return combined_dataset, combined_loader


def buffer_data(n_rounds, index, dataset):
    print("Batching: ", index)
    start_index = index * n_rounds
    end_index = (index + 1) * n_rounds

    sampler = SubsetRandomSampler(range(start_index, end_index))
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    return loader


def detect_annomalies(data, anomaly_detector):
    """
    return the percentage of the anomalies found on the batch
    """
    anomaly_scores = anomaly_detector.predict(data)

    score = (np.count_nonzero(anomaly_scores == -1) / len(anomaly_scores))

    return score
    


# take whatever layer you want from the net
class IntModel(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x
