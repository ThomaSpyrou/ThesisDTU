# modules
import sys
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import ks_2samp

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
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, _ in img_batch:
            images = images.to(device)
            features = model(images)

            batch_features.append(features.cpu().numpy().flatten())
        
        batch_features = np.concatenate(batch_features)

    return batch_features
