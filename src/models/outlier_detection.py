import torch
import sys
import torch.nn as nn

from models_arch import *
sys.path.append('../')
from utils.utilities import *


def init_vae():
    train_data, test_data = load_dataset()

    X_train = train_data.data.astype('float32') / 255
    y_train = torch.tensor(train_data.targets, dtype=torch.int64)
    X_test = test_data.data.astype('float32') / 255
    y_test = torch.tensor(test_data.targets, dtype=torch.int64)

    latent_dim=1024

    # initialize outlier detector
    model = OutlierVAE(threshold=.015,
                    score_type='mse',  
                    latent_dim=latent_dim)
    
    return model

