# modules
import numpy as np
import pandas as pd
import sys
import torch 
import torch.nn as nn
from scipy.stats import ks_2samp
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append('../')
from utils.utilities import *
from models_arch import *
from demon_helper import *


def calculate_ks(batch_curr, batch_prev):
    # Flatten the images to 1D arrays
    batch_curr = batch_curr.view(batch_curr.shape[0], -1)
    batch_prev = batch_prev.view(batch_prev.shape[0], -1)

    statistic, pvalue = ks_2samp(batch_curr.numpy(), batch_prev.numpy())


def get_data():
    train_dataset, test_dataset = load_dataset()

    # for now combine the dataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    return combined_loader



def main():
    ask = True
    ask_period = 500
    n_rounds = 500 # n_rounds and ask_period should be (for how many images should the expert be asked)
    previous_batch = []
    curr_batch = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_data()

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in data_loader:
            # make predictions
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)

            if ask is True:
                # ask the expert
                correct += (predict == labels).sum().item()
                estimated_acc = correct / n_rounds
                # print('Accuracy on the test set: {:.2%}'.format(correct / 500))

                # for how many pics they will be answered
                ask_period = ask_period - 1
                if ask_period == 0: 
                    ask = False
                    ask_period = 500
                    break
            

            # total += labels.size(0)
            # correct += (predict == labels).sum().item()



if __name__ == '__main__':
    main()
