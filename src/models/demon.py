# modules
import sys
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.models as models
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler

sys.path.append('../')
from utils.utilities import *
from src.models.demon_helper import *


def get_data():
    train_dataset, test_dataset = load_dataset()

    # for now combine the dataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    return combined_dataset, combined_loader


def target_model_run(data_loader, device, model, n_rounds):
    model.to(device)
    model.eval()

    _, data_loader = get_data()

    acc_list = []

    with torch.no_grad():
        correct = 0
        for images, labels in data_loader:
            # make predictions
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)

            # return only when ask the expert
            correct += (predict == labels).sum().item()
            estimated_acc = correct / labels.size(0)

            acc_list.append(estimated_acc)
    
    avg_estimated_acc = sum(acc_list) / len(acc_list)
            
    return avg_estimated_acc


def buffer():
    """
    this function will batch the data into n_rounds 
    """
    pass


def main():
    # always will start with ask period this will be the reference dataset
    ask_expert = False
    # ask for 500 imgs as a ref dataset
    ask_period = 500
    valid_period = True
    n_rounds = 1000

    ref_batch = []
    curr_batch = []
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target model
    target_model = models.resnet50(pretrained=True)
    num_ftrs_target = target_model.fc.in_features
    target_model.fc = nn.Linear(num_ftrs_target, 10)

    dataset, data_loader = get_data()

    if valid_period:
        # ask expert for labels when starting the system so this data will be used as ref
        # this will run only once 
        
        subset_index = range(n_rounds)
        dataset_subset = data_utils.Subset(dataset, subset_index)
        data_loader_subset = DataLoader(dataset_subset, batch_size=32)

        estimated_acc_ref = target_model_run(data_loader_subset, device, target_model, n_rounds)

        ref_batch = get_features(data_loader_subset, model, device)

        print("Estimated accuracy of the model: ", estimated_acc_ref)

        valid_period = False

    rest_subset_index = range(n_rounds, len(dataset))
    rest_dataset = data_utils.Subset(dataset, rest_subset_index)

    # iterate the data
    num_batches = int(len(rest_dataset) / n_rounds)

    for index in range(num_batches):
        start_index = index * n_rounds
        end_index = (index + 1) * n_rounds
        
        # spliting data into chunks of 500 each time
        sampler = SubsetRandomSampler(range(start_index, end_index))
        loader = DataLoader(rest_dataset, batch_size=32, sampler=sampler)

        if ask_expert:
            # comapre with the latest but also with the ref
            pass
        else:
            for image, _ in loader:
                # running ks test
                curr_batch = get_features(loader, model, device)
                ks_static, p_value = ks_test_f_score(ref_batch, curr_batch)

                # run anomaly detector
                
                print(ks_static, p_value)

    print('done')

main()