# modules
import sys
import torch 
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.models as models
from sklearn.ensemble import IsolationForest
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
warnings.filterwarnings("ignore")

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

    with torch.no_grad():
        correct = 0
        total = 0 
        for images, labels in data_loader:
            # make predictions
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
        print(total, correct)
    avg_estimated_acc = 100 * correct / total
            
    return avg_estimated_acc


def buffer():
    """
    this function will batch the data into n_rounds 
    """
    pass


def validation_period(n_rounds, dataset, device, target_model, model):
    subset_index = range(n_rounds)
    dataset_subset = data_utils.Subset(dataset, subset_index)
    data_loader_subset = DataLoader(dataset_subset, batch_size=32)

    estimated_acc_ref = target_model_run(data_loader_subset, device, target_model, n_rounds)
    ref_batch, _ = get_features(data_loader_subset, model, device)
    batch_brightness, batch_contrast, batch_sharpness = extract_batch_img_features(data_loader_subset)

    print("Estimated accuracy of the model: ", estimated_acc_ref)

    return ref_batch, batch_brightness, batch_contrast, batch_sharpness


def main():
    # always will start with ask period this will be the reference dataset
    ask_expert = False
    # size of chunks
    n_rounds = 1000

    ref_batch = []
    ref_score = []
    curr_batch = []
    model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # get the feature from the second last layer
    #model = torch.nn.Sequential(*list(model.children())[:-2]) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target model
    target_model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    # num_ftrs_target = target_model.fc.in_features
    # target_model.fc = nn.Linear(num_ftrs_target, 10)

    dataset, data_loader = get_data()

    # validation period 
    ref_batch, ref_batch_brightness, ref_batch_contrast, ref_batch_sharpness = \
                validation_period(n_rounds, dataset, device, target_model, model)

    # fit anomaly detector
    ref_batch_brightness = np.array(ref_batch_brightness).reshape(-1, 1)
    anomaly_detector = IsolationForest(random_state=42).fit(ref_batch_brightness)

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
            # running ks test
            curr_batch, _ = get_features(loader, model, device)
            batch_brightness, batch_contrast, batch_sharpness = extract_batch_img_features(loader)

            ks_static, p_value = ks_test_f_score(ref_batch, curr_batch)

            print(ks_static, p_value)

            # run anomaly detector
            curr_curr_batch_x = np.array(batch_brightness).reshape(-1, 1)
            anomaly_scores = anomaly_detector.score_samples(curr_curr_batch_x)

            threshold = -0.5  #??
            outlier = np.where(anomaly_scores < threshold)[0]
            
            if len(outlier) == 0:
                print("outlier")
            else:
                print("skip")

    print('done')

main()