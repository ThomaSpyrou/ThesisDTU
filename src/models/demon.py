# modules
import sys
import torch 
import warnings
import numpy as np
import time
import pandas as pd
import torch.nn as nn
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torchvision.models as models
from datetime import datetime
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")

sys.path.append('../')
from utils.utilities import *
from demon_helper import *


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

    avg_estimated_acc = 100 * correct / total
            
    return avg_estimated_acc


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
    n_rounds = 500

    ref_batch = []
    curr_batch = []
    # get embeddings of img layer could be configured
    # layers that can be configures  ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
    model = IntModel(output_layer = 'layer4')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target model
    target_model = torch.load("chpt/resnet.pth")


    dataset, data_loader = get_data()
    # validation period 
    ref_batch, ref_batch_brightness, ref_batch_contrast, ref_batch_sharpness = \
                validation_period(n_rounds, dataset, device, target_model, model)

    # fit anomaly detector
    df_feature = pd.DataFrame()
    df_feature['brightness'] = ref_batch_brightness
    df_feature['sharpness'] = ref_batch_sharpness
    df_feature['contrast'] = ref_batch_contrast

    features = ['brightness', 'sharpness', 'contrast']

    # ref_batch_xs = np.array(ref_batch_brightness).reshape(-1, 1)
    anomaly_detector = IsolationForest(n_estimators=50, random_state=42, max_samples='auto', \
                                       contamination=float(0.1),max_features=1.0).fit(df_feature[features])

    rest_subset_index = range(n_rounds, len(dataset))
    rest_dataset = data_utils.Subset(dataset, rest_subset_index)

    # iterate the data
    num_batches = int(len(rest_dataset) / n_rounds)
    counter_ks_test= 0
    counter_iforest = 0 
    counter_dsvdd = 0
    dsvdd_thrs = 0.2
    if_thrs = 0.2
    time_list = []

    for index in range(num_batches):
        loader = buffer_data(n_rounds, index, rest_dataset)

        if ask_expert:
            # comapre with the latest but also with the ref
            pass
        else:
            time_start = time.time()

            # running ks test
            curr_batch, _ = get_features(loader, model, device)
            batch_brightness, batch_contrast, batch_sharpness = extract_batch_img_features(loader)

            ks_static, p_value = ks_test_f_score(ref_batch, curr_batch)
            time_after = time.time()
            diff_time = time_after - time_start
            time_list.append(diff_time)
            dsvdd_score  = load_svdd_detector(device, loader)
            print(dsvdd_score)

            # run anomaly detector
            df_feature = pd.DataFrame()
            df_feature['brightness'] = batch_brightness
            df_feature['sharpness'] = batch_sharpness
            df_feature['contrast'] = batch_contrast
            
            features = ['brightness', 'sharpness', 'contrast']

            anomaly_score = detect_annomalies(df_feature[features], anomaly_detector)

        if p_value <= 0.05:
            counter_ks_test += 1

        if dsvdd_score >= dsvdd_thrs:
            counter_dsvdd += 1

        if anomaly_score >= if_thrs:
            counter_iforest += 1

    print("ks:", counter_ks_test, "\n", "dsvdd:", counter_dsvdd, "\n", "iforest:", counter_iforest)
    print("Time:", calculate_average_time(time_list))

main()

