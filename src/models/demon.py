# modules
import sys
import torch 
import warnings
import numpy as np
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
    ref_score = []
    curr_batch = []
    # get embeddings of img layer could be configured
    model = IntModel(output_layer = 'layer4')

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10)
    # get the feature from the second last layer
    #model = torch.nn.Sequential(*list(model.children())[:-2]) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # target model
    target_model = torch.load("chpt/resnet.pth")
    # target_model = models.resnet18(pretrained=True)
    # num_ftrs_target = target_model.fc.in_features
    # target_model.fc = nn.Linear(num_ftrs_target, 10)

    dataset, data_loader = get_data()
    # validation period 
    ref_batch, ref_batch_brightness, ref_batch_contrast, ref_batch_sharpness = \
                validation_period(n_rounds, dataset, device, target_model, model)
    
    # import pdb
    # pdb.set_trace()

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
    counter_ask = 0
    for index in range(num_batches):
        loader = buffer_data(n_rounds, index, rest_dataset)

        if ask_expert:
            # comapre with the latest but also with the ref
            pass
        else:
            time = datetime.now()

            # running ks test
            curr_batch, _ = get_features(loader, model, device)
            batch_brightness, batch_contrast, batch_sharpness = extract_batch_img_features(loader)
            print("len: ", str(len(curr_batch)), str(len(ref_batch)))

            ks_static, p_value = ks_test_f_score(ref_batch, curr_batch)
            if p_value <= 0.05:
                counter_ask += 1

            print(ks_static, p_value)
            bins = np.logspace(np.log10(1e-6), np.log10(1), 100)
            r_hist = (1e-6, 1)

            

            # # Plot histogram on log scale
            # plt.hist(ref_batch, bins=bins, range=r_hist, density = True, alpha=0.5, rwidth = 0.6, label='Reference')
            # plt.hist(curr_batch, bins=bins, range=r_hist, alpha=0.5, rwidth = 0.6, density = True, label='Current')

            # plt.xscale('log')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.legend()
            # plt.savefig('chpt/' + 'line' + str(p_value) + '.png')
            # plt.close()

            # import pdb
            # pdb.set_trace()

            # run anomaly detector
            df_feature = pd.DataFrame()
            df_feature['brightness'] = batch_brightness
            df_feature['sharpness'] = batch_sharpness
            df_feature['contrast'] = batch_contrast

            plt.plot(np.sort(ref_batch_brightness), np.linspace(0, 1, len(ref_batch_brightness)), label='Ref')
            plt.plot(np.sort(batch_brightness), np.linspace(0, 1, len(batch_brightness)), label='Curr')
            plt.ylabel("P value")
            plt.xlabel("Brightness")
            plt.legend()
            
            features = ['brightness', 'sharpness', 'contrast']

            anomaly_score = detect_annomalies(df_feature[features], anomaly_detector)
            print(anomaly_score)
            # plt.savefig('chpt/' + 'line' + str(anomaly_score) + '.png')
            # plt.close()


    print('done', counter_ask)

main()