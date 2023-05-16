import torch
import torchvision.models as models
import sys
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)
import multiprocessing
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_cifar10

sys.path.append("../")
from utils.utilities import *
multiprocessing.freeze_support()


def resnet():
    model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    train, test = load_data()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test:
            print(1)
            # make predictions
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)

            # return only when ask the expert
            correct += (predict == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print('Accuracy of the ResNet model on the CIFAR-10 test images: %d %%' % accuracy)


def isolation_forest():
    # Load CIFAR10 dataset
    cifar10 = load_cifar10()
    X = cifar10['data']  # Shape: (50000, 3072)
    y = cifar10['target']  # Shape: (50000,)

    # Choose one class as normal and all the others as anomalies
    normal_class = 0  # Choose the first class as normal
    anomaly_classes = np.arange(1, 10)  # Choose the rest of the classes as anomalies
    y_norm = np.where(y == normal_class, 1, -1)  # Set normal class to 1 and all others to -1

    # Train Isolation Forest on the normal class
    clf = IsolationForest(n_estimators=100, contamination='auto')
    clf.fit(X[y_norm == 1])

    # Predict anomaly score on the entire dataset
    anomaly_score = clf.score_samples(X)

    # Print the anomaly score of the first 10 samples
    print(anomaly_score[:10])


isolation_forest()