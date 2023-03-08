import torch
import numpy as np
from scipy.stats import ks_2samp
import sys
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import nannyml as nml
from IPython.display import display
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.utilities import * 
from utils.generate_noise import *


def calculate_ks(distribution_1, distribution_2):
    _, p_value = ks_2samp(distribution_1, distribution_2)

    return p_value


def raise_alert(ks_statistic, threshold=0.2):
    if ks_statistic < threshold:
        print("Data drift detected! KS statistic: {:.4f}".format(ks_statistic))
    else:
        print("KS statistic: {:.4f}".format(ks_statistic))


def main():
    train_loader, test_loader = load_data()
    gaus_loader = gaussian_noise(train_loader)

    train_data_iter = iter(train_loader)
    print(train_data_iter)
    train_images, _ = next(train_data_iter)

    test_data_iter = iter(gaus_loader)
    print(test_data_iter)
    test_images = next(test_data_iter)

    train_flat_images = train_images.view(-1).numpy()
    test_flat_images = test_images.view(-1).numpy()

    ks_statistic = calculate_ks(train_flat_images, test_flat_images)
    
    raise_alert(ks_statistic)


def nanny_documentation_examples():
    reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]

    display(reference_df.head(3))

    estimator = nml.CBPE(
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card'},
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='classification_multiclass',
        metrics=['roc_auc', 'f1'],
        chunk_size=6000,
    )
    estimator.fit(reference_df)

    results = estimator.estimate(analysis_df)
    display(results.filter(period='analysis').to_df())

    metric_fig = results.filter(period='analysis').plot()
    metric_fig.show()

    fig2 = results.plot(kind='performance')
    fig2.show()


nanny_documentation_examples()