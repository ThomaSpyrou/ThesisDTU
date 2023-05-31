import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp
import sys
from alibi_detect.cd.pytorch import preprocess_drift
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timeit import default_timer as timer
import pandas as pd
# import nannyml as nml
from IPython.display import display
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from alibi_detect.cd import MMDDrift
from alibi_detect.models.tensorflow import scale_by_instance
from alibi_detect.utils.fetching import fetch_tf_model
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.datasets import fetch_cifar10c, corruption_types_cifar10c

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
    # an example of nanny ml to check plots and figures that they create
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


def permute_c(x):
    return np.transpose(x.astype(np.float32), (0, 3, 1, 2))


def detect_drift_mmdd():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = y_train.astype('int64').reshape(-1,)
    y_test = y_test.astype('int64').reshape(-1,)

    corruption = ['gaussian_noise', 'motion_blur', 'brightness', 'pixelate']
    X_corr, y_corr = fetch_cifar10c(corruption=corruption, severity=5, return_X_y=True)
    X_corr = X_corr.astype('float32') / 255

    np.random.seed(1)
    n_test = X_test.shape[0]
    idx = np.random.choice(n_test, size=n_test // 2, replace=False)
    idx_h0 = np.delete(np.arange(n_test), idx, axis=0)
    X_ref,y_ref = X_test[idx], y_test[idx]
    X_h0, y_h0 = X_test[idx_h0], y_test[idx_h0]

    n_corr = len(corruption)
    X_c = [X_corr[i * n_test:(i + 1) * n_test] for i in range(n_corr)]

    # set random seed and device
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    X_ref_pt = permute_c(X_ref)
    X_h0_pt = permute_c(X_h0)
    X_c_pt = [permute_c(xc) for xc in X_c]
    print(X_ref_pt.shape, X_h0_pt.shape, X_c_pt[0].shape)

    encoding_dim = 32

    # define encoder
    encoder_net = nn.Sequential(
        nn.Conv2d(3, 64, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(128, 512, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, encoding_dim)
    ).to(device).eval()

    # define preprocessing function
    preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=512)

    # initialise drift detector
    cd = MMDDrift(X_ref_pt, backend='pytorch', p_val=.05,
                preprocess_fn=preprocess_fn, n_permutations=100)
    
    make_predictions(cd, X_h0_pt, X_c_pt, corruption)


def make_predictions(cd, x_h0, x_corr, corruption):
    labels = ['No!', 'Yes!']
    t = timer()
    preds = cd.predict(x_h0)
    dt = timer() - t
    print('No corruption')
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print(f'Time (s) {dt:.3f}')

    if isinstance(x_corr, list):
        for x, c in zip(x_corr, corruption):
            t = timer()
            preds = cd.predict(x)
            dt = timer() - t
            print('')
            print(f'Corruption type: {c}')
            print('Drift? {}'.format(labels[preds['data']['is_drift']]))
            print(f'p-value: {preds["data"]["p_val"]:.3f}')
            print(f'Time (s) {dt:.3f}')

detect_drift_mmdd()