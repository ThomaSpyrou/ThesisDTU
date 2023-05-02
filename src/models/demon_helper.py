# modules
import sys
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import ks_2samp
from torchvision import models
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler, Subset

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
    batch_score = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, _ in img_batch:
            images = images.to(device)
            features = model(images)
            embeddings = torch.flatten(features, start_dim=1)

            prob = torch.softmax(features, dim=1)
            score, _ = torch.max(prob, dim=1)

            batch_score.append(score.cpu().numpy().flatten())
            batch_features.append(embeddings.cpu().numpy().T.flatten())

        batch_features = np.concatenate(batch_features)
        batch_score = np.concatenate(batch_score)

    return batch_features, batch_score


def extract_batch_img_features(img_batch):
    """
    return: features/characteristics of imgs in batches brightness, sharpness, contrast
            output lenght should be the same as n_rounds for each list
    """
    batch_brightness, batch_contrast, batch_sharpness = [], [], []

    for images, _ in img_batch:
        for img in images:
            brightness, contrast, sharpness = extract_img_features(img)

            batch_brightness.append(brightness.cpu().numpy().flatten())
            batch_contrast.append(contrast.cpu().numpy().flatten())
            batch_sharpness.append(sharpness.cpu().numpy().flatten())

    batch_brightness = np.concatenate(batch_brightness)
    batch_contrast = np.concatenate(batch_contrast)
    batch_sharpness = np.concatenate(batch_sharpness)

    return batch_brightness, batch_contrast, batch_sharpness


def get_data():
    train_dataset, test_dataset = load_speific_class()

    # for now combine the dataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    return combined_dataset, combined_loader


def buffer_data(n_rounds, index, dataset):
    print("Batching: ", index)
    start_index = index * n_rounds
    end_index = (index + 1) * n_rounds

    sampler = SubsetRandomSampler(range(start_index, end_index))
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    return loader


def detect_annomalies(data, anomaly_detector):
    """
    return the percentage of the anomalies found on the batch
    """
    anomaly_scores = anomaly_detector.predict(data)
    score = (np.count_nonzero(anomaly_scores == -1) / len(anomaly_scores))

    return score
    

# take whatever layer you want from the net
class IntModel(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x
    

def load_speific_class():
    # get specific class 
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    # Transformations
    RC = transforms.RandomCrop(32, padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    GSC = transforms.ColorJitter(brightness=.99, hue=.5)
    TPIL = transforms.ToPILImage()
    RANINV = transforms.RandomInvert()
    AFFTR = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM, RANINV, GSC])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM, RANINV, GSC])

    # Downloading/Louding CIFAR10 data
    # , transform = transform_with_aug)
    trainset = CIFAR10(root='./data', train=True, download=True)
    # , transform = transform_no_aug)
    testset = CIFAR10(root='./data', train=False, download=True)
    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    # Separating trainset/testset data/label
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets

    # Define a function to separate CIFAR classes by class index


    def get_class_i(x, y, i):
        """
        x: trainset.train_data or testset.test_data
        y: trainset.train_labels or testset.test_labels
        i: class label, a number between 0 to 9
        return: x_i
        """
        # Convert to a numpy array
        y = np.array(y)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(y == i)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        x_i = [x[j] for j in pos_i]

        return x_i


    class DatasetMaker(Dataset):
        def __init__(self, datasets, transformFunc=transform_no_aug):
            """
            datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
            """
            self.datasets = datasets
            self.lengths = [len(d) for d in self.datasets]
            self.transformFunc = transformFunc

        def __getitem__(self, i):
            class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
            img = self.datasets[class_label][index_wrt_class]
            img = self.transformFunc(img)
            return img, class_label

        def __len__(self):
            return sum(self.lengths)

        def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
            """
            Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
            """
            # Which class/bin does i fall into?
            accum = np.add.accumulate(bin_sizes)
            if verbose:
                print("accum =", accum)
            bin_index = len(np.argwhere(accum <= absolute_index))
            if verbose:
                print("class_label =", bin_index)
            # Which element of the fallent class/bin does i correspond to?
            index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
            if verbose:
                print("index_wrt_class =", index_wrt_class)

            return bin_index, index_wrt_class

    # ================== Usage ================== #


    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_trainset = \
        DatasetMaker(
            [get_class_i(x_train, y_train, classDict['plane']),
            get_class_i(x_train, y_train, classDict['plane'])],
            transform_with_aug
        )
    cat_dog_testset = \
        DatasetMaker(
            [get_class_i(x_test, y_test, classDict['plane']),
            get_class_i(x_test, y_test, classDict['plane'])],
            transform_no_aug
        )

    kwargs = {'num_workers': 2, 'pin_memory': False}

    # Create datasetLoaders from trainset and testset
    trainsetLoader = DataLoader(
        cat_dog_trainset, batch_size=64, shuffle=True, **kwargs)
    testsetLoader = DataLoader(
        cat_dog_testset, batch_size=64, shuffle=False, **kwargs)
    
    return cat_dog_trainset, cat_dog_testset