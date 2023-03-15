# modules 
import argparse
import sys
import torch
import pdb
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')
from utils.utilities import *
from monitor_acc_model import *


# parsers
parser = argparse.ArgumentParser(description='Estimate Accuracy')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--num_class', default='10')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--dimhead', default="512", type=int)

args = parser.parse_args()
EPOCHS = int(args.n_epochs)


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = IamWatchingU()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(EPOCHS):
        model.train()