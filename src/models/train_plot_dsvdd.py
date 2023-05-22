import torch
from torch import optim
import sys
import os
import numpy as np
from sklearn import metrics
import argparse
import time
import matplotlib.pyplot as plt


sys.path.append('../')
from utils.utilities import *

def init_center_c(model, trainloader, device):
    eps = 0.1
    c = torch.zeros(model.rep_dim, device=device)
    model.eval()
    n_samples = 0
    with torch.no_grad():
        for (inputs, _) in trainloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def train(epoch, model, center, trainloader, optimizer, scheduler, logger, device):
    train_loss = 0.
    model.train()  # train mode

    scheduler.step()  # update optimizer lr

    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        dist = torch.sum((outputs - center) ** 2, dim=1)
        loss = torch.mean(dist)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f ' % (
            epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1)), end='\r')
    print('')
    logger.write('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f \n' % (
        epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


def test(model, center, testloader, device):
    test_loss = 0.
    scores_list = []
    targets_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            dist = torch.sum((outputs - center) ** 2, dim=1)
            scores_list.append(dist.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            print('  Test... Iter: %4d/%4d ' % (batch_idx + 1, len(testloader)), end='\r')
    print('')

    test_loss = test_loss / (batch_idx + 1)
    scores = np.concatenate(scores_list)
    targets = np.concatenate(targets_list)
    auroc = metrics.roc_auc_score(targets, scores)

    precision, recall, _ = metrics.precision_recall_curve(targets, scores)
    aupr = metrics.auc(recall, precision)

    # Calculate accuracy and F1 score
    predictions = scores > 0.5  # Adjust threshold as needed
    accuracy = metrics.accuracy_score(targets, predictions)
    f1_score = metrics.f1_score(targets, predictions)

    return auroc, aupr, test_loss, accuracy, f1_score


def plot_metrics(train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'b', label='Train Loss')
    plt.plot(epochs, test_loss_list, 'r', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, 'b', label='Train Accuracy')
    plt.plot(epochs, test_accuracy_list, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("traindsvdd.png")


def main(args):
    logger, result_dir, dir_name = config_backup_get_log(args, __file__)
    device = get_device()
    set_seed(args.seed, device)

    trainloader = get_trainloader(args.data, args.dataroot, args.target, args.bstrain, args.nworkers)
    testloader = get_testloader(args.data, args.dataroot, args.target, args.bstest, args.nworkers)

    import models
    model = models.DSVDD().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    chpt_name = 'DSVDD_%s_target%s_seed%s.pth' % (args.data, str(args.target), str(args.seed))
    chpt_name = os.path.join("./chpt", chpt_name)

    center = init_center_c(model, trainloader, device)
    print('==> Start training ..')

    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    start = time.time()
    for epoch in range(args.maxepoch):
        train_loss = train(epoch, model, center, trainloader, optimizer, scheduler, logger, device)
        train_loss_list.append(train_loss)

        auroc, aupr, test_loss, accuracy, f1_score = test(model, center, testloader, device)
        test_loss_list.append(test_loss)
        train_accuracy_list.append(accuracy)
        test_accuracy_list.append(auroc)

        print('Epoch: %4d Train Loss: %.4f Test Loss: %.4f Train Accuracy: %.4f Test Accuracy: %.4f' %
              (epoch, train_loss, test_loss, accuracy, auroc))
        logger.write('Epoch: %4d Train Loss: %.4f Test Loss: %.4f Train Accuracy: %.4f Test Accuracy: %.4f \n' %
                     (epoch, train_loss, test_loss, accuracy, auroc))

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Final Test Results:')
    print('AUROC: %.6f AUPR: %.6f' % (auroc, aupr))
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    logger.write('Final Test Results:\n')
    logger.write('AUROC: %.6f AUPR: %.6f\n' % (auroc, aupr))
    logger.write("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

    plot_metrics(train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list)


if __name__ == '__main__':
    args = process_args()
    main(args)
