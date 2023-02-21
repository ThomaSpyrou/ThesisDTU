# modules 
import os
import argparse
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms


from utilities import *
from models_arch import *

EPOCHS=100


def load_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../data/processed', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../../data/processed', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_resnet():
    device = 'cpu'
    trainloader, testloader = load_data()

    # Load ResNet50  
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Train
    model.to(device)

    train_loss, train_acc, train_softmax = [], [], []
    test_loss, test_acc, test_softmax  = [], [], []


    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        softmax_outputs = []

        for batch_index, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            softmax_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).tolist())

            progress_bar(batch_index, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (running_loss/(batch_index+1), 100.*correct/total, correct, total))

            if batch_index % 100 == 99:
                train_loss.append(running_loss / 100)
                train_acc.append(100. * correct / total)
                train_softmax.append(softmax_outputs)
                print('[Epoch %d, Batch %5d] Train Loss: %.3f, Train Acc: %.3f' % (
                    epoch+1, batch_index+1, running_loss/100, 100.*correct/total))
                running_loss = 0.0
                correct = 0
                total = 0
                softmax_outputs = []



    # Test the model
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        softmax_outputs = []

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            softmax_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).tolist())

        test_loss.append(running_loss / len(testloader))
        test_acc.append(100. * correct / total)
        test_softmax.append(softmax_outputs)
        print('Test Loss: %.3f, Test Acc: %.3f' % (running_loss/len(testloader), 100.*correct/total))


def train_vit():
        
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--bs', default='512')
    parser.add_argument('--size', default="32")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

    args = parser.parse_args()

    size = int(args.size)
    use_amp = bool(~args.noamp)

    device = 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader = load_data()  

    ##### Training
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return train_loss/(batch_idx+1)


    ##### Validation
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        return test_loss, acc

    # ViT for cifar10
    net = ViT(image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)  
        
    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    list_loss = []
    list_acc = []

        
    for epoch in range(start_epoch, EPOCHS):
        trainloss = train(epoch)
        val_loss, acc = test(epoch)
        
        scheduler.step(epoch-1) # step cosine scheduling
        
        list_loss.append(val_loss)
        list_acc.append(acc)

if __name__ == '__main__':
    # train_resnet()
    train_vit()
