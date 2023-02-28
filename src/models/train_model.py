# modules 
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

sys.path.append('../')
from utils.utilities import *
from models_arch import *


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--net', default='')
parser.add_argument('--num_class', default='10')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)

args = parser.parse_args()
EPOCHS = int(args.n_epochs)

# ResNet pre-trained model
def train_resnet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader = load_data()

    # Load ResNet50  
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=5e-4)

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

        train_loss.append(running_loss / 100)
        train_acc.append(100. * correct / total)
        train_softmax.append(softmax_outputs)

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
    size = int(args.size)
    use_amp = bool(~args.noamp)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training on :", device)
    trainloader, testloader = load_data()  

    # ViT for cifar10
    model = ViT(image_size = size,
        patch_size = args.patch,
        num_classes = int(args.num_class),
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)  
        
    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    train_loss, train_acc, train_softmax = [], [], []
    test_loss, test_acc, test_softmax  = [], [], []

    # Train
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        softmax_outputs = []

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).tolist())

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (running_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        train_loss.append(running_loss / 100)
        train_acc.append(100. * correct / total)
        train_softmax.append(softmax_outputs)

        scheduler.step(epoch-1)

    # Validation
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    softmax_outputs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).tolist())
    
        test_loss.append(running_loss / len(testloader))
        test_acc.append(100. * correct / total)
        test_softmax.append(softmax_outputs)
        print('Test Loss: %.3f, Test Acc: %.3f' % (running_loss/len(testloader), 100.*correct/total))
    

if __name__ == '__main__':
    if args.net == "resnet":
        train_resnet()
    elif args.net == "vit":
        train_vit()
    else:
        train_resnet()
        train_vit()
