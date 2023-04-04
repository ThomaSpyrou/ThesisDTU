import os
import sys
import time
import torch
from datetime import datetime
import glob
import shutil
from pprint import pprint
import json
import torchvision
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import torch.nn.functional as F


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def load_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='../../data/processed', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../../data/processed', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def load_dataset():
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

    return trainset, testset


def elbo(x, x_recon, mu, log_var):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss - kld_loss


def extract_img_features(img):
    brightness = transforms.functional.rgb_to_grayscale(img).mean()
    contrast = transforms.functional.adjust_contrast(img, 2).mean() - transforms.functional.adjust_contrast(img, 0.5).mean()
    sharpness = transforms.functional.adjust_sharpness(img, 2).mean() - transforms.functional.adjust_sharpness(img, 0.5).mean()

    return brightness, contrast, sharpness


class CIFAR10(torchdata.Dataset):
    def __init__(self, dataroot, target, imagesize, train):
        self.target = target
        self.train = train

        if imagesize is None:
            transform = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(imagesize),
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        self.raw_dataset = torchvision.datasets.CIFAR10(root=dataroot, train = train, transform=transform, download = True )
        targets = np.asarray(self.raw_dataset.targets)
        if self.train:
            self.idxs = np.where(targets == target)[0]
        else: # test
            self.idxs = np.arange(targets.size)

    def __len__(self):
        return self.idxs.size

    def __getitem__(self, i):
        idx = self.idxs[i]
        data = self.raw_dataset[idx][0]
        min, max = data.min(), data.max()
        data = (data-min)/(max-min)

        if self.train: # when training, label is not considered
            label = torch.LongTensor([1])
        else: # if label is one, it means it is anomalous
            label = torch.LongTensor([self.raw_dataset.targets[idx]!=self.target])
        return data,label
    

def get_trainloader(datatype, dataroot, target, batchsize, nworkers, imagesize = None):
    traindataset = CIFAR10(dataroot, target, imagesize = imagesize, train=True)
    trainloader = torchdata.DataLoader(traindataset, batch_size = batchsize, shuffle = True, num_workers = nworkers)
    
    return trainloader


def get_testloader(datatype, dataroot, target, batchsize, nworkers, imagesize = None):
    testdataset = CIFAR10(dataroot, target, imagesize = imagesize, train=False)
    testloader = torchdata.DataLoader(testdataset, batch_size = batchsize, shuffle = False, num_workers = nworkers)
    
    return testloader



def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./', help='data path')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--data', type=str, default='fmnist', help='data type, mnist|cifar10|fmnist')
    parser.add_argument('--target', type=int, default=0, help='target integer in mnist dataset')
    parser.add_argument('--bstrain', type=int, default=200, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=200, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=8, help='the number of workers used in DataLoader')
    parser.add_argument('--dropoutp', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--mcdropoutT', type=int, default=20, help='the number of mc samplings')
    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')

    args = parser.parse_args()
    args.lr = 1e-3
    args.milestones = [75]
    args.maxepoch = 100

    return args


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    return device 


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def config_backup_get_log(args, filename):
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    if not os.path.isdir('./chpt'):
        os.mkdir('./chpt')

    # set result dir
    current_time = str(datetime.now())
    dir_name = '%s_%s'%(current_time, args.suffix)
    result_dir = 'results/%s'%dir_name

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        os.mkdir(result_dir+'/codes')

    # deploy codes
    files = glob.iglob('*.py')
    model_files = glob.iglob('./models/*.py')

    for file in files:
        shutil.copy2(file, result_dir+'/codes')
    for model_file in model_files:
        shutil.copy2(model_file, result_dir+'/codes/models')


    # printout information
    print("Export directory:", result_dir)
    print("Arguments:")
    pprint(vars(args))

    logger = open(result_dir+'/%s.txt'%dir_name,'w')
    logger.write("%s \n"%(filename))
    logger.write("Export directory: %s\n"%result_dir)
    logger.write("Arguments:\n")
    logger.write(json.dumps(vars(args)))
    logger.write("\n")

    return logger, result_dir, dir_name
