import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import sys
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision.models import resnet34, vgg16, densenet121
from torch.optim.lr_scheduler import MultiplicativeLR, StepLR
from torch.optim import Adam

from utils.optimizer import Adamomentum
from utils.train import train_model
from utils.utils import get_result_path, print_log, plot_curve, save_model

pretrained_seed = random.randint(1, 10000)
dataset_name = 'cifar100'
network_arch = 'vgg16'
result_subfolder = '{}'.format(dataset_name)
batch_size = 128
epochs = 200
use_dataparallel = False
num_workers = 4
LrDecay_steps = 60
weight_decay=0
optimizers = ["Adam",'Adamomentum']

def main():
    random.seed(pretrained_seed)
    torch.manual_seed(pretrained_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed_all(pretrained_seed)

    result_path = get_result_path(
                                    dataset_name = dataset_name, 
                                    network_arch= network_arch, 
                                    random_seed = pretrained_seed, 
                                    result_subfolder = result_subfolder
                                    )

    # Init logger
    log_file_name = os.path.join(result_path, "log.txt")
    log = open(log_file_name, "w")
    print_log("Log file: {}".format(log_file_name), log)
    print_log("save path : {}".format(result_path), log)
    print_log('torch version :{}'.format(torch.__version__), log)
    print_log('python version :{}'.format(sys.version), log)
    print_log('Use Cuda :{}'.format(use_cuda), log)
    print_log('Dataset :{}'.format(dataset_name), log)
    print_log('Network arch :{}'.format(network_arch), log)
    # print_log('Image size :{}'.format(img_size), log)
    print_log('Batch size :{}'.format(batch_size), log)
    print_log('Epochs :{}'.format(epochs), log)
    print_log('Random seed :{}'.format(pretrained_seed), log)
    print_log('Num Workers :{}'.format(num_workers), log)
    print_log('Lr Decay Steps size :{}'.format(LrDecay_steps), log)
    print_log('Weight Decay :{}'.format(weight_decay), log)

    # Data
    transform = T.Compose(
        [
            # T.Resize(img_size), 
            T.ToTensor(), 
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]
    )

    if dataset_name == 'cifar10':
        num_classes = 10
        trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    elif dataset_name == 'cifar100':
        num_classes = 100
        trainset = CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
        testset = CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    dataloaders = {'train': trainloader, 'val': testloader}

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    phases = ['train', 'val']
    losses = {f'{phase}': {f'{opt_name}' : [] for opt_name in optimizers} for phase in phases}
    acc = {f'{phase}': {f'{opt_name}' : [] for opt_name in optimizers} for phase in phases}

    for opt_name in optimizers:

        # Model
        if network_arch == 'resnet34':
            model = resnet34(pretrained=False)
            model.fc = nn.Linear(512, num_classes)
        elif network_arch == 'vgg16':
            model = vgg16(pretrained=False)
            model.fc = nn.Linear(4096, num_classes)
        elif network_arch == 'densenet121':
            model = densenet121(pretrained=False)
            model.classifier = nn.Linear(1024, num_classes)

        if use_dataparallel:
            model = nn.DataParallel(model)
        print_log('Model :{}'.format(model), log)
        model = model.to(device)

        print_log('Optimizer : {}'.format(opt_name), log)
        if opt_name=='Adam':
            optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=weight_decay)
        elif opt_name=='Adamomentum':
            optimizer = Adamomentum(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=weight_decay)
        
        scheduler = StepLR(optimizer, step_size=LrDecay_steps, gamma=0.2)
        
        # Train and evaluate
        model, losses, acc = train_model(model, criterion, optimizer, opt_name, 
                                        dataloaders, log, result_path, 
                                        losses, acc,
                                        scheduler=scheduler, epochs=epochs, device=device)

if __name__ == '__main__':
    main()