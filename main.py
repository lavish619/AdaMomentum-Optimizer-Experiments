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

from utils.optimizer import Adamomentum
from utils.train import train_model
from utils.utils import get_result_path, print_log, plot_curve, save_model

pretrained_seed = random.randint(1, 10000)
dataset_name = 'cifar10'
network_arch = 'resnet34'
result_subfolder = '{}'.format(dataset_name)
img_size = 128
batch_size = 32
epochs = 10
use_dataparallel = False

def main():
    random.seed(pretrained_seed)
    torch.manual_seed(pretrained_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed_all(pretrained_seed)

    result_path = get_result_path(dataset_name = dataset_name, 
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
    print_log('Image size :{}'.format(img_size), log)
    print_log('Batch size :{}'.format(batch_size), log)
    print_log('Epochs :{}'.format(epochs), log)
    print_log('Random seed :{}'.format(pretrained_seed), log)

    
    # Data
    transform = T.Compose(
        [T.Resize(img_size), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
                                            shuffle=True, num_workers=2)

    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    dataloaders = {'train': trainloader, 'val': testloader}

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
    print_log('Model :{}'.format(network_arch), log)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adamomentum(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    # Train and evaluate
    model, losses, acc = train_model(model, criterion, optimizer, dataloaders, log, epochs=10, device=device)

    # Save model
    save_model(model, result_path)

    # Plot
    plot_curve(losses, acc, result_path)


if __name__ == '__main__':
    main()