import torch
import random
import numpy as np
import numpy as np
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# import torch.utils.data.distributed
import pickle
import os

from get_data import get_data
from experiments import plot_combined, plot_step



def parse_option():
    parser = argparse.ArgumentParser('Downloading mini-MNIST data')
    parser.add_argument('--num_layers', nargs='+', type=int, default=[5], help='list of num of layers in model')
    parser.add_argument('--nodes', nargs='+', type=int, default=[1000], help='list of nodes in each layer')
    parser.add_argument('--size', nargs='+', type=int, default=[28], help='list of size of mini-MNIST images')
    parser.add_argument('--exp', nargs='+', type=str, default=['schedulers'], help='which experiment to run')
    parser.add_argument('--lr', type=int, default=0.01, help='nodes in each layer')
    parser.add_argument('--epochs', type=int, nargs='+', default=[100], help='Size of mini-MNIST images')
    parser.add_argument('--type', type=str, default='full_batch', help="'full_batch' or 'mini_batch'")
    parser.add_argument('--dataset', type=str, default='mnist', help=" 'mnist' or 'mini_mnist' or 'cifar10' or 'cifar100' ")
    parser.add_argument('--model_num', type=int, default='0', help="{0: 'linear model', 1: 'vgg9', 2: 'lenet', 3: 'googlenet', 4: 'mobilenet', 5: 'efficientnet', 6: 'vit'}")
    args = parser.parse_args()

    for exp in args.exp:
        assert exp in ['schedulers', 'step']
    if args.model_num:
        if args.model_num == 6:
            assert args.dataset in ['imagenet100']
        if args.model_num == 2:
            args.size = [28]
            assert args.dataset in ['mnist', 'mini_mnist', 'cifar10', 'cifar100']
        else:
            args.size = [32]
            assert args.dataset in ['cifar10', 'cifar100', 'imagenet100', 'imagenet1k']
    #print(f"ALERT --> USING INITIAL LR: {args.lr}")
    return args


def seed_everything(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def main():
    args = parse_option()
    seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(42)

    d = {0: 'linear model', 1: 'vgg9', 2: 'lenet', 3: 'resnet18', 4: 'mobilenet', 5: 'efficientnet', 6: 'vit'}
    print(f"using {d[args.model_num]}")

    if args.model_num == 0:
        print(f"list of layers: {args.num_layers} | list of nodes: {args.nodes} | list of sizes: {args.size}")

    for exp in args.exp:
        for size in args.size:
            torch.cuda.empty_cache()
            print(args.dataset)
            train_loader, test_loader, args = get_data(args.dataset, args.type, size, args)
            for num_layers in args.num_layers:
                for nodes in args.nodes:
                    for epochs in args.epochs:
                        print(f"exp for epochs: {epochs}")
                        if args.model_num == 0:
                            print(f"Running for: {num_layers} layers with {nodes} nodes each on img size {size} || {args.type} || {args.dataset} || {len(train_loader.dataset)} || {len(test_loader.dataset)}")
                        else:
                            print(f"Running {args.type} || {args.dataset} || {len(train_loader.dataset)} || {len(test_loader.dataset)}")
                        if exp == 'schedulers':
                            plot_combined(train_loader, test_loader, num_layers, nodes, args.lr, epochs, args, size, model_num=args.model_num)
                        elif exp == "step":
                            plot_step(train_loader, test_loader, num_layers, nodes, args.lr, epochs, args, size, model_num=args.model_num)
                        else:
                            raise ValueError
    


if __name__ == '__main__':
    main()





