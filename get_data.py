from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import os
import torch

def get_data(dataset_name, type_, img_size, opt):
    train_loader, test_loader = None, None
    print(f"working with: {dataset_name}")
    num_gpus = 1# opt.ngpus_per_node
    workers = 16 // num_gpus
    # print(num_gpus)
    train_sampler = None

    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])

        cifar10_train = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_train)

        if type_ == 'full_batch':
            subset_indices_train = range(5000)
            subset_dataset_train = Subset(cifar10_train, subset_indices_train)

            subset_indices_test = range(1000)
            subset_dataset_test = Subset(cifar10_test, subset_indices_test)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(subset_dataset_train)

            train_loader = DataLoader(subset_dataset_train, batch_size=5000//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(subset_dataset_test, batch_size=1000//num_gpus, shuffle=False, num_workers=workers)
        else:
            train_loader = DataLoader(cifar10_train, batch_size=1500//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(cifar10_test, batch_size=1500//num_gpus, shuffle=False, num_workers=workers)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        cifar100_train = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
        cifar100_test = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_train)

        if type_ == 'full_batch':
            subset_indices_train = range(4000)
            subset_dataset_train = Subset(cifar100_train, subset_indices_train)

            subset_indices_test = range(1000)
            subset_dataset_test = Subset(cifar100_test, subset_indices_test)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(subset_dataset_train)

            train_loader = DataLoader(subset_dataset_train, batch_size=4000//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(subset_dataset_test, batch_size=1000//num_gpus, shuffle=False, num_workers=workers)
        else:
            train_loader = DataLoader(cifar100_train, batch_size=1500//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(cifar100_test, batch_size=1500//num_gpus, shuffle=False, num_workers=workers)

    elif dataset_name == "mnist":
        # Define transformations
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if type_ == 'full_batch':
            # dataset_folder = f'./{opt.dataset}/{size}'
            mnist_train = datasets.ImageFolder(root='./data/mini_mnist/train', transform=transform)
            mnist_test = datasets.ImageFolder(root='./data/mini_mnist/val', transform=transform)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(mnist_train)

            train_loader = DataLoader(mnist_train, batch_size=5500//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(mnist_test, batch_size=1000//num_gpus, shuffle=False, num_workers=workers)
        else:
            mnist_train = datasets.MNIST(root='./data/mini_mnist/train', train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST(root='./data/mini_mnist/val', train=False, download=True, transform=transform)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(mnist_train)

            train_loader = DataLoader(mnist_train, batch_size=5000//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(mnist_test, batch_size=5000//num_gpus, shuffle=False, num_workers=workers)

    elif dataset_name == "imagenet100":
        data_dir = "/mnt/SSD_2/Alok/IISc/parikshit/dataset/sampled_imagenet100" 
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create datasets
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        if type_ == "full_batch":
            subset_indices_train = range(5000)
            subset_dataset_train = Subset(train_dataset, subset_indices_train)
            subset_indices_test = range(1000)
            subset_dataset_test = Subset(test_dataset, subset_indices_test)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(subset_dataset_train)

            train_loader = DataLoader(subset_dataset_train, batch_size=5000//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(subset_dataset_test, batch_size=1000//num_gpus, shuffle=False, num_workers=workers)
        else: 
            train_loader = DataLoader(train_dataset, batch_size=128//num_gpus, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers)
            test_loader = DataLoader(test_dataset, batch_size=128//num_gpus, shuffle=False, num_workers=workers)
    
    elif dataset_name == "imagenet1k":
        data_dir = "/mnt/SSD_2/Alok/IISc/parikshit/dataset/imagenet1k/data1k" 
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if type_ == "full_batch":
            raise ValueError("For imagenet1k, only mini_batch is supported currently.")

        # Create datasets
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=workers)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=workers)
    
    else: 
        raise ValueError
    # opt.train_sampler = train_sampler
    return train_loader, test_loader, opt
