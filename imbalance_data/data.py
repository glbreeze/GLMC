import os
import torchvision
from . import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data
from .transform import get_transform, TwoCropTransform
from .cifar import CIFAR10, CIFAR100
from torchvision import datasets, transforms

data_folder = 'data/' # for greene,  '../dataset' for local

def get_dataset(args):
    transform_train, transform_val = get_transform(args.dataset, args.aug)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbalance(imbalance_rate=args.imbalance_rate, imbalance_type=args.imbalance_type,
                                                       train=True, transform=transform_train, file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbalance(imbalance_rate=args.imbalance_rate, imbalance_type=args.imbalance_type,
                                                      train=False, transform=transform_val, file_path=args.root)
        print("load cifar10")
        return trainset, testset

    elif args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbalance(imbalance_rate=args.imbalance_rate, imbalance_type=args.imbalance_type,
                                                         train=True, transform=transform_train,
                                                         file_path=os.path.join(args.root, 'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbalance(imbalance_rate=args.imbalance_rate, imbalance_type=args.imbalance_type,
                                                        train=False, transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return trainset, testset

    elif args.dataset == 'fmnist':
        trainset = datasets.FashionMNIST("data", download=True, train=True, transform=transform_train)
        testset = datasets.FashionMNIST("data", download=True, train=False, transform=transform_val)
        print("load fmnist")
        return trainset, testset

    elif args.dataset == 'tinyi':  # image_size:64 x 64
        trainset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), transform_val)
        print("load Tiny ImageNet")
        return trainset, testset

    elif args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset

    elif args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset


def get_dataset_balanced(args, aug=None):
    if aug is None:
        transform_train, transform_val = get_transform(args.dataset, args.aug)
    else:
        transform_train, transform_val = get_transform(args.dataset, aug)
    if args.dataset == 'cifar10':
        trainset= CIFAR10('data', train=True, download=True, transform=transform_train, coarse=False)
        testset = CIFAR10('data', train=False, download=True, transform=transform_val, coarse=False)
        print("load cifar10")
        return trainset, testset

    elif args.dataset == 'cifar100':
        trainset= CIFAR100('data', train=True, download=True, transform=transform_train, coarse=args.coarse)
        testset = CIFAR100('data', train=False, download=True, transform=transform_val, coarse=args.coarse)
        print("load cifar100")
        return trainset, testset

    elif args.dataset == 'stl10':
        trainset= datasets.STL10('data', split='train', download=True, transform=transform_train)
        testset = datasets.STL10('data', split='test', download=True, transform=transform_val)
        print("load stl10")
        return trainset, testset

    elif args.dataset == 'fmnist':
        trainset = datasets.FashionMNIST("data", download=True, train=True, transform=transform_train)
        testset = datasets.FashionMNIST("data", download=True, train=False, transform=transform_val)
        print("load fmnist")
        return trainset, testset

    elif args.dataset == 'tinyi':  # image_size:64 x 64
        trainset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), transform_val)
        print("load Tiny ImageNet")
        return trainset, testset
