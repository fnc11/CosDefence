import os
from pathlib import Path
import shutil
from numpy.random import default_rng
from collections import defaultdict
import json

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_poisoned_labels(orig_labels, label_flip_scheme):
    poisoned_labels = orig_labels.detach().clone()
    poisoned_labels[poisoned_labels == label_flip_scheme[0]] = label_flip_scheme[1]
    return poisoned_labels


def get_dataset(dataset_name):
    global base_path
    data_folder = base_path + f'/data/{dataset_name}/raw_data/'
    Path(data_folder).mkdir(parents=True, exist_ok=True)

    if dataset_name == "mnist":
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Download and load the training data
        trainset = datasets.MNIST(data_folder, download=True, train=True, transform=data_transforms)
    elif dataset_name == "fmnist":
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Download and load the training data
        trainset = datasets.FashionMNIST(data_folder, download=True, train=True, transform=data_transforms)
    else:
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # Download and load the training data
        trainset = datasets.CIFAR10(data_folder, download=True, train=True, transform=data_transforms)

    return trainset


def create_client_data(dataset_name='cifar10', class_ratio=10):
    class_ids = defaultdict(list)
    trainset = get_dataset(dataset_name)
    for i in range(len(trainset)):
        class_ids[trainset[i][1]].append(i)

    client_img_tensors = [[] for i in range(100)]
    client_lbl_tensors = [[] for i in range(100)]

    if dataset_name == "mnist":
        print(f"Class ratio used: {class_ratio}")
        # in mnist maximum number of samples each class at least have is 5421
        if class_ratio == 10:
            minor_share = 28  # so we will put 28 images from each minority class => 28*9*10 = 2520 
        elif class_ratio == 4:
            minor_share = 41
        else:
            minor_share = 54
    elif dataset_name == "fmnist":
        # in fashion mnist all classes have 6000 examples each
        if class_ratio == 10:
            minor_share = 31
        elif class_ratio == 4:
            minor_share = 46
        else:
            minor_share = 60
    else:
        # in cifar10 all classes have 5000 examples each
        if class_ratio == 10:
            minor_share = 26
        elif class_ratio == 4:
            minor_share = 38
        else:
            minor_share = 50

    major_share = class_ratio * minor_share  # to maintain ration 10:1

    # Here we are assuming first 10 clients will have class 0 as majority and next 10 will have class 1 as majority
    # and so on.
    # We define two data loader for each class, their job is to place major and minor share of images from that class
    # for 100 clients.
    ## the below 10 is to divide whole dataset into two sets major share set and minor share set
    boundary = major_share*10 # before this boundary index we pick data for major shares

    # print("preparing class\n")
    # print(class_ratio)
    # print(minor_share)
    # print(major_share)
    # print(boundary)
    for i in range(10):
        major_loader = torch.utils.data.DataLoader(trainset, batch_size=major_share,
                                                   sampler=torch.utils.data.SubsetRandomSampler(class_ids[i][:boundary]))
        major_iter = iter(major_loader)

        minor_loader = torch.utils.data.DataLoader(trainset, batch_size=minor_share,
                                                   sampler=torch.utils.data.SubsetRandomSampler(class_ids[i][boundary:]))
        minor_iter = iter(minor_loader)

        for j in range(100):
            if j // 10 == i:
                # put major_share
                data = next(major_iter)
                client_img_tensors[j].extend(data[0])
                client_lbl_tensors[j].extend(data[1])
            else:
                # put minor share
                data = next(minor_iter)
                client_img_tensors[j].extend(data[0])
                client_lbl_tensors[j].extend(data[1])

    # Converting list of tensor images into 1 big tensor of all images and same for labels
    client_data = []
    for i in range(100):
        img_data = torch.stack(client_img_tensors[i])
        lbl_data = torch.stack(client_lbl_tensors[i])
        client_data.append((img_data, lbl_data))  # If we want to save as tuple or tensor we can decide

    poison_params = [0, 10, 20, 40]  # no of clients poisoned out of 100
    label_flips = [(2, 9)]
    total_clients = 100
    seed = 42
    rng = default_rng(seed)
    global base_path
    root_save_folder = os.path.join(base_path, f'data/{dataset_name}/fed_data/')
    # in case folder were present already, cleans the folder inside so that if we accidentally
    # ran ccds flag twice we don't poison data more than necessary
    shutil.rmtree(root_save_folder, ignore_errors=True)
    
    Path(root_save_folder).mkdir(parents=True, exist_ok=True)

    print(root_save_folder)
    for poison_param in poison_params:
        for k in range(len(label_flips)):
            save_folder = os.path.join(root_save_folder,f'label_flip{k}/', f'poisoned_{poison_param}CLs/')
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            print(save_folder)
            clients_selected = rng.choice(total_clients, size=poison_param, replace=False)
            print(clients_selected)
            
            # saving these clients which are selected as being poisonous
            pinfo_data = {}
            pinfo_data['poisoned_clients'] = clients_selected.tolist()
            poison_config_file = os.path.join(save_folder ,'poison_config.txt')
            with open(poison_config_file, 'w') as pconfig_file:
                json.dump(pinfo_data, pconfig_file)
            
            for i in range(total_clients):
                img_tensor_file = save_folder + f'client_{i}_img.pt'
                lbl_tensor_file = save_folder + f'client_{i}_lbl.pt'
                torch.save(client_data[i][0], img_tensor_file)
                # Performing label flipping
                poisoned_labels = client_data[i][1]
                if i in clients_selected:
                    poisoned_labels = get_poisoned_labels(client_data[i][1], label_flips[k])
                torch.save(poisoned_labels, lbl_tensor_file)


class ClientDataset(Dataset):
    def __init__(self, img_tensors, lbl_tensors, transform=None):
        self.img_tensors = img_tensors
        self.lbl_tensors = lbl_tensors
        self.transform = transform

    def __len__(self):
        return self.lbl_tensors.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.img_tensors[idx], self.lbl_tensors[idx]


def create_client_data_loaders(client_nums, data_folder, batch_size, random_mode=False):
    data_loaders = []
    for idx in range(client_nums):
        # loading data to tensors
        img_tensor_file = data_folder + f'client_{idx}_img.pt'
        lbl_tensor_file = data_folder + f'client_{idx}_lbl.pt'
        img_tensors = torch.load(img_tensor_file)  # this contains 494 images, currently 76
        lbl_tensors = torch.load(lbl_tensor_file)

        # creating a dataset which can be fed to dataloader
        client_dataset = ClientDataset(img_tensors, lbl_tensors)
        data_loaders.append(DataLoader(client_dataset, batch_size=batch_size, shuffle=random_mode))
    return data_loaders

def get_test_data_loader(dataset_name, batch_size):
    global base_path
    data_folder = base_path + f'/data/{dataset_name}/raw_data/'
    Path(data_folder).mkdir(parents=True, exist_ok=True)

    if dataset_name == "mnist":
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Download and load the training data
        testset = datasets.MNIST(data_folder, download=True, train=False, transform=data_transforms)
    elif dataset_name == "fmnist":
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                    ])
        # Download and load the training data
        testset = datasets.FashionMNIST(data_folder, download=True, train=False, transform=data_transforms)
    else:
        # Define a transform to normalize the data
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # Download and load the training data
        testset = datasets.CIFAR10(data_folder, download=True, train=False, transform=data_transforms)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    return test_loader


if __name__ == '__main__':
    create_client_data()
