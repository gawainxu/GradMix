import torch
from torch.utils.data import Dataset

import numpy as np
import argparse

from  datautil import get_train_datasets, get_test_datasets, get_outlier_datasets
from feature_linear import set_model, train, test


def parse_option():

    parser = argparse.ArgumentParser('argument for dino testing')

    parser.add_argument("--dino_type", type=str, default="dino_vits16",
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50'])
    parser.add_argument('--datasets', type=str, default='FUB',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet",
                                 "imagenet100", "imagenet100_m", 'mnist', "svhn", "cub", "aircraft", "FUB"],
                        help='dataset')
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--test_mode", type=str, default="linear", choices=["linear", "cosine", "knn"])
    parser.add_argument("--linear_input_dim", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--print_freq", type=int, default=1)

    opt = parser.parse_args()

    return opt


def load_dino(opt):

    model = torch.hub.load('facebookresearch/dino:main', opt.dino_type)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    return model


def load_data(opt):

    train_dataset = get_train_datasets(opt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                              shuffle=False, num_workers=1,
                                              pin_memory=True, drop_last=True, persistent_workers=True)

    test_dataset = get_test_datasets(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1,
                                              pin_memory=True, drop_last=True, persistent_workers=True)

    outlier_dataset = get_outlier_datasets(opt)
    outlier_loader = torch.utils.data.DataLoader(outlier_dataset, batch_size=1,
                                              shuffle=False, num_workers=1,
                                              pin_memory=True, drop_last=True, persistent_workers=True)

    return train_loader, test_loader, outlier_loader


def read_dino_features(data_loader, model):

    features = []
    labels = []

    with torch.inference_mode():
        for i, (img, label, _) in enumerate(data_loader):
            print(i)
            img = img.repeat(1, 3, 1, 1)
            img = img.cuda(non_blocking=True) if torch.cuda.is_available() else img
            f = model(img)
            f = f.cpu()
            label = label.cpu()
            features.append(torch.squeeze(f))
            labels.append(torch.squeeze(label))

    return features, labels


def sort_features(mixedFeatures, labels, num_classes):
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])

    for i, (features, l) in enumerate(zip(mixedFeatures, labels)):
        l = l.item()
        features = features.reshape([-1])
        sortedFeatures[l].append(features)

    return sortedFeatures


def linear_probe(train_features, train_labels, test_features, test_labels, opt):

    train_set = features_set(train_features, train_labels)
    test_set = features_set(test_features, test_labels)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=1, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)
    classifier, criterion, optimizer = set_model(opt)
    losses, top1, top5 = train(classifier, criterion, optimizer, train_dataloader, opt)
    test_acc = test(classifier, test_dataloader, opt)
    print("original testing accuracy", test_acc)


class features_set(Dataset):

    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return  self.features[index], self.labels[index]


if __name__ == "__main__":

    opt = parse_option()
    model = load_dino(opt)
    train_loader, test_loader, outlier_loader = load_data(opt)

    train_features, train_labels = read_dino_features(train_loader, model)
    sorted_train_features = sort_features(train_features, train_labels, num_classes=3)

    test_features, test_labels = read_dino_features(test_loader, model)
    outlier_features, _ = read_dino_features(outlier_loader, model)

    linear_probe(train_features, train_labels, test_features, test_labels, opt)





