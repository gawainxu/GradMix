import os
import argparse

import torch
import torch.nn as nn

from main_linear import load_model
from networks.resnet_big import SupConResNet
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from losses import SupConLoss
from datautil import get_train_datasets, get_test_datasets


def parse_option():
    parser = argparse.ArgumentParser('argument for gradient analysis')

    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--backbone_model_dir", type=str,
                        default="/home/zhi/projects/comprehensive_OSR/save/SupCon/tinyimgnet_models")
    parser.add_argument("--backbone_model_name", type=str,
                        default="tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented")
    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet",
                                 "imagenet100", "imagenet100_m", 'mnist', "svhn", "cub", "aircraft"], help='dataset')
    parser.add_argument("--trail", type=int, default=5, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--num_classes", type=int, default=200)

    parser.add_argument("--training_supcon", type=str, default="trainging_linear",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])

    opt = parser.parse_args()

    opt.backbone_model_dir = os.path.join(opt.backbone_model_dir, opt.backbone_model_name)
    opt.backbone_model_path = os.path.join(opt.backbone_model_dir, "last.pth")

    return opt


def set_model(opt):

    in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34" or opt.model == "resnet50":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    model = load_model(model, opt.backbone_model_path)
    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)

    return model, criterion1, criterion2


def set_loader(opt):
    # construct data loader

    train_dataset = get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    if opt.datasets != "imagenet100":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True,
                                                   persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                                  drop_last=True,
                                                  persistent_workers=True)
    else:
        train_loader = train_dataset
        test_loader = test_dataset

    return train_loader, test_loader



if __name__ == "__main__":

    opt = argparse()
