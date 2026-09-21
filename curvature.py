import os
import pickle
import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from datautil import get_train_datasets, get_test_datasets
from networks.resnet_big import SupConResNet, LinearClassifier, ResNet50, MoCoResNet
from networks.maskcon import MaskCon
from networks.simCNN import simCNN_contrastive
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model_path', type=str, default="")

    opt = parser.parse_args()
    return opt


def set_loader(opt):
    # construct data loader

    if opt.upsample is True and opt.last_feature_path is not None:

        last_features_list = []
        last_feature_labels_list = []

        with open(opt.last_feature_path, "rb") as f:
            last_features, _, _, last_feature_labels = pickle.load(f)
            last_features_list.append(last_features)
            last_feature_labels_list.append(last_feature_labels)

        last_model = load_model(opt)
        train_dataset = get_train_datasets(opt, last_features_list=last_features_list,
                                           last_feature_labels_list=last_feature_labels_list, last_model=last_model)
    else:
        train_dataset = get_train_datasets(opt)
        test_dataset = get_test_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True,
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=opt.num_workers, pin_memory=True,
                                              sampler=train_sampler, drop_last=True, persistent_workers=True)

    return train_loader, test_loader

def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3

    if opt.method == "MoCo":
        model = MoCoResNet(opt, name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None
    elif opt.method == "SimCLR_CE":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = torch.nn.CrossEntropyLoss()
        linear = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    elif opt.method == "MaskCon":
        model = MaskCon(arch="resnet18", T1=opt.method_T1, T2=opt.method_T2)
        criterion1 = None
        criterion2 = None
        linear = None
    elif opt.method == "SimCLR" or opt.method == "SupCon":
        if opt.model in ["resnet18", "resnet34", "resnet50"]:
            model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "resnet50_pretrain":
            model = ResNet50(feat_dim=opt.feat_dim, freeze_layers=opt.frozen_layers)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None

    if opt.model_path is not None:
        print("model loaded")
        load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available() and opt.use_cuda is True:

        if torch.cuda.device_count() > 1:
            if opt.method == "MoCo" or opt.method == "MaskCon":
                model.encoder_k = torch.nn.DataParallel(model.encoder_k)
                model.encoder_q = torch.nn.DataParallel(model.encoder_q)
            else:
                model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        if linear is not None:
            linear = linear.cuda()
        if criterion1 is not None:
            criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        cudnn.benchmark = True

    return model, linear, criterion1, criterion2

def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    if torch.cuda.is_available() and opt.use_cuda is True:
        model.cuda()
    model.eval()

    return model


def curvature(opt, model, criterion1, criterion2):

    epss = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def model_gradient(opt, model, criterion1, criterion2):

    pass


def permutate_models(opt, model, criterion1, criterion2):

    pass


if __name__ == "__main__":

    opt = parse_opts()

    model, linear, criterion1, criterion2 = set_model(opt)
    train_loader, test_loader = set_loader(opt)


