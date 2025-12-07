import os
import argparse

import torch
import torch.nn as nn

from main_linear import load_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from masked_datasets import ImageNet100_masked

from gradcam import backward_hook, forward_hook
from gradcam import process_heatmap, process_featuremap


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
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    model = load_model(model, opt.backbone_model_path)
    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)

    return model, criterion1, criterion2


def set_loader(opt):
    # construct data loader

    datasets = ImageNet100_masked(root=opt.data_root)
    data_sampler = None
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True, sampler=data_sampler,
                                              drop_last=True, persistent_workers=True)
    return data_loader


if __name__ == "__main__":

    opt = argparse()
    model, criterion1, criterion2 = set_model(opt)
    data_loader = set_loader(opt)





""" 
1. EPG for attribution maps
   - load model
   - load data (two views, annotation_mask)
   - compute attribution map
   - EPG computing 
2. gradient projection
   - load model
   - load data (two views, annotation_mask)
   - backprojection
   - compute gradient projection
"""