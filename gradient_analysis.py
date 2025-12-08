import os
import argparse

import torch
import numpy as np

from main_linear import load_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from masked_datasets import ImageNet100_masked


def parse_option():
    parser = argparse.ArgumentParser('argument for gradient analysis')

    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18"])
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
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
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--data_mode", type=str, default="unmasked")

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


def load_model(opt, model):

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.to(opt.device)
    model.eval()

    return model


def read_gradients(model, data_loader, criterions, opt):

    criterion1, criterion2 = criterions
    grad_ssl = []
    grad_sup = []
    grad_all = []

    for idx, (images, images_masked, labels, _) in enumerate(data_loader):
        if "unmasked" in opt.data_mode:
            images1 = images[0]
            images2 = images[1]
        else:
            images1 = images_masked[0]
            images2 = images_masked[1]
        bsz = labels.shape[0]

        if opt.device == "cuda":
            images1 = images1.cuda()
            images2 = images2.cuda()

        images = torch.cat([images1, images2], dim=0)
        features = model(images)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

        loss_sup = criterion2(features, labels)
        loss_ssl = criterion1(features)
        loss_all = loss_sup + opt.alpha * loss_ssl
        g_sup = torch.autograd.grad(loss_sup, model.head[2].weight,
                                    retain_graph=True)[0]
        g_ssl = torch.autograd.grad(loss_ssl, model.head[2].weight,
                                    retain_graph=True)[0]
        g_all = torch.autograd.grad(loss_all, model.head[2].weight,
                                    retain_graph=True)[0]
        grad_sup.append(g_sup.detach.cpu().numpy())
        grad_ssl.append(g_ssl.detach.cpu().numpy())
        grad_all.append(g_all.detach.cpu().numpy())

    return grad_sup, grad_ssl, grad_all


def set_loader(opt):
    # construct data loader

    datasets = ImageNet100_masked(root=opt.data_root)
    data_sampler = None
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True, sampler=data_sampler,
                                              drop_last=True, persistent_workers=True)
    return data_loader


def grad_projection(grad1, grad2):

    # to project g1 to g2
    prod = np.inner(grad1, grad2)
    prod = prod / np.linalg.norm(grad2) / np.linalg.norm(grad2)
    proj = prod * grad2

    return proj


if __name__ == "__main__":

    opt = argparse()
    model, criterion1, criterion2 = set_model(opt)
    model = load_model(opt, model)
    data_loader = set_loader(opt)
    grad_sup, grad_ssl, grad_all = read_gradients(model, data_loader,
                                        (criterion1, criterion2), opt)




""" 
1. gradient projection
   - load model
   - load data (two views, annotation_mask)
   - backprojection
   - compute gradient projection
"""