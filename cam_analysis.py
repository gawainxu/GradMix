#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 23:05:06 2024

@author: zhi
"""

import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import PIL

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from networks.resnet_big import SupConResNet, LinearClassifier
from masked_datasets import ImageNet100_masked
from losses import SupConLoss

# https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569
# https://github.com/jacobgil/pytorch-grad-cam

# global variables
gradients = None
activations = None


def parse_option():
    parser = argparse.ArgumentParser('argument for grad cam')
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["imagenet100_m", 'cifar10', "tinyimgnet", "imagenet100"],
                        help='dataset')
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--trail", type=int, default=0)

    # for "single" and "sim"
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--feature_id", type=int, default=5)
    parser.add_argument("--bsz", type=int, default=256)

    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth")
    parser.add_argument("--mode", type=str, default="supcon")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.parent_dir = Path(opt.main_dir).parent.absolute()
    opt.data_root = os.path.join(opt.parent_dir, opt.data_root)

    opt.output_path = opt.main_dir + "/cam/"
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    opt.model_path = opt.main_dir + opt.model_path

    opt.feature_path = "./featuremaps/" + str(opt.class_idx)
    if not os.path.exists(opt.feature_path):
        os.makedirs(opt.feature_path)

    return opt


def load_model(opt, model):

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.cpu()
    model.eval()

    return model


def backward_hook(module, grad_input, grad_output):
    global gradients
    print("Backward Hook Running")
    gradients = grad_output
    print(f"Gradient Size: {gradients[0].size()}")


def forward_hook(module, args, output):
    global activations
    print("Forward Hook Running")
    activations = output
    print(f"Gradient Size: {activations.size()}")


def process_heatmap(heatmap, img, save_path, opt):
    plt.close('all')

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu()
    img = img.detach().cpu()

    # draw the heatmap
    #plt.matshow(heatmap)

    # Create a figure and plot the first image
    fig, ax = plt.subplots()
    ax.axis('off')  # removes the axis markers

    # First plot the original image
    ax.imshow(to_pil_image(img, mode='RGB'))

    # Resize the heatmap to the same size as the input image and defines
    # a resample algorithm for increasing image resolution
    # we need heatmap.detach() because it can't be converted to numpy array while
    # requiring gradients
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((opt.img_size, opt.img_size), resample=PIL.Image.BICUBIC)

    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Plot the heatmap on the same axes,
    # but with alpha < 1 (this defines the transparency of the heatmap)
    #ax.imshow(overlay, alpha=0.2, interpolation='nearest')

    # Show the plot
    # plt.show()
    fig.savefig(save_path)
    return overlay


def process_featuremap(feature_maps, img, opt):
    feature_maps = feature_maps.detach()
    num_features = feature_maps.shape[0]
    for n in range(num_features):
        plt.close("all")
        fig, ax = plt.subplots()
        ax.axis('off')  # removes the axis markers
        ax.imshow(to_pil_image(img, mode='RGB'))
        f_n = feature_maps[n, :, :]
        f_n = to_pil_image(f_n, mode="F").resize((opt.img_size, opt.img_size), resample=PIL.Image.BILINEAR)
        cmap = colormaps['jet']
        f_n = (255 * cmap(np.asarray(f_n) ** 2)[:, :, :3]).astype(np.uint8)
        ax.imshow(f_n, alpha=0.4, interpolation='nearest')
        plt.savefig(opt.feature_path + "/" + str(n) + ".png")


def EPG_cam(cammap, mask):

    # normalize the cammap
    cammap = (cammap - cammap.min()) / (cammap.max() - cammap.min())
    cammap = np.asarray(cammap)[:, :, 0]
    tp = np.sum(np.multiply(cammap, mask))
    p = np.sum(mask)
    n = mask.shape[0] * mask.shape[1] - p
    fn = np.sum(np.multiply(cammap, 1-mask))

    recall = tp * 1. / (tp + fn)
    acc = tp * 1. / p

    return recall, acc


if __name__ == "__main__":

    opt = parse_option()

    # load model
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=3)
    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)
    model = load_model(opt, model)

    model.eval()
    model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    datasets = ImageNet100_masked(root=opt.data_root)
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=opt.bsz, shuffle=True,
                                              num_workers=4, pin_memory=True, sampler=None,
                                              drop_last=True, persistent_workers=True)

    # register hook
    backward_hook = model.encoder.layer4[-1].register_full_backward_hook(backward_hook)  # 4
    forward_hook = model.encoder.layer4[-1].register_forward_hook(forward_hook)  # 4

    scores = []

    for idx, (images, _, labels, masks) in enumerate(data_loader):
        print(idx)
        images1 = images[0]
        images2 = images[1]
        images = torch.cat([images1, images2], dim=0)
        masks = masks.numpy()

        features = model(images)
        features1, features2 = torch.split(features, [opt.bsz, opt.bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

        if opt.mode == "ssl":
            loss = criterion1(features)
        else:
            loss = criterion2(features, labels)
        optimizer.zero_grad()
        loss.backward()

        activation_maps = activations * F.relu(gradients[0])
        cam = torch.mean(activation_maps, dim=1)
        heatmap = F.relu(cam)
        for i in range(opt.bsz):
            save_path = opt.output_path + str(idx * opt.bsz + i) + ".png"
            cammap = process_heatmap(cam[i], images1[i], save_path, opt)
            score = EPG_cam(cammap, masks[i])
            scores.append(score)

    backward_hook.remove()
    forward_hook.remove()
    print("EPG score is", sum(scores) / len(scores))



"""
1. EPG for attribution maps
   - load model
   - load data (two views, annotation_mask)
   - compute attribution map
   - EPG computing 
"""