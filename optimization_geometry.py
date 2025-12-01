import os
import pickle
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd.functional import hessian

import numpy as np
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from networks.resnet_big import SupConResNet
from losses import SupConLoss
from datautil import mean_mapping, std_mapping, image_size_mapping, data_function_mapping
from datautil import label_to_dict, osr_splits_inliers
from util import TwoCropTransform



def parse_options():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument("--models_path", type=str, default="/home/zhi/projects/comprehensive_OSR_copy/save/SupCon/cifar100_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/")
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet34", "resnet50", "preactresnet18", "preactresnet34", "simCNN",
                                 "MLP"])
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="../datasets")
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_or_test", type=str, default="train",
                        help="which data to use, train or test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')

    opt = parser.parse_args()

    if torch.cuda.is_available() and opt.use_cuda == "gpu":
        opt.device = "cuda"
    else:
        opt.device = "cpu"

    return opt


def set_model(opt):

    in_channels = 3
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)

    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if opt.device == "cuda":
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()

        if criterion1 is not None:
            criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        cudnn.benchmark = True

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


def set_loader(opt):

    dataset = get_datasets(opt)

    if opt.dataset != "imagenet100":
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                                   num_workers=1, pin_memory=True, sampler=None,
                                                   drop_last=True,
                                                   persistent_workers=True)
    else:
        data_loader = dataset

    return data_loader


def get_datasets(opt):
    mean = mean_mapping[opt.dataset]
    std = std_mapping[opt.dataset]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.dataset]

    transform = transforms.Compose(
                [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.ToTensor(),
                 normalize, ])

    data_fun = data_function_mapping[opt.dataset]
    label_dict = label_to_dict(osr_splits_inliers[opt.dataset][opt.trail])
    transform = TwoCropTransform(transform)
    classes = osr_splits_inliers[opt.dataset][opt.trail]

    if opt.train_or_test == "train":
        train = True
    else:
        train = False

    dataset = data_fun(root=opt.data_root, train=train,
                       classes=classes, download=True,
                       transform=transform, label_dict=label_dict)
    print("dataset size", len(dataset))
    return dataset


def hessian_single_layer(model, x, labels, bsz, criterion1, criterion2):

    last_linear = model.head[-1]
    weight_matrix = last_linear.weight.clone()
    bias = last_linear.bias.clone()

    def compute_inpt(x, model):
        model.eval()
        inpt = model.encoder(x)
        inpt = model.head[0](inpt)
        inpt = model.head[1](inpt)

        return inpt

    def custome_forward(x):
        features = F.linear(inpt, weight_matrix, bias)
        features = F.normalize(features, dim=1)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss_sup = criterion2(features, labels)
        loss_ssl = criterion1(features)

        return loss_sup

    inpt = compute_inpt(x, model)
    #h_sup = hessian(custome_forward, inpt)
    loss_sup = custome_forward(inpt)
    g_sup = torch.autograd.grad(loss_sup, weight_matrix,
                                retain_graph=True)[0]
    for g in g_sup:
        h =  torch.autograd.grad(g, weight_matrix, allow_unused=True,
                                 retain_graph=True)[0]

    return h_sup


def main(opt):

    data_loader = set_loader(opt)
    model, criterion1, criterion2 = set_model(opt)

    losses_sup_all = []
    losses_ssl_all = []
    g_sup_all = []
    g_ssl_all = []
    h_sup_all = []
    h_ssl_all = []

    for epoch in range(opt.num_epochs):

        print("epoch", epoch)
        opt.model_path = os.path.join(opt.models_path, "ckpt_epoch_{}.pth".format(epoch))
        model = load_model(opt, model)

        losses_sup_epoch = []
        losses_ssl_epoch = []
        g_sup_epoch = []
        g_ssl_epoch = []
        h_sup_epoch = []
        h_ssl_epoch = []

        for idx, (images, labels, _) in enumerate(data_loader):
            images1 = images[0]
            images2 = images[1]
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

            print(loss_sup, loss_ssl)

            g_sup = torch.autograd.grad(loss_sup, model.head[2].weight,
                                        retain_graph=True)[0]
            g_ssl = torch.autograd.grad(loss_ssl, model.head[2].weight,
                                        retain_graph=True)[0]

            g_sup.requires_grad_(True)
            g_ssl.requires_grad_(True)

            h_sup, h_ssl = hessian_single_layer(model, images, labels, bsz, criterion1, criterion2)
            """
            flattened_grads = torch.cat(([grad.flatten() for grad in g_sup]))
            hessian = torch.zeros(flattened_grads.shape[0], flattened_grads.shape[0])
            for idx, grad in enumerate(g_sup):
                second_der = torch.autograd.grad(grad, model.head[2].weight, retain_graph=True, allow_unused=True)
                second_der = torch.cat(([grad.flatten() for grad in second_der]))
                hessian[idx, :] = second_der
            """

            losses_sup_epoch.append(loss_sup.detach().cpu().numpy())
            losses_ssl_epoch.append(loss_ssl.detach().cpu().numpy())
            g_sup_epoch.append(g_sup.detach().cpu().numpy())
            g_ssl_epoch.append(g_ssl.detach().cpu().numpy())
            #h_sup_epoch.append(h_sup.detach().cpu().numpy())
            #h_ssl_epoch.append(h_ssl.detach().cpu().numpy())

            opt.save_path = os.path.join(opt.models_path, "gradients_{}".format(epoch))
            with open(opt.save_path, "wb") as f:
                pickle.dump((losses_sup_epoch, losses_ssl_epoch, g_sup_epoch, g_ssl_epoch), f)

        losses_sup_all.append(losses_sup_epoch)
        losses_ssl_all.append(losses_ssl_epoch)
        g_sup_all.append(g_sup_epoch)
        g_ssl_all.append(g_ssl_epoch)
        #h_sup_all.append(h_sup_epoch)
        #h_ssl_all.append(h_ssl_epoch)





if __name__ == "__main__":

    opt = parse_options()
    main(opt)


