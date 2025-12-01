"""
This file is used for evaluating
the backbone features using mlp
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

from main_linear import set_loader, set_model, load_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from datautil import get_test_datasets
from networks.resnet_big import SupConResNet
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from util import set_optimizer, save_model
from util import AverageMeter


def parse_option():
    parser = argparse.ArgumentParser('argument for mlp probing')

    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18"])
    parser.add_argument("--backbone_model_path", type=str, default="")
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet",
                                 "imagenet100", "imagenet100_m", 'mnist', "svhn", "cub", "aircraft"], help='dataset')
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)

    opt = parser.parse_args()
    return opt



class mlp(nn.Module):

    def __init__(self, in_dim, hidden_size, num_classes):

        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):

        y = self.l1(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.l2(y)

        return y


def set_model(opt):

    criterion = nn.CrossEntropyLoss()
    classifier = mlp(in_dim=512, num_classes=opt.num_classes)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34" or opt.model == "resnet50":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    model = load_model(model, opt.backbone_model_path)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    print("Training start")
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            features = features.cuda(non_blocking=True)

        output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc = accuracy(output, labels, topk=(1, 5))
        acc1, acc5 = acc[0].item(), acc[1].item()
        top1.update(acc1, bsz)
        top5.update(acc5, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                  'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc = accuracy(output, labels, topk=(1, 5))
            acc1, acc5 = acc[0].item(), acc[1].item()
            top1.update(acc1, bsz)
            top5.update(acc5, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":

    opt = argparse()
    train_loader = set_loader(opt)
    test_dataset = get_test_datasets(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    optimizer = set_optimizer(opt, classifier)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, avg1_train, avg5_train = train(train_loader, model, classifier, criterion,
                                             optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy1:{:.2f}, accuracy5:{:.2f}, loss:{:.2f}'.format(
            epoch, time2 - time1, avg1_train, avg5_train, loss))

        loss_test, avg1_test, avg5_test = validate(test_loader, model, classifier, criterion, opt)
        print("test acc1", avg1_test)
        if avg1_test > best_top1_test_acc:
            best_top1_test_acc = avg1_test

        print("test acc5", avg5_test)
        if avg5_test > best_top5_test_acc:
            best_top5_test_acc = avg5_test

    save_file = opt.backbone_model_name.replace(".pth", "_linear_") + opt.temp_list + ".pth"
    save_file = os.path.join(opt.backbone_model_direct, save_file)
    save_model(model=model, linear=classifier, optimizer=optimizer, opt=opt, epoch=epoch, save_file=save_file)
    print("Best top1 accuacy is ", best_top1_test_acc)
    print("Best top5 accuacy is ", best_top5_test_acc)







