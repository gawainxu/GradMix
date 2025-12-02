#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 23:33:17 2025

@author: zhi
"""

import pickle
import random
import argparse
import numpy as np


def parse_option():

    parser = argparse.ArgumentParser('argument for feature analysis')
    
    parser.add_argument("--feature_path", type=str, default="./features/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256_old_augmented_300_train")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--mode", type=str, default="intra")

    opt = parser.parse_args()

    return opt


def sortFeatures(mixedFeatures, labels, num_classes):
        
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])

    print("mixedFeatures", np.array(mixedFeatures).shape)
    
    for i, l in enumerate(labels):
        l = l.item()                         
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures


def downsampling(features):

   data_len = len(features)
   idx = random.sample(list(range(data_len)), 50)
   features_downsampled = [features[i] for i in idx]
   features_downsampled = np.array(features_downsampled)
   
   return features_downsampled


def specturm(features):
    
    m = features.dot(features.T)
    
    return m


if __name__ == "__main__":
    
    opt = parse_option()
    
    with open(opt.feature_path, "rb") as f:
        features_head, features_backbone, _, labels = pickle.load(f)
        
    sorted_features_head = sortFeatures(features_head, labels, opt.num_classes)
    sorted_features_backbone = sortFeatures(features_backbone, labels, opt.num_classes)
    
    if opt.mode == "intra":
        ms = []
        for c in range(opt.num_classes):
            downsampled_features = downsampling(sorted_features_head[c])
            m = specturm(downsampled_features)
            m = np.array(m, float)
            ms.append(m)
        
        with open("./features/0.05_intra", "wb") as f:
           pickle.dump(ms, f)
        
    else:
        ucs = []
        for c in range(opt.num_classes):
            uc = np.mean(np.array(sorted_features_head[c]), axis=0)
            ucs.append(uc)
            
        ucs = np.array(ucs)
        m = specturm(ucs)
        m = np.array(m, float)
        
        with open("./features/0.5_inter", "wb") as f:
           pickle.dump(m, f)
           