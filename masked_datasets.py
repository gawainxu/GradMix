#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:03:47 2025

@author: zhi
"""

import os
import cv2
from voc_base import VOCFull
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import ImageFilter, Image
import numpy as np
import scipy


class my_VOC(VOCFull):
    
    """
    return the original voc dataset
    """
    
    def __init__(self, root, train):
        super(my_VOC, self).__init__(root=root, image_set=train, download=True)
                    
    def __getitem__(self, index):
        img, seg_target, dec_target = super().__getitem__(index)
        
        dec_target = dec_target["annotation"]["object"]
        if len(dec_target) > 0:
            dec_target = dec_target[0]
         
        # img is Image and of different sizes    
        return img, dec_target["name"], seg_target


class my_VOC_subsects(Dataset):
    
    """
    return a subset of voc dataset containing the classes in dict
    """
    
    # TODO check the classes
    class_names = ['dog','aeroplane', 'bicycle', "train", 'bird',
                   'boat', 'pottedplant', 'diningtable']
    
    def __init__(self, root, train="train", transform=None, class_names=class_names, mask=True):
        self.myvoc = VOCFull(root)
        self.train = train
        self.ori_len = len(self.myvoc)
        self.class_names = class_names
        self.transform = transform
      
        self.mask = mask
        self.filter_classes()
        
        
    def filter_classes(self):
        
        name_class_dict = {'dog': 0, 'aeroplane': 1, 'bicycle': 2,
                           'train': 3, 'bird': 4, 'boat': 5,
                           'pottedplant': 6, 'diningtable': 7}      
        self.data = []
        self.targets_dec = []
        self.targets_seg = []
        self.seg_bbox = []
        
        # pick the classes in dict
        for i in range(self.ori_len):
            i, dec_name, dec_bbox, l_seg = self.myvoc[i]
            if dec_name in self.class_names:
                self.data.append(i)
                self.targets_dec.append(name_class_dict[dec_name])
                self.targets_seg.append(l_seg)
                self.seg_bbox.append(dec_bbox)

    
    def __getitem__(self, index):
                
        if self.mask:
            print("masking")
            img = my_VOC_subsects.masking(self.data[index], self.targets_seg[index], self.seg_bbox[index])
        else:
            img = self.data[index]
            
        if self.transform is not None:
            return self.transform(img), self.targets_dec[index], self.targets_seg[index]
        else:
            return  img, self.targets_dec[index], self.targets_seg[index]
            
           
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def masking(img, mask, bbox):
        
        """
        img and mask are both PIL Image
        mask is 0/1 mask
        TODO how if just cut out the object???
        """
        # convert img and mask to np array
        img_np = np.array(img)
        mask_np = np.array(mask)
        # broadcast mask
        mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
        #mask_np_exp = scipy.ndimage.binary_dilation(mask_np)
        # mask * image and filter  --> masked_filtered_img
        processed_mask = np.where(mask_np >= 1, 1, 0)
        img_mask = np.multiply(img_np, processed_mask)
        
        return img_mask


class ImageNet100_masked(Dataset):

    # todo align with the class oders in imagenet-s
    def __init__(self, root, merge=True):

        self.data_train_path = os.path.join(root, "imagenet100_train")
        self.data_test_path = os.path.join(root, "imagenet100_test")
        self.mask_train_path = os.path.join(root, "ImageNetS919/train-semi-segmentation")
        self.mask_test_path = os.path.join(root, "ImageNetS919/validation-segmentation")

        self.train_data = []
        self.train_masks = []
        self.test_data = []
        self.test_masks = []
        self.train_labels = []
        self.test_labels = []

        self.class_dir_list = sorted(os.listdir(self.data_train_path))
        self.mask_dir_list = sorted(os.listdir(self.mask_train_path))
        self.class_map = {serie_num: index for index, serie_num in enumerate(self.class_dir_list)}

        self.transform = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
        self.newsize = 224

        for cd in self.class_dir_list:
            if cd not in self.mask_dir_list:
                continue
            class_data_dir = os.path.join(self.data_train_path, cd)
            class_mask_dir = os.path.join(self.mask_train_path, cd)
            label = self.class_map[cd]
            class_file_list = os.listdir(class_data_dir)
            class_mask_list = os.listdir(class_mask_dir)
            class_mask_list = [n.split(".")[0] for n in class_mask_list]
            for file_name in class_file_list:
                if file_name.split(".")[0] in class_mask_list:
                    im = Image.open(class_data_dir + "/" + file_name)
                    im = im.convert("RGB")
                    mask = Image.open(class_mask_dir + "/" + file_name.split(".")[0] + ".png")
                    self.width, self.height = im.size
                    if self.width <= self.newsize or self.height <= self.newsize:
                        im = im.resize((self.newsize, self.newsize))
                        mask = mask.resize((self.newsize, self.newsize))
                    else:
                        im = im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        mask = mask.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                    self.train_data.append(im)
                    mask = np.asarray(mask)[:,:, 0]
                    assert np.sum(mask)>=0
                    mask = (mask > 0).astype(int)
                    self.train_masks.append(mask)
                    self.train_labels.append(label)

            class_data_dir = os.path.join(self.data_test_path, cd)
            class_mask_dir = os.path.join(self.mask_test_path, cd)
            label = self.class_map[cd]
            class_file_list = os.listdir(class_data_dir)
            class_mask_list = os.listdir(class_mask_dir)
            class_mask_list = [n.split(".")[0] for n in class_mask_list]
            for file_name in class_file_list:
                if file_name.split(".")[0] in class_mask_list:
                    im = Image.open(class_data_dir + "/" + file_name)
                    im = im.convert("RGB")
                    mask = Image.open(class_mask_dir + "/" + file_name.split(".")[0] + ".png")
                    if self.width <= self.newsize or self.height <= self.newsize:
                        im = im.resize((self.newsize, self.newsize))
                        mask = mask.resize((self.newsize, self.newsize))
                    else:
                        im = im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        mask = mask.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                    self.test_data.append(im)
                    mask = np.asarray(mask)[:,:, 0]
                    assert np.sum(mask) >= 0
                    mask = (mask > 0).astype(int)
                    self.test_masks.append(mask)
                    self.test_labels.append(label)

        if merge:
            self.train_data = self.train_data + self.test_data
            self.train_masks = self.train_masks + self.test_masks
            self.train_labels = self.train_labels + self.test_labels

    def __getitem__(self, idx):

        img = self.train_data[idx]
        #img = self.transform(img)

        return img, self.train_labels[idx], self.train_masks[idx]

    def __len__(self):

        return len(self.train_data)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    root = "/home/zhi/projects/datasets"
    my_subsects = ImageNet100_masked(root)

    for i in range(len(my_subsects)):
        img, l, m = my_subsects[i]
        img = np.asarray(img)
        m = np.repeat(m[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        imgm = np.multiply(img, m)
        #imgm = Image.fromarray(imgm, 'RGB')
        #imgm.show()
        cv2.imwrite("/home/zhi/projects/temp/" + str(i) + ".png",  cv2.cvtColor(imgm, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/home/zhi/projects/temp/" + str(i) + "_ori.png",  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        