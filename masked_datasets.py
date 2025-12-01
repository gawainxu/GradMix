#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:03:47 2025

@author: zhi
"""

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
    


if __name__ == "__main__":
    
    root = "/home/zhi/projects/datasets"
    my_subsects = my_VOC_subsects(root)
    img, l, m = my_subsects[45]
        