import cv2
import torch
import torch.utils.data as data_utils
import os, sys

import pickle
import numpy as np
from scipy import misc
import struct
import copy
import logging

from cfgs.config import cfg
from morph import warp

def get_imglist(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

def read_data(img_path, label_path, affine_trans=False, hflip=False, scale_x=1.0, scale_y=1.0, warp_ratio=0):
    img = misc.imread(img_path, mode='RGB')
    f = open(label_path, "rb")
    binary_data = f.read()
    label = []
    for idx in range(len(binary_data)):
        label.append(struct.unpack('B', binary_data[idx:idx+1])[0])
    label = np.array(label, dtype=np.uint8)

    # affine
    h, w, c = img.shape
    label = np.reshape(label, (h, w))

    if warp_ratio > 0 and np.random.rand() <= warp_ratio:
        img, label = warp(img, label)

    if affine_trans:
        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
        label = cv2.resize(label, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

    if hflip and np.random.rand() > 0.5:
        img = cv2.flip(img, flipCode=1)
        label = cv2.flip(label, flipCode=1)

    h, w, c = img.shape
    label = np.reshape(label, (h * w))
    img = np.swapaxes(np.swapaxes(img, 1,2), 0,1)
    return [img, label]

class Dataset(data_utils.Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, train_or_test, shuffle=True, affine_trans=True, hflip=True, warp=True):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == "train" else cfg.test_list
        self.train_or_test = train_or_test
        self.affine_trans = affine_trans
        self.hflip = hflip
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.warp_ratio = 0.5 if warp == True else 0
        fname_list = [fname_list] if type(fname_list) is not list else fname_list

        self.imglist = []
        for fname in fname_list:
            self.imglist.extend(get_imglist(fname))
        self.shuffle = shuffle

    def __getitem__(self, index):
        if self.affine_trans:
            self.scale_x = (np.random.uniform() - 0.5) / 4 + 1
            self.scale_y = (np.random.uniform() - 0.5) / 4 + 1

        img_path = self.imglist[index]
        label_path = img_path.replace('image', 'label').replace('png', 'dat')
        img, label = read_data(img_path, label_path, self.affine_trans, self.hflip, self.scale_x, self.scale_y, self.warp_ratio)
    
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.imglist)



if __name__ == '__main__':
    ds = Dataset('train')
    num_data = ds.__len__()
    print('共%d个数据'%num_data)
    import time
    START = time.time()
    for i in range(num_data):
        img, label = ds.__getitem__(i)
        #print(img.size(), label.size())
    END = time.time()
    print('读取共耗时%.4fs'%(END-START))