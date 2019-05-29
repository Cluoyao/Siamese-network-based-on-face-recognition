#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:14:40 2019

@author: luoyao
"""

import torch.utils.data as data
import os
import numpy as np 
import random
import os.path as osp
import torch

from pathlib import Path
from collections import defaultdict
from PIL import Image

def mkdir(d):
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system(f'mkdir -p {d}')

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def read_pairs_ddfa(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

def create_label_dict(path):
    label_dict = defaultdict(list)
    names_list = Path(path).read_text().strip().split('\n')
    for f_name in names_list:
        f_s = f_name.split('\000')
        label_dict[int(f_s[1])].append(f_s[0])
    
    return label_dict

def split_label(path):
    names_list = Path(path).read_text().strip().split('\n')
    img_name_nlabel = []
    for img_name in names_list:
        img_name_nlabel.append(img_name.split('\000')[0])
        
    return img_name_nlabel

class LFW_Pairs_Dataset(data.Dataset):
    def __init__(self, lfw_dir, pairs_txt, transform=None):
        self.transform = transform
        self.pairs = read_pairs(pairs_txt)
        self.lfw_dir = lfw_dir
        self.img_loader = Image.open

    def __getitem__(self, index):
        issame = None
        pair = self.pairs[index]
        if len(pair) == 3:
            path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(self.lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            img0 = self.img_loader(path0)
            img1 = self.img_loader(path1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, issame

    def __len__(self):
        return len(self.pairs)

class DDFA_Dataset(data.Dataset):
    def __init__(self, root, filelists, transform=None):
        self.root = root
        self.transform = transform
        self.label_dict = create_label_dict(filelists)
        self.lines = split_label(filelists)
        self.img_loader = Image.open

    def __getitem__(self, index):
        #random choose person1~personN
        label_1 = random.choice( range(len(self.label_dict)) )
        img1_name = random.choice( self.label_dict[label_1] )
        
        # %50 for same, %50 for diff 0:indicates diff,1:indicates same
        #is_same = random.randint(0,1)    
        # 100% probability
        #is_same = 1    #constrict the same people
        # 60% for different, 40% for same
        is_same = np.random.choice([0,1], p=[0.6, 0.4])

        if is_same:
            img2_name = random.choice(self.label_dict[label_1])
        else:
            while True:
                label_2 = random.choice( range(len(self.label_dict)) )
                if label_2 != label_1:
                    break
            img2_name = random.choice( self.label_dict[label_2] )

        img1_path = osp.join(self.root, img1_name)
        img2_path = osp.join(self.root, img2_name)
        
        # Only add the pair if both paths exist
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = self.img_loader(img1_path)
            img2 = self.img_loader(img2_path)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32))

    def __len__(self):
        return len(self.lines)
    
class DDFA_Pairs_Dataset(data.Dataset):
    def __init__(self, root, pairs_txt, transform=None):
        self.transform = transform
        self.pairs = read_pairs_ddfa(pairs_txt)
        self.root = root
        self.img_loader = Image.open

    def __getitem__(self, index):

        pair = self.pairs[index]
        img0_name = pair[0]
        img1_name = pair[1]
        if int(pair[2]):
            issame = True
        else:
            issame = False
        path0 = os.path.join(self.root, img0_name)
        path1 = os.path.join(self.root, img1_name)

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            img0 = self.img_loader(path0)
            img1 = self.img_loader(path1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, issame

    def __len__(self):
        return len(self.pairs)