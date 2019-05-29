#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:53:20 2019

@author: luoyao
"""

import torch.nn as nn
import torchvision

class sia_net(nn.Module):
    def __init__(self , model):
        super(sia_net, self).__init__()
        #取掉model的后两层
        self.fc1 = nn.Sequential(
                nn.Sequential(*list(model.children())[:-2]),
                nn.AdaptiveAvgPool2d(1))

        self.fc1_0 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 512))

    def forward_once(self, x):
        x = self.fc1(x)
        x = x.view(x.size()[0], -1) 
        feature = self.fc1_0(x)     #feature

        return feature
    
    def forward(self, input_l, input_r):
        feature_l = self.forward_once(input_l)
        feature_r = self.forward_once(input_r)

        return feature_l, feature_r

def load_resnet50():
    resnet = torchvision.models.resnet50()
    model = sia_net(resnet)
    
    return model