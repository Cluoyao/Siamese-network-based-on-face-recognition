#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:19:18 2019

@author: luoyao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Learn_simi_metri_loss(nn.Module):
    """
    S. Chopra, R. Hadsell and Y. LeCun, 
    Learning a similarity metric discriminatively, with application to face verification
    """
    def __init__(self, opt_style='resample', margin=5):
        super(Learn_simi_metri_loss, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

    def forward(self, feature_l, feature_r, label):
        if self.opt_style == 'resample':
            #feature contrain
            E_w_f = F.pairwise_distance(feature_l, feature_r, p=1, keepdim = True)

            Q = self.margin
            e = 2.71828
            iden_contrastive = label * (2/Q) * E_w_f**2 + (1-label)*2 * Q * e**(-2.77*E_w_f/Q)

            total_loss = iden_contrastive.mean()
            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')

class Dim_reduction_loss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, opt_style='resample', margin=1.0):
        super(Dim_reduction_loss, self).__init__()
        self.opt_style = opt_style
        self.margin = margin

    def forward(self, feature_l, feature_r, label):
        if self.opt_style == 'resample':
            #feature contrain
            diff_iden = F.pairwise_distance(feature_l, feature_r, p=2, keepdim = True)
            iden_contrastive = 0.5 * (label) * torch.pow(diff_iden, 2) + (1-label) * torch.pow(torch.clamp(self.margin - diff_iden, min=0.0), 2)

            total_loss = iden_contrastive.mean()
            return total_loss
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')