#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:25:53 2019

@author: luoyao
"""

import torch
import matplotlib.pylab as plt
import numpy as np
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from siamese_network import load_resnet50
from siamese_loss import Learn_simi_metri_loss, Dim_reduction_loss
from siamese_utils import Normalize, LFW_Pairs_Dataset, DDFA_Dataset, mkdir
## config
devices_id = [0]
base_lr = 0.0001
lr = base_lr
momentum = 0.9
weight_decay = 5e-4
batch_size = 32
workers = 8
start_epoch = 1
epochs = 50
log_file = "./training_debug/logs/contrastive_loss/"
snapshot = "./training_debug/logs/model/"
mkdir(log_file)
mkdir(snapshot)

root = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
ddfa_train_dir = "./label_train_aug_120x120.list.train"


def adjust_lr(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr
    lr = base_lr
    for param_group in optimizer.param_groups:
        lr = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint fo {filename}')


def train(train_loader, model, criterion, optimizer, epoch):
    #status:training!
    model.train()
  
    for i, (img_l, img_r, label) in enumerate(train_loader):

        label.requires_grad = False
        label = label.cuda(non_blocking = True)

        feature_l, feature_r = model(img_l, img_r)

        loss = criterion(feature_l, feature_r, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #log
        if i % epochs == 0:
            print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, lr, loss.data.cpu().numpy()))
            print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, lr, loss.data.cpu().numpy()), file=open(log_file + 'contrastive_print.txt','a'))



def main():
    
    #step1:define the model structure
    model = load_resnet50()
    torch.cuda.set_device(devices_id[0])
    model = nn.DataParallel(model, device_ids=devices_id).cuda()

    #step2: loss and optimization method
    criterion = Learn_simi_metri_loss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = base_lr,
                                momentum = momentum,
                                weight_decay = weight_decay,
                                nesterov = True)

    #step3:data
    normalize = Normalize(mean=127.5, std=128)
    
#    train_dataset = LFW_Pairs_Dataset(
#            lfw_dir=lfw_train_dir,
#            pairs_txt=lfw_train_pairs, 
#            transform = transforms.Compose([transforms.ToTensor(), normalize])
#            )
    
    train_dataset = DDFA_Dataset(
            root = root,
            filelists = ddfa_train_dir,
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=workers,
                              shuffle=False, pin_memory=True, drop_last=True)

    cudnn.benchmark = True
    
    for epoch in range(start_epoch, epochs+1):
        #adjust learning rate
        adjust_lr(optimizer, base_lr, epoch, epochs, 30)
        #train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        #save model paramers
        filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
                {
                        'epoch':epoch,
                        'state_dict':model.state_dict()
                },
                filename
                )
        #validate(val_loader, model, criterion, epoch)


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
         bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


if __name__ == "__main__":

    main()

    ## observe model structure
#    model = load_resnet50()


    ## abserve batch sample data, set batch size = 8
#    train_dataset = DDFA_Dataset(
#            root = root,
#            filelists = ddfa_train_dir,
#            transform = transforms.Compose([transforms.ToTensor()])
#            )
#    
#    train_loader = DataLoader(train_dataset, batch_size = 8, num_workers=workers,
#                              shuffle=False, pin_memory=True, drop_last=True)
#    dataiter = iter(train_loader)
#    
#    example_batch = next(dataiter)
#    concatenated = torch.cat((example_batch[0],example_batch[1]), 0)
#    imshow(torchvision.utils.make_grid(concatenated))
#    print(example_batch[2].numpy())








