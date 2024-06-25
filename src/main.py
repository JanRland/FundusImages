#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script conducts the learning of the networks. We ran the script
on the HPC HHU Hilbert with 5 gpus and 4 cpus. 

@author: Iraj Masoudian
@author: Doguhan Bahcivan
@author: Daniel Tiet
@author: Erik Yi-Li Sun Gal
@author: Hung Luu
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""
from loader import ImageDataset
from model import getModel, loadModel
from util import train, evaluation_loop
import sys
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)  

if __name__ == "__main__":
    
    # hyperparameters
    # [0] -> filename
    # [1] -> learning rate
    # [2] -> epochs
    # [3] -> batch_size
    # [4] -> num_gpus
    # [5] -> num_cpus
    # [6] -> model m
    # [7] -> weight decay
    # [8] -> ID
    inputParameters=sys.argv
    if len(inputParameters)<2:
        print("No arguments")
        sys.exit(-1)
        
    lr = float(inputParameters[1])
    np.random.seed(42)
    torch.manual_seed(42)
    m=int(inputParameters[6])
    batch_size = int(inputParameters[3])
    n_epochs = int(inputParameters[2])
    num_gpus = int(inputParameters[4])
    num_cpus = int(inputParameters[5])
    weight_decay =  float(inputParameters[7])
    fileId = int(inputParameters[8])
    

    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model=getModel(m)
    model.to(device)
    model.load_state_dict(loadModel(m))
    model.eval()
    model=torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
    model.double()
    
    # loss function MAE
    loss_function= nn.L1Loss()
    
    # data loader
    if m==3:
        inputDim=299
    else:
        inputDim=224
    trainSet=ImageDataset(inputDim, "mages", "train.csv",ids="trainIds_all.npy")
    valSet=ImageDataset(inputDim, "images", "val.csv")
    testSet=ImageDataset(inputDim, "images", "test.csv")
    
    trainLoader=DataLoader(trainSet, pin_memory=True,batch_size=batch_size,num_workers=num_cpus,shuffle=True, drop_last=True)
    valLoader=DataLoader(valSet, pin_memory=True,batch_size=batch_size,num_workers=num_cpus,shuffle=True, drop_last=True)
    testLoader=DataLoader(testSet, pin_memory=True,batch_size=batch_size,num_workers=num_cpus,shuffle=True, drop_last=True)
    
    # optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,verbose=True)
    
    
    # run training
    print("-------------------------------------------------------")
    print("Hyperparameters")
    print(inputParameters)
    print("Saved best result as:")
    saveDir="res_" + str(m) + "_" + str(fileId) + ".pth"
    print(saveDir)
    
    
    print("Initial Loss Train")
    init_loss=evaluation_loop(loss_function, trainLoader, model, device)
    print(init_loss)
    print("Initial Val Train")
    init_loss=evaluation_loop(loss_function, valLoader, model, device)
    print(init_loss)
    
    trainLoss, valLoss=train(model,n_epochs,loss_function,optimizer,scheduler, trainLoader, valLoader, saveDir,device, m)
    print("-------------------------------------------------------")
    print("trainLoss:")
    print(trainLoss)
    print("valLoss:")
    print(valLoss)
    print("-------------------------------------------------------")
    test_loss=evaluation_loop(loss_function, testLoader, model, device)
    print("Test Loss")
    print(test_loss)


