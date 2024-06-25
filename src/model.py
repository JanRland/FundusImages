#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains helper functions to call the networks.

@author: Iraj Masoudian
@author: Doguhan Bahcivan
@author: Daniel Tiet
@author: Erik Yi-Li Sun Gal
@author: Hung Luu
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import torch
from torchvision import models
import torch.nn as nn


def getModel(m):
    """
    

    Parameters
    ----------
    m : Int
        Index of the network, that will be used:
            1: ResNet152
            2: DenseNet201
            3: Inception-v3

    Returns
    -------
    model : torch.nn class
        The neural network that was chosen. 

    """
    model=None
    if m==0:
        pass

    elif m==1:
        model=models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        
    elif m==2:
        model=models.densenet201(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
        
    elif m==3:
        model=models.inception_v3(init_weights=False)
        
        # aux net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, 1)
        
        # primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)

    return model



def loadModel(m):
    """
    

    Parameters
    ----------
    m : Int
        Index of the network, that will be used:
            1: ResNet152
            2: DenseNet201
            3: Inception-v3
        The loaded weights correspond to the pretrained=True weights
        from the getModel() function. In the analysis we had to manually
        save the pretrained weights as the high performance cluster did
        not provide internet access.

    Returns
    -------
    state_dic : Dictionary
        Dictionary containing the weights of the corresponding model.

    """
    state_dic=None
    if m==1:
        # weights provided by pytorch
        state_dic=torch.load("resnet152.pth")
    elif m==2:
        # weights provided by pytorch
        state_dic=torch.load("densenet201.pth")
    elif m==3:
        # weights provided by pytorch
        state_dic=torch.load("inception.pth")
    return state_dic
