#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains helper functions to load the datasets.

@author: Iraj Masoudian
@author: Doguhan Bahcivan
@author: Daniel Tiet
@author: Erik Yi-Li Sun Gal
@author: Hung Luu
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, inputDim, trainDir, labelDir,ids=None):
        """
        

        Parameters
        ----------
        inputDim : Int
            The format of the image. An image of format 3xHxW will be 
            transformed to 3x(inputDim)x(inputDim). The original format of the
            images has to be such that hight and width are the same (H=W).
        trainDir : Str
            Folder in which all images are saved.
        labelDir : Str
            csv-file which contains all the annotations for the images, 
            i.e. age, sex, id, disease.
        ids : str, optional
            npy-file in which the ids of the images, which should be used,
            are saved. By default all ids from the labelDir file are used.

        Returns
        -------
        None.

        """
        self.trainDir = trainDir
        labels=pd.read_csv(labelDir)
        self.age=np.array(labels["age"])
        self.tIds=np.array(labels["ID"])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(inputDim)
        ])
        if ids is not None:
            self.ids=np.load(ids)
        else:
            self.ids=self.tIds
        self.N_ids=self.ids.shape[0]

        # dont keep it open
        del labels
        
    def getFile(self, x):
        """
        

        Parameters
        ----------
        x : Int
            Index of the sample.

        Returns
        -------
        cIdStr : Str
            Image file name.
        cId : Int
            Id of the image.

        """

        cIdStr=""
        cId=int(self.ids[x])
        cIdStr=str(cId)+".png"
        return cIdStr, cId
    
    def __len__(self):
        """
        

        Returns
        -------
        l : Int
            Length of the dataset.

        """
        l=self.N_ids
        return l

    def __getitem__(self, idx):
        """
        

        Parameters
        ----------
        idx : Int
            Index of the sample.

        Returns
        -------
        data : torch.tensor
            Image transformed as a torch.tensor.
        label : torch.tensor
            Age transformed as a torch.tensor.

        """
        # data 
        cIdStr, cId = self.getFile(idx)
        img = Image.open(os.path.join(self.trainDir, cIdStr))
        data = np.array(img)
        data=self.transform(data).double()

        con=np.where(self.tIds==cId)
        
        label=torch.tensor(self.age[con]).double()
        
        return data, label
