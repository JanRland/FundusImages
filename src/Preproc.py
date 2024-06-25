#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains helper functions we used for the preprocessing.

@author: Iraj Masoudian
@author: Doguhan Bahcivan
@author: Daniel Tiet
@author: Erik Yi-Li Sun Gal
@author: Hung Luu
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import numpy as np


def crop(image):
    """
    

    Parameters
    ----------
    image : numpy array
        Image to be cropped.

    Returns
    -------
    rImg : numpy array
        Cropped image.

    """
    non_black_rows = np.where(image.sum(axis=1) > 2000)[0]
    non_black_cols = np.where(image.sum(axis=0) > 2000)[0]

    nImg=image.copy()
    rImg=nImg[min(non_black_rows):(max(non_black_rows)+1),min(non_black_cols):(max(non_black_cols)+1),:]

    return rImg

def getChannelDist(x, j):
    """
    

    Parameters
    ----------
    x : PIL Image, numpy array
        Image. 
    j : Int
        Color channel to return.

    Returns
    -------
    xNp : numpy array
        color channel of the image. 

    """
    xNp=np.array(x.copy())
    return xNp[:,:,j]


def getProbs(x, cutoff):
    """
    

    Parameters
    ----------
    x : numpy array
        Flattened image with one color channel
    cutoff : Int
        All values below the cutoff are not considered for the probability
        calculations. 

    Returns
    -------
    probs : List
        List of the relative frequencies of the color intensities. 
        The index in the list corresponds to the color intensity, e.g.
        probs[32]=0.3, then 30% of the pixels on the image had the intesity
        of 32. 

    """
    probs=[0]*cutoff
    N=np.where(x>cutoff)[0].shape[0]
    for i in range(cutoff,256):
        n_i=np.where(x==i)[0].shape[0]
        probs.append(n_i/float(N))
    return probs

def getCDF(x):
    """
    

    Parameters
    ----------
    x : List
        List of the relative frequencies of the color intensities. 

    Returns
    -------
    cdf : List
        Cumulative probability function.

    """

    cdf=[]
    probs=getProbs(x)
    cProb=0
    for i in probs:
        cProb+=i
        cdf.append(cProb)
    return cdf

def newImage(x):
    """
    

    Parameters
    ----------
    x : List
        List of the relative frequencies of the color intensities. 

    Returns
    -------
    newPixels : List
        List of transformed pixel values.

    """
    newPixels=[]
    cP=0
    for p in x:
        #print(cP)
        cP+=p
        newPixels.append(int(cP*255))
    return newPixels

def createEqualizedImage(x):
    """
    

    Parameters
    ----------
    x : PIL Image, numpy array
        Image that should be transformed

    Returns
    -------
    y : numpy array
        Equalized image.

    """
    
    red=getChannelDist(x, 0)
    green=getChannelDist(x, 1)
    blue=getChannelDist(x, 2)

    prob_red=getProbs(red.flatten(), 20)
    prob_green=getProbs(green.flatten(), 20)
    prob_blue=getProbs(blue.flatten(), 20)

    N=red.flatten().shape[0]

    new_red=np.array(newImage(prob_red, N)).astype("int")
    new_green=np.array(newImage(prob_green, N)).astype("int")
    new_blue=np.array(newImage(prob_blue, N)).astype("int")

    y=np.array(x.copy())
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j,0]=new_red[y[i,j,0]]
            y[i,j,1]=new_green[y[i,j,1]]
            y[i,j,2]=new_blue[y[i,j,2]]
    return y