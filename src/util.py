#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains helper functions to train the networks.

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

def train(model,epochs,loss_function,optimizer,scheduler, trainLoader, valLoader, saveDir,device, m):
    """
    

    Parameters
    ----------
    model : torch.nn class
        Network that should be trained
    epochs : Int
        Number of epochs the model should be trained for. 
    loss_function : torch.nn Loss function
        Loss function for the training
    optimizer : torch.optim
        Optimizing strategy like stochastic gradient descent
    scheduler : torch.optim.lr_scheduler
        Scheduler to reduce the learning rate, when pateau is reached
    trainLoader : torch.utils.data.DataLoader
        Loads the training set
    valLoader : torch.utils.data.DataLoader
        Loads the validation set
    saveDir : Str
        Directory to save the best result in
    device : Str
        "cuda" or "cpu" depending on what is used
    m : Int
        Index of the network, that will be used:
            1: ResNet152
            2: DenseNet201
            3: Inception-v3
        For Inception-v3 the images have to be formated to 3x299x299 instead
        of 3x224x224 and the auxiliary network has to be trained
        
    Returns
    -------
    train_losses : List
        List of the loss per epoch on the training set 
    val_losses : List
        List of the loss per epoch on the validation set

    """
    
    val_losses=[]
    train_losses=[]

    for epoch in range(epochs):
        model.train()
        train_loss = training_loop(loss_function, trainLoader, model, optimizer, epoch, device, m)
        train_losses.append(train_loss)
        model.eval()
        val_loss = evaluation_loop(loss_function, valLoader, model, device)
        val_losses.append(val_loss)
        
        if val_loss<=min(val_losses):
            print("saving epoch: " + str(epoch+1))
            print("train loss: " + str(train_loss))
            print("val loss: " + str(val_loss))
            torch.save(model.state_dict(), saveDir)

        scheduler.step(train_loss)
        print(f'Epoch [{epoch + 1}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f} ')
        print(train_losses)
        print(val_losses)


    return train_losses, val_losses

def training_loop(loss_function, loader, model, optimizer, epoch,  device, m):
    """

    Parameters
    ----------
    loss_function : torch.nn Loss function
        Loss function for the training
    loader : torch.utils.data.DataLoader
        Loads the training set
    model : torch.nn class
        Network that should be trained
    optimizer : torch.optim
        Optimizing strategy like stochastic gradient descent
    epoch : Int
        Current epoch
    device : Str
        "cuda" or "cpu" depending on what is used
    m : Int
        Index of the network, that will be used:
            1: ResNet152
            2: DenseNet201
            3: Inception-v3
        For Inception-v3 the images have to be formated to 3x299x299 instead
        of 3x224x224 and the auxiliary network has to be trained

    Returns
    -------
    average_mse_loss : Float
        Loss of current epoch

    """
    total_loss = 0.0
    model.train()
    for data, labels in loader:
        if m==3:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            score1, score2 = model(data).values()
            loss1 = loss_function(score1, labels)
            loss2 = loss_function(score2, labels)
            loss = loss1+0.4*loss2 

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # progress_bar.update(1)
    average_mse_loss = total_loss / len(loader)
    return average_mse_loss



def evaluation_loop(loss_function, eval_loader, val_model, device):
    """
    

    Parameters
    ----------
    loss_function : torch.nn Loss function
        Loss function for the training
    eval_loader : torch.utils.data.DataLoader
        Loads the evaluation set which could be training, validation or 
        test set.
    val_model : torch.nn class
        Network that should be trained
    device : Str
        "cuda" or "cpu" depending on what is used

    Returns
    -------
    average_val_loss : Float
        Loss on evaluation set

    """
    total_val_loss = 0.0
    val_model.eval()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = val_model(images)
            loss = loss_function(outputs, labels)
            total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(eval_loader)
    return average_val_loss

