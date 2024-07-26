import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time, argparse


class Backprop:

        
    def __init__(self, model, loss = "CE", optimizer = None, num_classes = 10, **kwargs):

        self.model = model
        self.loss =loss
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)
        self.opt = optimizer
        self.num_classes = num_classes
        if self.loss == "mse": self.output_criterion = nn.MSELoss()#y_pred, labels_float)
        elif self.loss == "CE": self.output_criterion = nn.CrossEntropyLoss()#y_pred, label)
        
    def step(self, input_data, labels, **kwargs):
        self.opt.zero_grad()
        if 'LBFGS' in type(self.opt).__name__:
            c = kwargs['closure']
            self.opt.step(c)
            return(c())
        
        labels_float = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        y_pred, hidden_zs = self.model(input_data)

        if self.loss == "mse": 
            l = self.output_criterion(y_pred, labels_float)
        elif self.loss == "CE": 
            l = self.output_criterion(y_pred, labels)
        
        l.backward()
        self.opt.step()
        return(l)


