import torch
from torch import nn
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
### DATA ### 

#To download MNIST
#!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
#!tar -zxvf MNIST.tar.gz

def load_data(dataset, batchsize = 128, download=False, train_path = None, test_path = None):
    
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if dataset == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(),])
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    if dataset == 'imdb':
        class IMDB(Dataset):
        
            def __init__(self, mode = 'train'):
                if mode == 'train':
                    self.data = np.load('./data/IMDB Dataset_clean_train.npy')
                    print("For mode = " + mode + " dataset shape: ", self.data.shape)
                elif mode == 'test':
                    self.data = np.load('./data/IMDB Dataset_clean_test.npy')
                    print("For mode = " + mode + " dataset shape: ", self.data.shape)
        
            def __len__(self):
                return(self.data.shape[0])
        
            def __getitem__(self, idx):
                d_x = self.data[idx, :100]
                d_y = self.data[idx, -1]
                d_x = torch.Tensor(d_x)
                d_y = torch.Tensor([d_y])
                return(d_x[np.newaxis,:], d_y)

        trainset = IMDB(mode = 'train')
        testset = IMDB(mode = 'test')
        
    if dataset == "higgs":
        
        # Download from: https://archive.ics.uci.edu/dataset/280/higgs
        print(os.getcwd())
        
        cols = ['label', 
                'lepton  pT', 
                'lepton  eta', 
                'lepton  phi', 
                'missing energy magnitude', 
                'missing energy phi', 
                'jet 1 pt', 
                'jet 1 eta', 
                'jet 1 phi', 
                'jet 1 b-tag', 
                'jet 2 pt', 
                'jet 2 eta', 
                'jet 2 phi', 
                'jet 2 b-tag', 
                'jet 3 pt', 
                'jet 3 eta', 
                'jet 3 phi', 
                'jet 3 b-tag', 
                'jet 4 pt', 
                'jet 4 eta', 
                'jet 4 phi', 
                'jet 4 b-tag', 
                'm_jj',
                'm_jjj', 
                'm_lv',
                'm_jlv', 
                'm_bb', 
                'm_wbb', 
                'm_wwbb']
        dat = pd.read_csv(train_path)
        dat.columns = cols
        train_idx = [x for x in range(dat.shape[0])]
        train_idx = [int(x < 500000) for x in train_idx] 
        dat['training'] = train_idx

        class Higgs(Dataset):
            
            def __init__(self, dataframe, training = True):
                self.dataframe = dataframe
                if training:
                    self.dataframe = self.dataframe.loc[self.dataframe.training == 1, :]
                else:
                    self.dataframe = self.dataframe.loc[self.dataframe.training == 0, :]
                    
            def __len__(self):
                return(self.dataframe.shape[0])
                
            def __getitem__(self, i):
                row = self.dataframe.iloc[i,:]
                inp = row[[x for x in self.dataframe.columns if x not in ['label', 'training']]]
                out = row[['label']]
                inp = np.array(inp).astype('float32')
                out = np.array(out).astype('uint8')
                return(inp[np.newaxis,...], out)
                

        trainset = Higgs(dat)
        testset = Higgs(dat, training = False)

        
    if dataset == "cifar":
        transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.RandomAffine(degrees = (0, 360), translate=(0.1, 0.3), scale=(0.5, 0.75)), 
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, 
                                               drop_last=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, 
                                              drop_last=True, num_workers=0, pin_memory=False)
    return train_loader, test_loader