import torch
from torch import nn

from torchvision import datasets, transforms

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

    if dataset == "higgs":
        dat = pd.read_csv(train_path)
        dat.loc[dat.Label == 's', 'Label']=1
        dat.loc[dat.Label == 'b', 'Label']=0
        class Higgs(Dataset):
            def __init__(self, dataframe):
                self.dataframe = dataframe
            def __len__(self):
                return(self.dataframe.shape[0])
            def __getitem__(self, i):
                row = self.dataframe.iloc[i,:]
                inp = row[[x for x in self.dataframe.columns if x not in ['EventId', 'Label']]]
                out = row[['Label']]
                return(np.array(inp).astype('float32'), np.array(out[0]).astype('uint8'))

        trainset = Higgs(dat)
        dat = pd.read_csv(test_path)
        dat.loc[dat.Label == 's', 'Label']=1
        dat.loc[dat.Label == 'b', 'Label']=0
        testset = Higgs(dat)

        
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
                                               drop_last=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, 
                                              drop_last=True, num_workers=4, pin_memory=False)
    return train_loader, test_loader