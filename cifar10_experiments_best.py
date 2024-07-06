
if __name__ == '__main__':
    import os
    from Code.Trainers import HSICBottleneck, Backprop
    from Code.Models import MLP, ChebyKAN, KAN
    from Code.Data import load_data
    from Code.Utils import show_result
    import time
    import torch; torch.manual_seed(1)
    from torch import optim
    import pandas as pd
    from torch import nn
    
    batchsize = 2048
    epochs = 100
    loss = "CE" #"mse"
    device = "cuda"
    dropout = 0.2
    degree = 3
    trainer_ = Backprop
    batchnorm = 1
    wide = 0
    init = torch.nn.init.orthogonal_
    lr = 0.0005
    
    # Entropy is the number of states
    # Entropy increases when information is lost
    # With time, entropy decreases
    # Increasing entropy requires information
    # Information requires energy
    
    for dataset in ['mnist', 'cifar', 'fashion_mnist']:
        
        train_loader, test_loader = load_data(dataset = dataset, batchsize=batchsize)

        act = nn.SiLU
        model_name = 'kan'
        
        for degree in [2,3,4,5]:
        #for act in [nn.SiLU, nn.GELU, nn.ELU]:
        
                if dataset in ['mnist', 'fashion_mnist']:
                    layer_sizes = [784, 128, 64, 32]
                else:
                    layer_sizes = [32*32*3, 1024, 512, 256]


                print("Dataset: ", dataset)
                print("Layer sizes: ", layer_sizes)
                print("Degree: ", degree)
                model = KAN(layer_sizes = layer_sizes, output_size = 10, dropout = dropout, init = init, batchnorm = batchnorm, activation = act, degree = degree)
                
                model = model.to(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr) 

                num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
                print("--------------------------------------------------------------------")
       
                
                experiment_name = dataset.upper() + "_backprop_" + model_name + "_" + str(len(layer_sizes)) + "layers" + "_activation_" + act.__name__ + "_degree_" + str(degree)
                
                if 'cuda' in device:
                    assert next(model.parameters()).is_cuda
  
                    
                trainer = trainer_(model = model, optimizer = optimizer, num_classes = 10, batchsize = batchsize)
                logs = list()
            
                experiment_name += "_batchsize_" + str(batchsize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + \
                                    str(num_parameters) + "_optimizer_" + type(optimizer).__name__ + "_init_" + init.__name__
                for epoch in range(epochs):
                    trainer.model.train()
                    start = time.time()
                    for batch_idx, (data, target) in enumerate(train_loader):

                        data = data.view(batchsize, -1)
                        
                        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
                        try:
                            trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
                        except AttributeError:
                            pass
                    end = time.time()
                    if epoch % 2 == 0:
                        show_result(trainer, train_loader, test_loader, epoch, logs, device)
                        logs[epoch//2].append(end-start)
                
                df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
                
                try:
                    df.to_csv(os.path.join("./Results2", experiment_name + ".csv"))
                    print(experiment_name, "done.")
                except FileNotFoundError:
                    breakpoint()