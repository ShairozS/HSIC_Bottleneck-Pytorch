
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
    
    batchsize = 512
    train_loader, test_loader = load_data(dataset = 'cifar', batchsize=batchsize)
    epochs = 100
    loss = "CE" #"mse"
    device = "cuda"
    dropout = 0.2
    degree = 3
    trainer_ = HSICBottleneck
    batchnorm = 1
    wide = 1
    activation = nn.GELU()

    # Entropy is the number of states
    # Entropy increases when information is lost
    # With time, entropy decreases
    # Increasing entropy requires information
    # Information requires energy
    
    for model_name in ['mlp', 'kan']:

        #wide = 0 if 'mlp' not in model_name else wide

        
        for init in [torch.nn.init.kaiming_uniform_]:

            for layer_sizes in [[32*32*3, 512, 256], [32*32*3, 1024, 512, 256]]:
            #for layer_sizes in [[32*32*3, 256], [32*32*3, 512, 256], [32*32*3, 1024, 512, 256]]:
                if wide and model_name == 'mlp':
                    layer_sizes = [10*x if x!=32*32*3 else x for x in layer_sizes]
                elif wide and model_name == 'kan':
                    continue
                if model_name == 'mlp':
                    model = MLP(layer_sizes = layer_sizes, output_size = 10, dropout = dropout, init = init, batchnorm = batchnorm)
                elif model_name == 'kan':
                    model = KAN(layer_sizes = layer_sizes, output_size = 10, dropout = dropout, init = init, batchnorm = batchnorm)
                
                model = model.to(device)
                    
                print("Layer sizes: ", layer_sizes)
                num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
                print("--------------------------------------------------------------------")
                
                for lr in [0.0005]:
        
                    for o in ["Adam", "SGD"]:

                        if wide:
                            model_name = "mlpWide"
                        
                        experiment_name = "CIFAR_hsic_" + model_name + "_" + str(len(layer_sizes)) + "layers"
                        
                        if 'cuda' in device:
                            assert next(model.parameters()).is_cuda
                        
                        if o == "Adam":
                            optimizer = optim.AdamW(model.parameters(), lr=lr) 
                        elif o=="SGD":
                            optimizer = optim.SGD(model.parameters(), lr=lr)
                        elif o=="SGDM":
                            optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, nesterov = True)
                        elif o=='LBFGS':
                            optimizer = optim.LBFGS(model.parameters())
                        else:
                            assert False, "optimizer not recognized"
                            
                        trainer = trainer_(model = model, optimizer = optimizer, num_classes = 10, batchsize = batchsize)
                        logs = list()
                        
                        for epoch in range(epochs):
                            trainer.model.train()
                            start = time.time()
                            for batch_idx, (data, target) in enumerate(train_loader):

                                data = data.view(batchsize, -1)
                                if o=="LBFGS":
                                    def closure():
                                        optimizer.zero_grad()
                                        output = model(data.view(batchsize, -1).to(device))[0]
                                        loss = nn.CrossEntropyLoss()(output, target.to(device))
                                        loss.backward()
                                        return loss
                                    trainer.step(data.view(batchsize, -1).to(device), target.to(device), closure=closure)
                                    continue
                                    
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
                        experiment_name += "_batchsize_" + str(batchsize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + \
                                            str(num_parameters) + "_optimizer_" + o + "_init_" + init.__name__
                        try:
                            df.to_csv(os.path.join("./Results", experiment_name + ".csv"))
                            print(experiment_name, "done.")
                        except FileNotFoundError:
                            breakpoint()