
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
    
    batchsize = 4096
    train_loader, test_loader = load_data(dataset = 'mnist', batchsize=batchsize)
    epochs = 100
    loss = "CE" #"mse"
    device = "cuda"
    dropout = 0
    degree = 3
    init =  torch.nn.init.kaiming_uniform_
    trainer_ = Backprop
    for model_name in ['mlp', 'kan']:
        
        for init in [torch.nn.init.kaiming_normal_, torch.nn.init.kaiming_uniform_]:
            
            for layer_sizes in [[784, 32], [784, 64, 32], [784, 128, 64, 32]]:
                
                if model_name == 'mlp':
                    model = MLP(layer_sizes = layer_sizes, output_size = 10, dropout = dropout, init = init)
                elif model_name == 'kan':
                    model = KAN(layer_sizes = layer_sizes, output_size = 10, dropout = dropout, init = init)
                
                model = model.to(device)
                    
                num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
                print("--------------------------------------------------------------------")
                
                for lr in [0.05, 0.005, 0.0005]:
        
                    for o in ["Adam", "SGD", "SGDM"]:
                        experiment_name = "MNIST_backprop_" + model_name + "_" + str(len(layer_sizes)) + "layers"
                        if 'cuda' in device:
                            assert next(model.parameters()).is_cuda
                        
                        if o == "Adam":
                            optimizer = optim.AdamW(model.parameters(), lr=lr) 
                        elif o=="SGD":
                            optimizer = optim.SGD(model.parameters(), lr=lr)
                        elif o=="SGDM":
                            optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, nesterov = True)
                            
                        trainer = trainer_(model = model, optimizer = optimizer, num_classes = 10, batchsize = batchsize)
                        logs = list()
                        
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
                        experiment_name += "_batchsize_" + str(batchsize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + \
                                            str(num_parameters) + "_optimizer_" + o + "_init_" + init.__name__
                        try:
                            df.to_csv(os.path.join("./Results", experiment_name + ".csv"))
                            print(experiment_name, "done.")
                        except FileNotFoundError:
                            breakpoint()