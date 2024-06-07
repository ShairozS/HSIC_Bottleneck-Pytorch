import os
from Code.Trainers import HSICBottleneck
from Code.Models import MLP, ChebyKAN, KAN
from Code.Data import load_data
from Code.Utils import show_result
import time
import torch; torch.manual_seed(1)
from torch import optim
import pandas as pd


batchsize = 128
train_loader, test_loader = load_data(dataset = 'mnist', batchsize=batchsize)
epochs = 100

### MLP Small (2-layer) Architectures ###


## Experiment 1
experiment_name = "MNIST_hsic_mlp_2layers"
device = "cuda"
layer_sizes = [784, 64]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 2
experiment_name = "MNIST_hsic_mlp_2layers"
device = "cuda"
layer_sizes = [784, 64]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")

## Experiment 3
experiment_name = "MNIST_hsic_mlp_2layers"
device = "cuda"
layer_sizes = [784, 64]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 4
experiment_name = "MNIST_hsic_mlp_2layers"
device = "cuda"
layer_sizes = [784, 64]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")



### MLP Medium (3-layer) Architectures ###

## Experiment 1
experiment_name = "MNIST_hsic_mlp_3layers"
device = "cuda"
layer_sizes = [784, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 2
experiment_name = "MNIST_hsic_mlp_3layers"
device = "cuda"
layer_sizes = [784, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 3
experiment_name = "MNIST_hsic_mlp_3layers"
device = "cuda"
layer_sizes = [784, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 4
experiment_name = "MNIST_hsic_mlp_3layers"
device = "cuda"
layer_sizes = [784, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")




### MLP Large (6-layer) Architectures ###


## Experiment 1
experiment_name = "MNIST_hsic_mlp_6layers"
device = "cuda"
layer_sizes = [784, 128, 64, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 2
experiment_name = "MNIST_hsic_mlp_6layers"
device = "cuda"
layer_sizes = [784, 128, 64, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.AdamW(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 3
experiment_name = "MNIST_hsic_mlp_6layers"
device = "cuda"
layer_sizes = [784, 128, 64, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")


## Experiment 4
experiment_name = "MNIST_hsic_mlp_6layers"
device = "cuda"
layer_sizes = [784, 32, 16]
model = MLP(layer_sizes = layer_sizes, output_size = 10).to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); print("Model trainable parameters: ", num_parameters)
lr = 0.0005
optimizer = optim.SGD(model.parameters(), lr=lr)
trainer = HSICBottleneck(model = model, optimizer = optimizer)
logs = list()
for epoch in range(epochs):
    trainer.model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batchsize, -1)
        trainer.step(data.view(batchsize, -1).to(device), target.to(device))
        trainer.tune_output(data.view(batchsize, -1).to(device), target.to(device))
    end = time.time()
    if epoch % 2 == 0:
        show_result(trainer, train_loader, test_loader, epoch, logs, device)
        logs[epoch//2].append(end-start)
df = pd.DataFrame(logs); df.columns = ['Epoch', 'Train_loss', 'Test_loss', 'Time']; df.head()
experiment_name += "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_parameters_" + str(num_parameters) + "_optimizer_" + str(optimizer).split("(")[0]
df.to_csv(experiment_name + ".csv")
print(experiment_name, "done.")
