import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
import pdb

def get_filename(args):
    if args.BP == 1:
        filename = "{}_{}_{}".format(args.dataset, args.model, args.loss)
    else:
        filename = "{}_{}_{}_F{}_{}{}{}_S{}L{}".format(args.dataset, args.model, args.loss, args.forward, args.kernel_x[0], args.kernel_h[0], args.kernel_y[0], args.sigma_, args.lambda_, )
    if args.bn_affine == 1:
        filename += "_bn"
    if args.Latinb == 1:
        try:
            filename += "_Latinb{}{}".format(args.Latinb_type, args.Latinb_lambda)
        except:
            filename += "_Latinb{}".format(args.Latinb_lambda)
    return filename

def show_result(model, train_loader, test_loader, epoch, logs, device):
    model.model.eval()
    with torch.no_grad():
        counts, correct, counts2, correct2 = 0, 0, 0, 0        
        
        for batch_idx, (data, target) in enumerate(train_loader): 
            #if len(data.shape) > 3: # Channel dimension exists
            #    data = torch.mean(data, axis = 1).squeeze()
            target = target.squeeze()
            output = model.model.forward(data.view(data.size(0), -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True).cpu()
            cor = (pred[:,0] == target).float().sum()
            if cor > len(pred):
                print("Warning, for batch ", batch_idx)
                print("Correct value exceeds count, skipping...")
                pdb.set_trace()
                continue
            correct += cor
            counts += len(pred)
            
        
        for batch_idx, (data, target) in enumerate(test_loader): 
            target = target.squeeze()
            output = model.model.forward(data.view(data.size(0), -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True).cpu()
            correct2 += (pred[:,0] == target).float().sum()
            counts2 += len(pred)
            
        train_acc = correct/counts
        test_acc = correct2/counts2
        print("EPOCH {}. \t Training  ACC: {:.4f}. \t Testing ACC: {:.4f}".format(epoch, train_acc, test_acc))
        logs.append([epoch, train_acc.numpy(), test_acc.numpy()])



@torch.no_grad()
def measure_number_dead_neurons(net, train_loader):
    """Function to measure the number of dead neurons in a trained neural network.

    For each neuron, we create a boolean variable initially set to 1. If it has an activation unequals 0 at any time, we
    set this variable to 0. After running through the whole training set, only dead neurons will have a 1.

    Reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/02-activation-functions.html
    """
    neurons_dead = [
        torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
        for layer in net.layers[:-1]
        #if isinstance(layer, nn.Linear)
    ]  # Same shapes as hidden size in BaseNetwork

    net.eval()
    for imgs, labels in tqdm(train_loader, leave=False):  # Run through whole training set
        layer_index = 0
        imgs = imgs.to(device)
        imgs = imgs.view(imgs.size(0), -1)
        for layer in net.layers[:-1]:
            imgs = layer(imgs)
            if isinstance(layer, ActivationFunction):
                # Are all activations == 0 in the batch, and we did not record the opposite in the last batches?
                neurons_dead[layer_index] = torch.logical_and(neurons_dead[layer_index], (imgs == 0).all(dim=0))
                layer_index += 1
    number_neurons_dead = [t.sum().item() for t in neurons_dead]
    print("Number of dead neurons:", number_neurons_dead)
    print(
        "In percentage:",
        ", ".join(
            [f"{(100.0 * num_dead / tens.shape[0]):4.2f}%" for tens, num_dead in zip(neurons_dead, number_neurons_dead)]
        ),
    )