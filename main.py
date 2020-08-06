import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import syft as sy

import settings
from models.cnn import Cnn as Model
from train import train
from test import test

# Load arguments
config = settings.DefaultConfig()
bob = config.workers[0]
alice = config.workers[1]

# Cuda setup
use_cuda = not config.no_cuda and torch.cuda.is_available()
torch.manual_seed(config.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# Worker setup
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Federated dataset
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))
        .federate((bob, alice)),
    batch_size=config.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('datasets/', train=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])),
    batch_size=config.test_batch_size, shuffle=True, **kwargs)

if __name__ == '__main__':
    # initialize the model
    model = Model().to(device)
    # optimize
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train(config, model, device, federated_train_loader, optimizer)
    # fit
        #test(config, model, device, test_loader)
    # save
    if (config.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")