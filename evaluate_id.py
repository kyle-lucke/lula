from collections import OrderedDict

import torch
from torch import distributions as dist
from torch.distributions import Categorical
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torch.utils.data as data
import numpy as np
from math import *

# from models.models import LeNetMadry
# from models import wrn

from models.resnet import resnet32
from models.small_convnet_svhn import SmallConvNetSVHN

from tqdm.auto import tqdm, trange
from util import dataloaders as dl
from util.misc import set_seed
from laplace import kfla
import laplace.util as lutil
import util.evaluation as evalutil

import sys, os
import argparse
import traceback

import lula.model
import lula.train

from model_utils import *


def remap_sd(old_sd, layer_mapping):
    # re-map keys in state dict:
    new_sd = OrderedDict()
    for old_layer, parameter in old_sd.items():

        if old_layer in layer_mapping:
            
            new_layer = layer_mapping[old_layer]
            
            new_sd[new_layer] = parameter

        else:
            new_sd[old_layer] = parameter

    return new_sd

def load_base_model(dataset):
    if dataset == 'SVHN':
        model = SmallConvNetSVHN()
            
        layer_mapping = {'fc1.weight': 'clf.0.weight' ,
                         'fc1.bias': 'clf.0.bias',
                         'fc2.weight': 'clf.3.weight',
                         'fc2.bias': 'clf.3.bias'}
        
        old_sd = torch.load('trained-base-models/svhn-cnn/best.pth', map_location=device)

        new_sd = remap_sd(old_sd, layer_mapping)

        model.load_state_dict(new_sd)
        
    elif dataset == 'CIFAR10':
        model = resnet32(10)

        old_sd = torch.load('trained-base-models/cifar10-resnet32/best.pth', map_location=device)

        layer_mapping = {'linear1.weight': 'clf.0.weight' ,
                         'linear1.bias': 'clf.0.bias',
                         'linear2.weight': 'clf.2.weight',
                         'linear2.bias': 'clf.2.bias'}

        new_sd = remap_sd(old_sd, layer_mapping)
        
        model.load_state_dict(new_sd)
        
    model.to(device)
    model.eval()
    
    return model

def load_lula_model(dataset, type='LULA'):

    base_model = load_base_model(dataset)

    # Additionally, load these for LULA
    if type == 'LULA':
        print('LOADING LULA')
        lula_params = torch.load(f'./pretrained_models/kfla/{dataset}{modifier}lula_best.pt')

        if args.ood_dset == 'best':
            state_dict, n_units, noise = lula_params
            print(f'LULA uses this OOD dataset: {noise}')
        else:
            state_dict, n_units = lula_params

        model = lula.model.LULAModel_LastLayer(base_model, n_units).cuda()
        model.to_gpu()
        model.load_state_dict(state_dict)
        model.disable_grad_mask()
        model.unmask()
        model.eval()

    if type in ['LA', 'LULA']:
        training_weight_decay = 1e-4
        var0 = torch.tensor(1/(training_weight_decay*len(train_loader.dataset))).float().to(device)
        model = kfla.KFLA(model)
        model.get_hessian(train_loader)
        model.estimate_variance(var0)

    return model

def generate_misclf_ds(base_model, dl):

  misclf_labels_correct = []
  misclf_labels_error = []

  inputs = []
    
  for inps, labels in dl:
    inps, labels = inps.to(device), labels.to(device)

    predicted_labels = torch.argmax(base_model(inps), -1)
  
    misclf_labels_correct_b = predicted_labels == labels
    misclf_labels_error_b = predicted_labels != labels
    
    misclf_labels_correct.append(misclf_labels_correct_b)
    misclf_labels_error.append(misclf_labels_error_b)
    inputs.append(inps)

  # debug:
  print('acc:', torch.mean(torch.concat(misclf_labels_correct).float()))
  
  ds_out = data.TensorDataset(torch.concat(inputs),
                          torch.concat(misclf_labels_correct).int(),
                              torch.concat(misclf_labels_error).int())
  
  dl_out = data.DataLoader(ds_out, batch_size=args.batch_size, num_workers=8)
  
  return dl_out 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100'])
parser.add_argument('--lenet', default=False, action='store_true')
parser.add_argument('--base', default='plain', choices=['plain', 'oe'])
parser.add_argument('--timing', default=False, action='store_true')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ood_dset', default='best',
                    choices=['imagenet', 'uniform', 'smooth', 'best'])

args = parser.parse_args()

args_dict = vars(args)

for k, v in args_dict.items():
    print(f'{k}: {v}')
print()
    
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 10
num_channel = 3

if args.dataset == 'CIFAR10':
    print('CIFAR10')

    _, transform_test = get_preprocessor(args.dataset)
    
    dataset = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=True,
                               transform=transform_test)
    
    dataset_val = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=False,
                                   transform=transform_test)

    testset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=False,
                               transform=transform_test)

    # load indices used during our training
    train_meta_idxs = np.load('trained-base-models/cifar10-resnet32/trainset_meta_idxs.npy')
    val_idxs = np.load('trained-base-models/cifar10-resnet32/val_idxs.npy')

    trainset = data.Subset(dataset, train_meta_idxs)
    valset = data.Subset(dataset_val, val_idxs)

    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=8)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             num_workers=8)

    meanstd = [ [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] ]
    
    
elif args.dataset == 'SVHN':

    _ , transform_test = get_preprocessor(args.dataset)
    
    dataset = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                            transform=transform_test)
    
    dataset_val = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                                transform=transform_test)
    
    # load indices used during our training
    train_meta_idxs = np.load('trained-base-models/svhn-cnn/trainset_meta_idxs.npy')
    val_idxs = np.load('trained-base-models/svhn-cnn/val_idxs.npy')

    trainset = data.Subset(dataset, train_meta_idxs)
    valset = data.Subset(dataset, val_idxs)

    testset = datasets.SVHN(root='~/data/SVHN', split='test', download=True,
                            transform=transform_test)

    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             num_workers=8)

    meanstd = [ [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] ]


if args.dataset == 'SVHN':
    modifier = '_cnn_'
elif args.dataset == 'CIFAR10':
    modifier = '_resnet32_'
    
base_model = load_base_model(args.dataset)
lula_model = load_lula_model(args.dataset)

print('validation:')
misclf_valid_dl = generate_misclf_ds(base_model, val_loader)
print()

print('test:')
misclf_test_dl = generate_misclf_ds(base_model, test_loader)
print()

# for predicitons use: lutil.predict_misclf(, return_targets=True)
# Then use: evalutil.get_confidence(lula_preds)

# determine threhsold
# threshold = determine_threhsold(lula_model, misclf_valid_dl) 

# print(threhsold)

# compute metrics:

