from collections import OrderedDict

import torch
from torch import distributions as dist
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
from util.evaluation import timing, predict
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100'])
parser.add_argument('--lenet', default=False, action='store_true')
parser.add_argument('--base', default='plain', choices=['plain', 'oe'])
parser.add_argument('--timing', default=False, action='store_true')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

args_dict = vars(args)

for k, v in args_dict.items():
    print(f'{k}: {v}')
print()
    
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = './pretrained_models/kfla'
path += '/oe' if args.base == 'oe' else ''
if not os.path.exists(path):
    os.makedirs(path)

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 10
num_channel = 3

if args.dataset == 'SVHN':
    modifier = '_cnn_'
elif args.dataset == 'CIFAR10':
    modifier = '_resnet32_'

def load_model():
    if args.dataset == 'SVHN':
        model = SmallConvNetSVHN()
            
        layer_mapping = {'fc1.weight': 'clf.0.weight' ,
                         'fc1.bias': 'clf.0.bias',
                         'fc2.weight': 'clf.3.weight',
                         'fc2.bias': 'clf.3.bias'}
        
        old_sd = torch.load('trained-base-models/svhn-cnn/best.pth', map_location=device)

        new_sd = remap_sd(old_sd, layer_mapping)

        model.load_state_dict(new_sd)
        
    elif args.dataset == 'CIFAR10':
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

print()


####################################################################################################
## LULA TRAINING
####################################################################################################

training_weight_decay = 1e-4
    
# Prior variance comes from the weight decay used in the MAP training
var0 = torch.tensor(1/(training_weight_decay*len(train_loader.dataset))).float().to(device)
print(f'var0: {var0.item():.4f}')

# Grid search
# Smooth Noise already attain maximum entropy in MNIST, so it's not useful
# noise_grid = ['imagenet', 'smooth']
noise_grid = ['imagenet']
n_units_grid = [512] if args.timing else [32, 64, 128, 256, 512, 1024]

lr = 0.1
nll = nn.CrossEntropyLoss(reduction='mean')

best_model = None
best_loss = inf

for noise in noise_grid:
    print(noise)
    print()

    if noise == 'imagenet':
        ood_train_loader = dl.ImageNet32(dataset=args.dataset, train=True, augm_flag=False, meanstd=meanstd)

        ood_val_loader = dl.ImageNet32(dataset=args.dataset, train=False, augm_flag=False, meanstd=meanstd)
        
    elif noise == 'smooth':
        ood_train_loader = dl.Noise(args.dataset, train=True)
        ood_val_loader = dl.Noise(args.dataset, train=False)
    else:
        ood_train_loader = dl.UniformNoise(args.dataset, train=True, size=len(train_loader.dataset))
        ood_val_loader = dl.UniformNoise(args.dataset, train=False, size=2000)
    
    best_model_noise = None
    best_loss_noise = inf

    for n_unit in n_units_grid:
        print(n_unit)
        n_lula_units = [n_unit]

        model = load_model()
        model_lula = lula.model.LULAModel_LastLayer(model, n_lula_units).to(device)
        # model_lula.to_gpu()
        model_lula.eval()
        model_lula.enable_grad_mask()

        try:
            ood_train_loader.dataset.offset = np.random.randint(len(ood_train_loader.dataset))
            model_lula = lula.train.train_last_layer(
                    model_lula, nll, val_loader, ood_train_loader, 1/var0, lr=lr,
                    n_iter=10, progressbar=True, mc_samples=10
                )

            # if args.timing:
                #print(f'Time Construction: {time_cons:.3f}, Time Training: {time_train:.3f}')
                #sys.exit(0)

            # Temp
            torch.save(model_lula.state_dict(), f'{path}/{args.dataset}_lula_temp.pt')

            # Grid search criterion (this modifies model_lula, hence the need of the temp above)
            # MMC distance to the optimal for both in- and out-dist val set, under a Laplace
            model_lula.disable_grad_mask()
            model_lula.eval()
            model_lula.unmask()

            # Do a LA over the trained LULA-augmented network
            model_kfla = kfla.KFLA(model_lula)
            model_kfla.get_hessian(train_loader)
            model_kfla.estimate_variance(var0)
            py_in = lutil.predict(val_loader, model_kfla, n_samples=10)
            py_out = lutil.predict(ood_val_loader, model_kfla, n_samples=10, n_data=2000)

            h_in = dist.Categorical(py_in).entropy().mean().cpu().numpy()
            h_out = dist.Categorical(py_out).entropy().mean().cpu().numpy()
            loss = h_in - h_out

            print(f'Loss: {loss:.3f}, H_in: {h_in:.3f}, H_out: {h_out:.3f}')
            print(best_loss_noise)

            # Save the current best
            if loss < best_loss_noise:
                state_dict = torch.load(f'{path}/{args.dataset}_lula_temp.pt')
                torch.save([state_dict, n_lula_units], f'{path}/{args.dataset}{modifier}lula_{noise}.pt')
                best_loss_noise = loss
        except Exception as e:
            print(f'Exception occured: {e}')
            traceback.print_tb(e.__traceback__)
            loss = inf

        print()

    print()

    # Save the current best across noises and n_units
    if best_loss_noise < best_loss:
        state_dict, n_lula_units = torch.load(f'{path}/{args.dataset}{modifier}lula_{noise}.pt')
        torch.save([state_dict, n_lula_units, noise], f'{path}/{args.dataset}{modifier}lula_best.pt')
        best_loss = best_loss_noise

# Cleanup
os.remove(f'{path}/{args.dataset}_lula_temp.pt')


####################################################################################################
## Test the best model
####################################################################################################

model = load_model()

state_dict, n_lula_units, noise = torch.load(f'{path}/{args.dataset}{modifier}lula_best.pt')
model_lula = lula.model.LULAModel_LastLayer(model, n_lula_units).to(device)

# model_lula.to_gpu()

model_lula.load_state_dict(state_dict)
model_lula.disable_grad_mask()
model_lula.eval()

# Test
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
py_in = predict(test_loader, model_lula).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = np.max(py_in).mean()
print(f'[In, LULA-{noise}] Accuracy: {acc_in:.3f}; MMC: {mmc:.3f}')
