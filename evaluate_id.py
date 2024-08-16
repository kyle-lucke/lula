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
import json

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, auc

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

N_SAMPLES = 20

def aupr(labels, scores):
    precision, recall, _  = precision_recall_curve(labels, scores)
    return auc(recall, precision)

def predict_confidence(dl, model):

    if N_SAMPLES < 20:
    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!! FIXME !!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!! only using 2 samples !!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    py, misclf_labels_correct, misclf_labels_error = lutil.predict_misclf(dl, model,
                                                                          n_samples=N_SAMPLES,
                                                                          return_targets=True)

    py = evalutil.get_confidence(py.cpu().numpy())
    
    return py, misclf_labels_correct.cpu().numpy(), misclf_labels_error.cpu().numpy()

# also known as recall or true positive rate (TPR)
def sensitivity(tp, fn):
    return tp / (tp + fn)

# Also known as selectivity, or true negative rate (TNR)
def specificity(tn, fp):
    return tn / (tn + fp)

# beta > 1 gives more weight to specificity, while beta < 1 favors
# sensitivity. For example, beta = 2 makes specificity twice as important as
# sensitivity, while beta = 0.5 does the opposite.
def f_score_sens_spec(sens, spec, beta=1.0):

    # return (1 + beta**2) * ( (precision * recall) / ( (beta**2 * precision) + recall ) )

    return (1 + beta**2) * ( (sens * spec) / ( (beta**2 * sens) + spec ) )

def threshold_scores(preds, tau):
    if isinstance(preds, np.ndarray):
        return np.where(preds>tau, 1.0, 0.0)

    elif torch.is_tensor(preds):
        return torch.where(preds>tau, 1.0, 0.0)

    else:
        raise TypeError(f"ERROR: preds is expected to be of type (torch.tensor, numpy.ndarray) but is type {type(preds)}")


def determine_threshold(val_loader, model, max_threshold_step=.01):

    model.eval()

    model.to(device)

    meta_preds, misclf_labels, _ = predict_confidence(val_loader, model)

    # determine how many elements we need for a pre-determined spacing
    # between thresholds. taken from:
    # https://stackoverflow.com/a/70230433
    num = round((meta_preds.max() - meta_preds.min()) / max_threshold_step) + 1 
    thresholds = np.linspace(meta_preds.min(), meta_preds.max(), num, endpoint=True)

    # compute performance over thresholds
    threshold_to_metric = {}
    for tau in thresholds:

        predicted_labels = threshold_scores(meta_preds, tau)

        tn, fp, fn, tp = confusion_matrix(misclf_labels, predicted_labels).ravel()
        
        specificity_value = specificity(tn, fp)
        sensitivity_value = sensitivity(tp, fn)

        f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                             specificity_value, beta=1.0)

        print(f'tau: {tau:.6f}, spec: {specificity_value:.4f}, sens: {sensitivity_value:.4f}, f_beta: {f_beta_spec_sens:.4f}, balance: {abs(specificity_value-sensitivity_value):.4f}')
        
        threshold_to_metric[tau] = f_beta_spec_sens

    # determine best threshold:
    best_item = max(threshold_to_metric.items(), key=lambda x: x[1])

    best_tau, best_metric = best_item
    print(f'best | tau: {best_tau:.6f}, metric: {best_metric:.4f}', end='\n\n')

    return best_item[0]

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
        lula_params = torch.load(f'./pretrained_models/kfla/{dataset}{modifier}lula_best.pt',
                                 map_location=device)

        if args.ood_dset == 'best':
            state_dict, n_units, noise = lula_params
            print(f'LULA uses this OOD dataset: {noise}')
        else:
            state_dict, n_units = lula_params

        model = lula.model.LULAModel_LastLayer(base_model, n_units).to(device)
        
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

    assert torch.all(misclf_labels_correct_b == ~misclf_labels_error_b)
        
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
parser.add_argument('--output-dir', '-o')

args = parser.parse_args()

args_dict = vars(args)

for k, v in args_dict.items():
    print(f'{k}: {v}')
print()

os.makedirs(args.output_dir, exist_ok=True)

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
threshold = determine_threshold(misclf_valid_dl, lula_model) 

# print(threhsold)

# compute metrics:

metrics_dict = dict()

test_preds, misclf_labels_correct_test, misclf_labels_error_test = predict_confidence(misclf_test_dl, lula_model)

predicted_labels_test = threshold_scores(test_preds, threshold)

tn, fp, fn, tp = confusion_matrix(misclf_labels_correct_test,
                                  predicted_labels_test).ravel()

specificity_value = specificity(tn, fp)
sensitivity_value = sensitivity(tp, fn)
        
f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                     specificity_value, beta=1.0)

metrics_dict['specificity'] = specificity_value
metrics_dict['sensitivity'] = sensitivity_value

for beta in [1.0, 2.0]:

    metrics_dict[f'f_beta_spec_sens@{beta}'] = f_score_sens_spec(sensitivity_value,
                                                                 specificity_value, beta=beta)

metrics_dict['aupr_success'] = aupr(misclf_labels_correct_test, test_preds)
metrics_dict['aupr_error'] = aupr(misclf_labels_error_test, -test_preds)

metrics_dict['ap_success'] = average_precision_score(misclf_labels_correct_test, test_preds)
metrics_dict['ap_error'] = average_precision_score(misclf_labels_error_test, -test_preds)

metrics_dict['roc_auc'] = roc_auc_score(misclf_labels_correct_test, test_preds)

for k, v in metrics_dict.items():
    print(f'{k}: {v}')
print()

json.dump(metrics_dict, open(os.path.join(args.output_dir, 'metrics.json'), 'w'))
