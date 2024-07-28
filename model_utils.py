import torch
import torchvision.transforms as transforms

def get_preprocessor(dataset):
  if dataset == 'MNIST':

    normalize = transforms.Normalize(mean=[0.1307,],
                                     std=[0.3081,])

    transform_train = transforms.Compose([
      transforms.ToTensor(),
      normalize])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
    
  elif dataset == 'SVHN':

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose([
      transforms.ToTensor(),
      normalize])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

    
  elif dataset == 'CIFAR10':

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),    
      transforms.ToTensor(),
      normalize])

    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
    
  elif dataset == 'CIFAR100':

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      normalize])

    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

  else:
    raise Exception('ERROR: unrecognized model name.')
    
  return transform_train, transform_test
