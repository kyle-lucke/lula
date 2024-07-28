import torch
import torch.nn as nn

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from .custom_layers import *

class SmallConvNetSVHN(nn.Module):

  #### Graph Output ####
  # x
  # conv1.net.0
  # conv1.net.1
  # relu
  # bn1
  # conv2.net.0
  # conv2.net.1
  # relu_1
  # bn2
  # maxpool1
  # dropout1
  # conv3.net.0
  # conv3.net.1
  # relu_2
  # bn3
  # conv4.net.0
  # conv4.net.1
  # relu_3
  # bn4
  # maxpool2
  # dropout2
  # conv5.net.0
  # conv5.net.1
  # relu_4
  # bn5
  # conv6.net.0
  # conv6.net.1
  # relu_5
  # bn6
  # maxpool3
  # dropout3
  # flatten
  # fc1
  # relu_6
  # dropout4
  # fc2
  ######################

  # Nodes used in feature extractor for white box probes:
  return_nodes = {'relu': 'x_1', 'maxpool1': 'x_2', 'maxpool2': 'x_3', 'maxpool3': 'x_4',
                  'flatten': 'x_5', 'fc2': 'y_hat'}

  return_nodes_bb = {'fc2': 'y_hat'}
  
  # print('FIXME: using modified return nodes')
  # return_nodes = {'relu_3': 'relu_3', 'maxpool2': 'maxpool2', 'relu_4': 'relu_4', 'relu_5': 'relu_5', 'maxpool3': 'maxpool3', 'flatten': 'flatten', 'relu_6': 'relu_6', 'fc2': 'fc2'}
  
  def __init__(self, feature_extractor=False):
    super().__init__()

    self.feature_extractor = feature_extractor
    
    feature_dim = 512

    self.conv1 = Conv2dSame(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = Conv2dSame(32, 32, 3)
    self.bn2 = nn.BatchNorm2d(32)
    self.maxpool1 = nn.MaxPool2d(2)
    self.dropout1 = nn.Dropout(0.3)

    self.conv3 = Conv2dSame(32, 64, 3)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = Conv2dSame(64, 64, 3)
    self.bn4 = nn.BatchNorm2d(64)
    self.maxpool2 = nn.MaxPool2d(2)
    self.dropout2 = nn.Dropout(0.3)

    self.conv5 = Conv2dSame(64, 128, 3)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = Conv2dSame(128, 128, 3)
    self.bn6 = nn.BatchNorm2d(128)
    self.maxpool3 = nn.MaxPool2d(2)
    self.dropout3 = nn.Dropout(0.3)
    self.flatten = nn.Flatten()

    self.clf = nn.Sequential(      
      nn.Linear(2048, feature_dim),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(feature_dim, 10)
    )
    
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.features(x) if self.feature_extractor else self.clf(self.features_before_clf(x))

  def features(self, x):
    x = self.features_before_clf(x)
    for m in list(self.clf.children())[:-1]:
      x = m(x)
    return x

  
  def features_before_clf(self, x):

    out = self.relu(self.conv1(x))
    out = self.bn1(out)
    out = self.relu(self.conv2(out))
    out = self.bn2(out)
    out = self.maxpool1(out)
    out = self.dropout1(out)

    out = self.relu(self.conv3(out))
    out = self.bn3(out)
    out = self.relu(self.conv4(out))
    out = self.bn4(out)
    out = self.maxpool2(out)
    out = self.dropout2(out)

    out = self.relu(self.conv5(out))
    out = self.bn5(out)
    out = self.relu(self.conv6(out))
    out = self.bn6(out)
    out = self.maxpool3(out)
    out = self.dropout3(out)
    out = self.flatten(out)
    
    return out
  
if __name__ == '__main__':

  torch.manual_seed(0)
  r_data = torch.rand(128, 3, 32, 32)

  model = SmallConvNetSVHN()
  model.eval()
  
  res1 = model(r_data)
  res2 = model(r_data)
  
  train_nodes, eval_nodes = get_graph_node_names(model)

  print('model:')
  print(model)
  print()
  
  print(res2.shape)
  print()
  
  print(torch.mean(res1 - res2))
  print()

  print(model.return_nodes)
  print()
  
  for t in eval_nodes:
    print(t)
