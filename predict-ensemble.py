import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet152, inception_v3

from preprocess import labels, transform_test_images

input_size = 224
batch_size = 16
num_epochs = 25

num_classes = torch.unique(labels).shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_resnet():
  model = resnet152()
  model.fc = nn.Linear(2048, num_classes)
  model = model.to(device)
  model.load_state_dict(torch.load('./resnet152.torch'))
  model.eval()
  return model

def get_inception():
  model = inception_v3()
  model.AuxLogits.fc = nn.Linear(768, num_classes)
  model.fc = nn.Linear(2048, num_classes)
  model = model.to(device)
  model.load_state_dict(torch.load('./inception.torch'))
  model.eval()
  return model

def predict():
  resnet_model = get_resnet()  
  inception_model = get_inception()
  print('Transforming resnet images')
  resnet_images = transform_test_images(224)
  print('Transforming inception images')
  inception_images = transform_test_images(299)

  n = len(resnet_images)
  resnet_outputs = torch.zeros(n, 37)
  inception_outputs = torch.zeros(n, 37)
  with torch.no_grad():
    for i in range(len(resnet_images)):
      resnet_outputs[i, :] = resnet_model(resnet_images[i].to(device).unsqueeze(0))
      inception_outputs[i, :] = inception_model(inception_images[i].to(device).unsqueeze(0))
  return resnet_outputs, inception_outputs

resnet_outputs, inception_outputs = predict()
