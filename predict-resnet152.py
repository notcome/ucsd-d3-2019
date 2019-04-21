import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet152

from preprocess import labels, transform_test_images

input_size = 224
batch_size = 16
num_epochs = 25

num_classes = torch.unique(labels).shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict():
  model = resnet152()
  model.fc = nn.Linear(2048, num_classes)
  model = model.to(device)
  model.load_state_dict(torch.load('./resnet152.torch'))
  model.eval()

  images = transform_test_images(input_size)
  preds = []
  with torch.no_grad():
    for image in images:
      image = image.to(device)
      output = model(image.unsqueeze(0))
      pred = torch.argmax(output)
      preds.append(pred)
  return preds

results = predict()
