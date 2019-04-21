import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet152

from preprocess import TrainingDataset, labels

input_size = 224
batch_size = 16
num_epochs = 25

num_classes = torch.unique(labels).shape[0]

dataset = TrainingDataset(input_size)

len_train_set = int(len(dataset) * 0.7)
len_val_set   = len(dataset) - len_train_set

train_set, val_set = random_split(dataset, [len_train_set, len_val_set])

datasets = {
  'train': train_set,
  'val': val_set
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for some reason the default_collate function causes a problem.
def collate(batch):
  images = torch.stack([x for x, _ in batch], 0)
  labels = torch.stack([k for _, k in batch], 0)
  return images, labels

def train(use_pretrained = False):
  loaders = {
    x: DataLoader(datasets[x],
      batch_size = batch_size, shuffle = True, collate_fn = collate)
    for x in ['train', 'val']
  }

  model = resnet152(use_pretrained)
  model.fc = nn.Linear(2048, num_classes)
  model = model.to(device)

  optimizer = optim.Adam(model.parameters(), 1e-5)

  criterion = nn.CrossEntropyLoss()

  since = time.time()

  val_acc_history = []

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in loaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          _, preds = torch.max(outputs, 1)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(loaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    # deep copy the model
    if phase == 'val' and epoch_acc > best_acc:
      best_acc = epoch_acc
      best_model_wts = copy.deepcopy(model.state_dict())
    if phase == 'val':
      val_acc_history.append(epoch_acc)

      print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  torch.save(model.state_dict(), './resnet152.torch')

train(True)
