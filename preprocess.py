import os

import simplejson

from skimage import io
import PIL

import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

index  = simplejson.load(open('./data/train.json'))
labels = torch.LongTensor([int(x['breedID']) for x in index])

def load_training_images():
  IMG = './data/images/train.torch'
  if os.path.isfile(IMG):
    return torch.load(IMG)
  else:
    images = []
    for i in range(len(index)):
      if i % 100 == 0:
        print('%d images loaded.' % i)
      image_name = os.path.join('./data/images/train', str(i) + '.jpg')
      image = io.imread(image_name)
      images.append(torch.tensor(image))
    torch.save(images, IMG)
    return images

images = load_training_images()

def toPIL(tensor):
  return PIL.Image.fromarray(tensor.numpy())

class TrainingDataset(Dataset):
  def __init__(self, input_size):
    self.labels = labels
    transform = transforms.Compose([
      toPIL,
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.images = [transform(x) for x in images]

  def __len__(self):
    return len(self.images)

  def __getitem(self, idx):
    return self.images[idx], self.labels[idx]
