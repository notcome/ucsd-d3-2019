import os

import simplejson

from skimage import io
import PIL

import torch

from torch.utils.data import Dataset
from torchvision import transforms

index  = simplejson.load(open('./data/train.json'))
# shift to make it zero-indexed
labels = torch.LongTensor([int(x['breedID']) for x in index]) - 1

def load_images(phase, index_range):
  IMG = './data/images/' + phase + '.torch'
  if os.path.isfile(IMG):
    return torch.load(IMG)
  else:
    images = []
    for i in index_range:
      if i % 100 == 0:
        print('%d images loaded.' % i)
      image_name = os.path.join('./data/images/', phase, str(i) + '.jpg')
      image = io.imread(image_name)
      images.append(torch.tensor(image))
    torch.save(images, IMG)
    return images

test_images  = load_images('test', range(3680, 7349))
train_images = load_images('train', range(len(index)))

def toPIL(tensor):
  return PIL.Image.fromarray(tensor.numpy())

class TrainingDataset(Dataset):
  def __init__(self, input_size):
    self.transform = transforms.Compose([
      toPIL,
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return len(train_images)

  def __getitem__(self, idx):
    return self.transform(train_images[idx]), labels[idx]

def transform_test_images(input_size):
  f = transforms.Compose([
    toPIL,
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return [f(x) for x in test_images]
