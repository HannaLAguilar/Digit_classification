import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import *

# Basic parameters
data_path = 'data/'
train_ratio = 0.8
valid_ratio = 0.2
batch_size = 128

# Dataset
transform = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.Grayscale(),
                                transforms.RandomHorizontalFlip(),  # data augmentation
                                transforms.RandomRotation(10),  # data augmentation
                                transforms.ToTensor()])

dataset = ImageFolder(data_path, transform=transform)
print('Total images:', len(dataset))
print('Total classes:', len(dataset.classes))

# Visualize classes
samples_by_class = num_class(dataset)
# sns.barplot(list(samples_by_class.keys()), list(samples_by_class.values()))

# Split dataset
train_idx, valid_idx, test_idx = train_valid_test(dataset, train_ratio=0.8, valid_ratio=0.2)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Data loaders
data_loaders = {
    'train': DataLoader(dataset, sampler=train_sampler, batch_size=batch_size),
    'valid': DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size),
    'test': DataLoader(dataset, sampler=test_sampler, batch_size=batch_size)
            }

# Visualize img example
images, labels = iter(data_loaders['train']).next()
images = images.numpy()
img = images[0].squeeze()
plt.figure(), plt.imshow(img, cmap='gray')
