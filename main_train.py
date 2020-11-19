import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

from utils import *
from model_cnn import Classifier

##########
#  DATA  #
##########

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
classes = dataset.classes
print('Total images:', len(dataset))
print('Total classes:', len(classes))

# Visualize classes
samples_by_class = num_class(dataset)
# sns.barplot(x=list(samples_by_class.keys()), y=list(samples_by_class.values()))

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

######################
# TRAINING THE MODEL #
######################

# CNN architecture
model = Classifier()
print(model)

# CPU/GPU settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on {}'.format(device.type))
model = model.to(device)

# Training
epochs = 10
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# train_loss, valid_loss, valid_acc = train_model(epochs, model, data_loaders, criterion, optimizer, device)

# plt.subplot(121)
# plt.plot(train_loss, label='Train loss')
# plt.plot(valid_loss, label='Valid loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(122)
# plt.plot(valid_acc, label='Valid Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

######################
# TEST THE MODEL #
######################
state_dict = torch.load('classifier_digit.pt')
model.load_state_dict(state_dict)
test_acc, a, b = test_model(classes, model, data_loaders, device)
