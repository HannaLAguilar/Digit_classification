import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(p=0.40)
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU())
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Classifier2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 8 * 32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 5, 1)
        self.norm2 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(9 * 9 * 48, 420)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(420, 10)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        # flattening the image
        x = x.view(-1, 9 * 9 * 48)
        x = self.dropout1(F.relu(self.fc1(x)))
        # final output
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
