import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicFCN(nn.Module):
    def __init__(self):
        super(BasicFCN, self).__init__()
        # fc layers
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # reshaping the batch of images into right shape, e.g. 32x28x28 to 32x784 for batch_size 32
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # final class scores are sent as it is
        x = self.fc3(x)
        return x



class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # conv layers 1,2
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # FC layers 1, 2, after Max pooling applied 3 times the size will be 4x4x128
        self.fc1 = nn.Linear(4 * 4 * 128, 500)
        self.fc2 = nn.Linear(500, 10)
        # drop out layer with p=0.2
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # passing through Convolution and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening the image
        x = x.view(-1, 4 * 4 * 128)
        # drop out layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # final class scores are sent as it is
        x = self.fc2(x)
        return x



def get_resnet18_pre():
    resnet18_model_pre = models.resnet18(pretrained=True)
    # first freezing all params
    for param in resnet18_model_pre.parameters():
        param.requires_grad = False

    # replacing all last fc layer with the classifier, which
    # itself consist of 2 layers, 'final_layer' is our final layer now
    classifier = nn.Sequential(OrderedDict({
        'fc1': nn.Linear(512, 256),
        'relu1': nn.ReLU(),
        'final_layer': nn.Linear(256, 10)
    }))
    resnet18_model_pre.fc = classifier
    return resnet18_model_pre


def get_resnet18():
    resnet18_model = models.resnet18(pretrained=False)
    resnet18_model.fc = nn.Sequential(OrderedDict({'final_layer': nn.Linear(512, 10)}))
    return resnet18_model