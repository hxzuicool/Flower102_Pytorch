import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data
import scipy.io
import shutil
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import os


# 选择设备
device = torch.device("cuda:0")
# 对三种数据集进行不同预处理，对训练数据进行加强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 数据目录
data_dir = "./data/flowers-102/"

# 获取三个数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'valid', 'test']}
traindataset = image_datasets['train']
validdataset = image_datasets['valid']
testdataset = image_datasets['test']

batch_size = 50
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=10) for x in ['train', 'valid', 'test']}

# print(dataloaders)
traindataloader = dataloaders['train']
validdataloader = dataloaders['valid']
testdataloader = dataloaders['test']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}


# 使用resnet152的网络结构，最后一层全连接重写输出102
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        self.fc = nn.Linear(in_features=2048, out_features=102)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


resnet152 = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)

net = Net(resnet152)


def valid_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in validdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['valid']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['valid']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'valid', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def test_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in testdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['test']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:
            test_model(model, criterion)
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        running_loss = 0.0
        running_corrects = 0
        model = model.to(device)
        for inputs, labels in traindataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)
        epoch_loss = running_loss / dataset_sizes['train']
        print(dataset_sizes['train'])
        print(running_corrects.double())
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        best_acc = max(best_acc, epoch_acc)
        # loss_list.append(epoch_loss)
        # acc_list.append(epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        print()
        step_lr.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
    epochs = 48
    model = train_model(net, criterion, optimizer, epochs)

    valid_model(model, criterion)

    torch.save(model, 'model.pkl')