import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn as nn
import time
import torchvision.datasets as datasets
import os
import networkStructure

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


def valid_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model.eval()
    for inputs, labels in validdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
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
    model.eval()
    for inputs, labels in testdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

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
        model.train()
        for inputs, labels in traindataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

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
    lr = 0.01
    momentum = 0.9
    epochs = 16
    batch_size = 50
    num_works = 10

    # 选择设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 数据目录
    data_dir = "./data/flowers-102/"

    # 获取数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'valid', 'test']}
    traindataset = image_datasets['train']
    validdataset = image_datasets['valid']
    testdataset = image_datasets['test']

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_works) for x in
                   ['train', 'valid', 'test']}

    traindataloader = dataloaders['train']
    validdataloader = dataloaders['valid']
    testdataloader = dataloaders['test']

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    resnet152 = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)

    net = networkStructure.Net(resnet152)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    model = train_model(net, criterion, optimizer, epochs)

    valid_model(model, criterion)

    torch.save(model, 'model.pkl')
