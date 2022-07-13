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



if __name__ == '__main__':
    labels_mat = scipy.io.loadmat('./data/flowers-102/imagelabels.mat')
    print(labels_mat)

    id_mat = scipy.io.loadmat('./data/flowers-102/setid.mat')
    print(id_mat)

    print(len(labels_mat['labels'][0]))
    print(len(id_mat['trnid'][0]))
    print(len(id_mat['valid'][0]))
    print(len(id_mat['tstid'][0]))



    jpgname = ''
    for i in id_mat['valid'][0]:
        if 0 < i < 10:
            jpgname = 'image_0000{}.jpg'.format(i)
        elif 10 <= i < 100:
            jpgname = 'image_000{}.jpg'.format(i)
        elif 100 <= i < 1000:
            jpgname = 'image_00{}.jpg'.format(i)
        elif 1000 <= i < 10000:
            jpgname = 'image_0{}.jpg'.format(i)

        if os.path.exists('./data/flowers-102/valid/{}'.format(labels_mat['labels'][0][i-1])):
            shutil.copyfile('./data/flowers-102/jpg/' + jpgname, './data/flowers-102/valid/{}/{}'.format(labels_mat['labels'][0][i-1], jpgname))
        else:
            os.mkdir('./data/flowers-102/valid/{}'.format(labels_mat['labels'][0][i-1]))
            shutil.copyfile('./data/flowers-102/jpg/' + jpgname,
                            './data/flowers-102/valid/{}/{}'.format(labels_mat['labels'][0][i - 1], jpgname))