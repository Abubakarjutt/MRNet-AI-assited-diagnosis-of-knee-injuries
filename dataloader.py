import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine


class MRDataset(data.Dataset):
    def __init__(self, root_dir, plane, train=True, transform=None, weights=None):
        super().__init__()
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records_abnormal = pd.read_csv(self.root_dir + 'train-abnormal.csv', header=None, names=['id', 'label'])
            self.records_acl = pd.read_csv(self.root_dir + 'train-acl.csv', header=None, names=['id', 'label'])
            self.records_meniscus = pd.read_csv(self.root_dir + 'train-meniscus.csv', header=None, names=['id', 'label'])

        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records_abnormal = pd.read_csv(self.root_dir + 'valid-abnormal.csv', header=None, names=['id', 'label'])
            self.records_acl = pd.read_csv(self.root_dir + 'valid-acl.csv', header=None, names=['id', 'label'])
            self.records_meniscus = pd.read_csv(self.root_dir + 'valid-meniscus.csv', header=None, names=['id', 'label'])

        self.records_abnormal['id'] = self.records_abnormal['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))
        self.records_acl['id'] = self.records_acl['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))
        self.records_meniscus['id'] = self.records_meniscus['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))

        self.paths = [self.folder_path + filename +'.npy' for filename in self.records_abnormal['id'].tolist()]
        self.labels_abnormal = self.records_abnormal['label'].tolist()
        self.labels_acl = self.records_acl['label'].tolist()
        self.labels_meniscus = self.records_meniscus['label'].tolist()

        self.transform = transform
        if weights is None:
            pos_abnormal = np.sum(self.labels_abnormal)
            neg_abnormal = len(self.labels_abnormal) - pos_abnormal
            weights_abnormal = torch.FloatTensor([1, neg_abnormal / pos_abnormal])


            pos_acl = np.sum(self.labels_acl)
            neg_acl = len(self.labels_acl) - pos_acl
            weights_acl = torch.FloatTensor([1, neg_acl / pos_acl])


            pos_meniscus = np.sum(self.labels_meniscus)
            neg_meniscus = len(self.labels_meniscus) - pos_meniscus
            weights_meniscus = torch.FloatTensor([1, neg_meniscus / pos_meniscus])



            self.weights = torch.Tensor([weights_abnormal[1] ,weights_acl[1], weights_meniscus[1]])

        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label_abnormal = self.labels_abnormal[index]
        label_acl = self.labels_acl[index]
        label_meniscus = self.labels_meniscus[index]
        if label_abnormal == 1 and label_acl == 1 and label_meniscus ==1:
            label = torch.FloatTensor([1, 1, 1])
        elif label_abnormal == 0 and label_acl == 1 and label_meniscus ==1:
            label = torch.FloatTensor([0, 1, 1])
        elif label_abnormal == 1 and label_acl == 0 and label_meniscus ==1:
            label = torch.FloatTensor([1, 0, 1])
        elif label_abnormal == 1 and label_acl == 1 and label_meniscus ==0:
            label = torch.FloatTensor([1, 1, 0])
        else:
            label = torch.FloatTensor([0, 0, 0])

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #     weight = np.array([self.weights[1]])
        #     weight = torch.FloatTensor(weight)
        # else:
        #     weight = np.array([self.weights[0]])
        #     weight = torch.FloatTensor(weight)

        return array, label, self.weights
