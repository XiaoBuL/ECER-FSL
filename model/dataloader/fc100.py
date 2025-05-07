# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Dataloader build upon the DeepEMD repository, available under https://github.com/icoz69/DeepEMD/tree/master/Models/dataloader
"""
#
#
import os
import os.path as osp
from random import shuffle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
Normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

class DatasetLoader(Dataset):

    def __init__(self, setname,args, augment=False):
        DATASET_DIR = 'path/to/your/FC100'
        if args.backbone_class == 'Visformer':
            image_size = 224
            resize = 256
        else:
            image_size = 84
            resize = 96
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
            self.transform = transforms.Compose([
                
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # Normalize
                # cifarf aug
                # transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # Normalize
                
                #semfew aug
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize
            ])
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                Normalize
                
                # # semfew aug
                # transforms.ToTensor(),
                # Normalize
            ])
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                Normalize
                
                
                # # semfew aug
                # transforms.ToTensor(),
                # Normalize
            ])
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]
        self.label2catname = {}

        for idx, this_folder in enumerate(folders):
            self.label2catname[idx] = this_folder.split('\\')[-1]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        filename = 'path/to/yourcode/sem_json/FC100_Tieredimagenet_NumToName.json'
        with open(filename, 'r') as f:
            self.idxtoname = json.load(f)
            self.idxtoname = self.idxtoname['cifarfs']
        # Transformation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        filename = path.split('/')[-1]
        class_name = path.split('/')[-2]
        class_name=self.idxtoname[str(int(class_name[2:]))]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, [class_name,filename]

class DivisionCifar(Dataset):

    def __init__(self, file):
        file = os.path.join('E:\deeplearning\datasets\FC100', file + '.csv')
        data = []
        label = []
        with open(file, 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')  # 去除文本中的换行符
                ann = ann.split(',')
                data.append(ann[0])
                label.append(ann[1])

        label_name = list(set(label))
        shuffle(label_name)
        self.label2catname = {}
        for i in range(len(label_name)):
            self.label2catname[i] = label_name[i]
        name2label = {k: v for v, k in self.label2catname.items()}

        self.label = [name2label[l] for l in label]
        self.data = data
        self.num_class = len(set(label))

        # Transformation

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass
