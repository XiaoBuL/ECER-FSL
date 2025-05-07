import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
DATASET_DIR = 'path/to/yourdataset/cifar_fs'

class DatasetLoader(Dataset):

    def __init__(self, setname, args,augment=False):

        # DATASET_DIR = os.path.join(args.data_dir, 'cifar_fs')

        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'meta-train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'meta-test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'meta-val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        data = []
        label = []

        if 'num_patch' not in vars(args).keys():
            print('no num_patch parameter, set as default: 9')
            self.num_patch = 9
        else:
            self.num_patch = args.num_patch

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if args.backbone_class == 'Visformer':
            image_size = 224
            resize = 256
        else:
            image_size = 84
            resize = 96
        if setname == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        filename = data.split('/')[-2:]
        filename[-1] = data.split('/')[-1].split('.')[0]+'.png'
        img = self.transform(Image.open(data).convert('RGB'))

        return img, label,filename


if __name__ == '__main__':
    pass