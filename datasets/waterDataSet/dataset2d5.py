from ast import arg
import torch
import torchvision
from torch.utils.data import  Dataset
from PIL import  Image
import numpy as np
import torchvision.transforms as transforms  
import torchvision.transforms.functional as f





class SEGData(Dataset):
    def __init__(self,dataset,resize,id_to_trainid):
        self.dataset=dataset
        self.resize = resize
        self.id_to_trainid = id_to_trainid
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        # 取出图片路径
        img_data = self.dataset[item].split(' ')[0]
        label_data = self.dataset[item].split(' ')[1].replace('\n','')
        img = Image.open(img_data)
        label = Image.open(label_data)
        data = np.array(label)
        transform =transforms.Resize(size=self.resize,interpolation=0)
        img = transform(img)
        label = transform(label)
        #转化为tensor
        img = transforms.ToTensor()(img)
        label = torch.tensor(np.array(label), dtype=torch.long) 

        label = np.asarray(label, np.float32)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label_copy = torch.tensor(label_copy,dtype=torch.long)
        return img, label_copy

class SEGData1(Dataset):
    def __init__(self,dataset,resize):
        self.dataset=dataset
        self.resize = resize
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        # 取出图片路径
        img_data = self.dataset[item].split(' ')[0]
        label_data = self.dataset[item].split(' ')[1].replace('\n','')
        img = Image.open(img_data)
        label = Image.open(label_data)
        # resize
        # transform =transforms.Resize(size=self.resize,interpolation=0)
        # img = transform(img)
        # label = transform(label)

        # 数据增强
        img,label = self.argu(img,label)
        return img, label

    def argu(self,img,mask):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.3,1), ratio=(1,1))
        img = f.resized_crop(img, i, j, h, w, self.resize, interpolation=Image.BICUBIC)
        img = transforms.ToTensor()(img)

        mask = f.resized_crop(mask, i, j, h, w, self.resize, interpolation=Image.BICUBIC)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8) 
        return img,mask

