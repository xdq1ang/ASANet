import sys
sys.path.append("./")
import os
#from cfg import *
from torch.utils.data import  Dataset
import torch
from PIL import  Image
#from colorList import color_dict
#from colorList import colorMapping
import numpy as np
import torchvision.transforms as transforms  
import random
from customizeTransforms import AddPepperNoise
from customizeTransforms import AddGaussianNoise


class SEGData(Dataset):
    def __init__(self,dataset,dataset_root,mode=None):
        self.dataset_root=dataset_root
        self.dataset=dataset
        self.mode=mode
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        # 取出图片路径
        img_data=self.dataset[item][0]
        label_data=self.dataset[item][1]
        # if self.mode=="val":
        #     print("testing_img:",img_data)
        img = Image.open(os.path.join(self.dataset_root+"/img/",img_data))
        label = Image.open(os.path.join(self.dataset_root+"/label/",label_data))
        #img=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1)(img)
        #加入椒盐噪声,高斯噪声
        #img=AddPepperNoise(snr=0.99,p=0.5)(img)
        #img=AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45))(img)
        p1 = random.randint(0,1)
        p2 = random.randint(0,1)
        transform= transforms.Compose([
            #transforms.Resize(size=473),
            transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2)])
        img=transform(img)
        label=transform(label)
        #######
        #旋转和仿射变换
        seed = np.random.randint(214)
        #torch.manual_seed(seed)
        random.seed(seed)
        img=transforms.RandomAffine(degrees=(0, 360))(img)
        #torch.manual_seed(seed)
        random.seed(seed)
        label=transforms.RandomAffine(degrees=(0, 360))(label)
        #img=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1)(img)
        img=transforms.ToTensor()(img)
        label=transforms.ToTensor()(label)
        #img=img.float()
        label=label.float()
        #img=img/255.0
        
        
        return img, label

