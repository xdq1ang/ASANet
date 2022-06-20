from torchvision.transforms import transforms
from dataset import SEGData
from utils.colorList import color_dict
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


IMG_PATH=r".\waterDataSet\img"
SEGLABE_PATH=r".\waterDataSet\label"
new_img_path=r".\waterDataSetAug\img"
new_label_path=r".\waterDataSetAug\label"
full_dataset=list(zip(os.listdir(IMG_PATH),os.listdir(SEGLABE_PATH)))
dataSet=SEGData(full_dataset,"waterDataSet")
num=dataSet.__len__()
print("原始数据集大小为：{}".format(num))
epoch=26
for e in range(epoch):
    print("正在数据扩充第{}轮".format(e))
    for i in tqdm(range(num)):
        new_num=len(os.listdir(new_img_path))
        img,label=dataSet. __getitem__(i)
        img=transforms.ToPILImage()(img)
        label=transforms.ToPILImage()(label)
        img.save(r".\waterDataSetAug\img\{}.jpeg".format(new_num+1))
        label.save(r".\waterDataSetAug\label\{}.png".format(new_num+1)) 
colorDict_RGB,colorDict_GRAY=color_dict(r"waterDataSetAug\label",classNum=60)
print("colorList创建成功：",colorDict_GRAY)