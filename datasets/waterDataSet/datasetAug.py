from torchvision.transforms import transforms
from dataset2d5 import SEGData1
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

class DataSet:
    def __init__(self,train_txt_path,val_txt_path,train_batch_size,val_batch_size,resize):
        self.train_data = []
        self.val_data = []
        for line in open(train_txt_path):
            self.train_data.append(line)
        for line in open(val_txt_path):
            self.val_data.append(line)
        self.resize = resize
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
    def getTrainData(self):
        self.train_SEGData = SEGData1(self.train_data,self.resize)
        return torch.utils.data.DataLoader(self.train_SEGData, num_workers=4, batch_size=self.train_batch_size, shuffle=False,drop_last=True)
    def getValData(self):
        self.val_SEGData = SEGData1(self.val_data,self.resize)
        return torch.utils.data.DataLoader(self.val_SEGData, num_workers=4, batch_size=self.val_batch_size, shuffle=False,drop_last=True)
    def getTrainStep(self):
        return self.train_data.__len__()//self.train_batch_size
    def getValStep(self):
        return self.val_data.__len__()//self.val_batch_size
dataset = DataSet(r"datasets\waterDataSet\train_list.txt",r"datasets\waterDataSet\train_list.txt",1,1,473)
dataSet = dataset.getTrainData()

IMG_PATH=r"datasets\waterDataSet\img"
SEGLABE_PATH=r"datasets\waterDataSet\label"
new_img_path=r"datasets\waterDataSet\imgAug"
new_label_path=r"datasets\waterDataSet\labelAug"
num=dataset.getTrainStep()
print("原始数据集大小为：{}".format(num))
epoch=26
for e in range(epoch):
    print("正在数据扩充第{}轮".format(e))
    for i in tqdm(range(num)):
        new_num=len(os.listdir(new_img_path))
        img,label=dataset.train_SEGData. __getitem__(i)
        img=transforms.ToPILImage()(img)
        label=transforms.ToPILImage()(label)
        img.save(r"datasets\waterDataSet\imgAug\{}.jpeg".format(new_num+1))
        label.save(r"datasets\waterDataSet\labelAug\{}.png".format(new_num+1)) 
# colorDict_RGB,colorDict_GRAY=color_dict(r"datasets\waterDataSet\labelAug",classNum=60)
# print("colorList创建成功：",colorDict_GRAY)