# 训练数据格式：
# 多类，输入数据为3通道，标签为单通道

import imp
from sklearn.metrics import recall_score
from utils.dataset2d5 import SEGData1
import torch
import os
from torch import nn
import numpy as np
from tqdm import tqdm
from networks.unet import UNet
from utils.showPic import showPic,savePic
from utils.metrics import metric
from utils.colorful import colorful
from utils.savePred import savePred
from tqdm import tqdm




def getModel(model_name):
    if model_name == "UNet":
        return UNet(n_channels=NUM_CHANNELS,n_classes=NUM_CLASSES).to(device)

class DataSet:
    train_data = []
    val_data = []
    def __init__(self,train_txt_path,val_txt_path,train_batch_size,val_batch_size,resize):
        for line in open(train_txt_path):
            self.train_data.append(line)
        for line in open(val_txt_path):
            self.val_data.append(line)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_SEGData = SEGData1(self.train_data,resize)
        self.val_SEGData = SEGData1(self.val_data,resize)
        self.trainLoader = torch.utils.data.DataLoader(self.train_SEGData, num_workers=0, batch_size=train_batch_size, shuffle=False,drop_last=True)
        self.valLoader = torch.utils.data.DataLoader(self.val_SEGData, num_workers=0, batch_size=val_batch_size, shuffle=False)
    def getTrainData(self):
        return self.trainLoader
    def getValData(self):
        return self.valLoader
    def getTrainStep(self):
        return self.train_data.__len__()//self.train_batch_size
    def getValStep(self):
        return self.val_data.__len__()//self.val_batch_size

def eval(model_name,model_path):
    '''
    model_name : "DensePPMUNet_a" ,"CE-Net","UNet","mrUNet","ResUNetA","LANet","PSPNet","deepLabv3"
    model_path : the tarined model path
    '''

    train_log_path = os.path.join(SAVE_PATH)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    net = getModel(model_name)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    net.eval()
    IOU = []
    F1 = []
    Recall = []
    for i, (img,label) in enumerate(tqdm(dataset.getValData())):
        img = img.to(device)
        label = label.squeeze(dim = 1).to(device).long()      # (12,1,128,128)   -> (12,128,128)   
        if model_name == "LANet":
            pred,aux_out=net(img)
            predict=torch.argmax(pred, dim=1)
        elif model_name=="PSPNet":
            pred= net(img, label)
            predict=torch.argmax(pred, dim=1)
        else:
            pred=net(img)
            predict=torch.argmax(pred, dim=1)
        me = metric(predict,label,LABEL)
        iou = list(me.iou())
        f1 ,recall= me.f1_recall()
        IOU.append(iou)
        F1.append(list(f1))
        Recall.append(list(recall))
        # 保存图片
        eval_path = os.path.join(SAVE_PATH,"pred")
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        # savePic(img.cpu()[0],label[0],predict[0],COLOR_DICT,eval_path)
        showPic(img[0].cpu(),label[0].cpu().detach().numpy(),predict[0].cpu().detach().numpy(),COLOR_DICT,eval_path)
        # 保存预测概率
        savePred(pred.cpu(),SAVE_PATH) 
    print("IoU:",np.nanmean(np.array(IOU),axis=0))
    print("F1:",np.nanmean(np.array(F1),axis=0))
    print("Recall:",np.nanmean(np.array(Recall),axis=0))

        

if __name__ == '__main__':
    # WHDLD数据集lable为单通道图。值为[0,1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_TXT_PATH = r"datasets\WHDLD\train_list.txt"
    VAL_TXT_PATH = r"datasets\WHDLD\val_list.txt"
    SAVE_PATH=r"snapshots\buildings_WHDLD\val_Unet_on_tar"
    model_path = r"datasets\buildings\训练日志\UNet\model\model.pt"
    TRAIN_BATCH_SIZE = 12
    VAL_BATCH_SIZE = 1
    RESIZE = 128
    COLOR_DICT = { 0:[0,0,0],
                   1:[255,0,0]}
    LABEL = [0,1]
    NUM_CLASSES = len(COLOR_DICT.keys())
    NUM_CHANNELS = 3
    dataset = DataSet(TRAIN_TXT_PATH,VAL_TXT_PATH,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,RESIZE)
    eval("UNet",model_path)
    



    


