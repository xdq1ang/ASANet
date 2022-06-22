from utils.dataset2d5 import SEGData1
import torch
import os
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from networks.unet import UNet
from networks.borderUnet import borderUnet
from utils.showPic import savePic2
from utils.metrics import metric
from utils.colorful import colorful
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as f
from utils.loss2 import FocalLoss2d


class myDataSet(SEGData1):
    def __getitem__(self, item):
        # 取出图片路径
        img_data = self.dataset[item].split(' ')[0]
        label_data = self.dataset[item].split(' ')[1]
        edge_data = self.dataset[item].split(' ')[2].replace('\n','')
        img = Image.open(img_data)
        label = Image.open(label_data)
        edge = Image.open(edge_data)
        img, label, edge = self.argu(img,label,edge)
        return img, label, edge
    def argu(self,img,mask,edge):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.3,1), ratio=(1,1))
        img = f.resized_crop(img, i, j, h, w, self.resize, interpolation=Image.BILINEAR)
        img = transforms.ToTensor()(img)

        mask = f.resized_crop(mask, i, j, h, w, self.resize, interpolation=Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long) 

        edge = f.resized_crop(edge, i, j, h, w, self.resize, interpolation=Image.NEAREST)
        edge = torch.tensor(np.array(edge), dtype=torch.long) 
        return img,mask,edge



def getModel(model_name):
    switch = {
    "UNet":lambda:UNet(n_channels=NUM_CHANNELS,n_classes=NUM_CLASSES).to(device),
    "borderUet":lambda:borderUnet(n_channels=NUM_CHANNELS,n_classes=NUM_CLASSES).to(device)
    }
    return switch[model_name]()

    
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
        self.train_SEGData = myDataSet(self.train_data,self.resize)
        return torch.utils.data.DataLoader(self.train_SEGData, num_workers=4, batch_size=self.train_batch_size, shuffle=False,drop_last=True)
    def getValData(self):
        self.val_SEGData = myDataSet(self.val_data,self.resize)
        return torch.utils.data.DataLoader(self.val_SEGData, num_workers=4, batch_size=self.val_batch_size, shuffle=False,drop_last=True)
    def getTrainStep(self):
        return self.train_data.__len__()//self.train_batch_size
    def getValStep(self):
        return self.val_data.__len__()//self.val_batch_size

def train(model_name):
    torch.cuda.empty_cache()
    BEST_SCORE = -1    #存储最优
    train_log_path = os.path.join(SAVE_PATH,model_name)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    summary=SummaryWriter(train_log_path)
    print("training")
    net = getModel(model_name)
    # init_img = torch.zeros((1,3,128,128),device=device)
    # summary.add_graph(net,init_img)
    optimizer = torch.optim.Adam(net.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8, patience = 6, verbose=True)
    train_step=dataset.getTrainStep()
    for epoch in range(EPOCH):
        with tqdm(total=train_step) as t:
            t.set_description('Epoch %i' % epoch)
            train_epoch_score=[]
            train_epoch_loss=[]
            train_epoch_border_loss = []
            net.train()
            for i, (img,label,edge) in enumerate(dataset.getTrainData()):
                img = img.to(device)
                label = label.squeeze(dim = 1).to(device)      # (12,1,128,128)   -> (12,128,128)   
                edge = edge.squeeze(dim = 1).to(device)  
                optimizer.zero_grad()   
                pred,border_pred=net(img)
                loss=SEG_LOSS(pred,label)    #不需要”对标签进行one-hot编码。
                border_loss = BORDER_LOSS(border_pred,edge)
                predict=torch.argmax(pred, dim=1)
                
                loss_all = loss + border_loss
                loss_all.backward()
                optimizer.step()
                #性能评估
                me = metric(predict,label,LABEL)
                train_score =me.miou()
                train_epoch_score.append(train_score)
                train_epoch_loss.append(loss_all.item())
                train_epoch_border_loss.append(border_loss.item())
                #设置tqdm输出
                t.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'])
                t.update(1)
            #保存训练日志
            summary.add_scalar('train_loss',np.array(train_epoch_loss).mean(),epoch)
            summary.add_scalar('train_border_loss',np.array(train_epoch_border_loss).mean(),epoch)
            summary.add_scalar('train_score',np.array(train_epoch_score).mean(),epoch)
            summary.add_scalar('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        val_score, val_loss= eval(net,dataset,epoch,summary,model_name)
        scheduler.step(val_loss)

        # 最优模型保存
        if val_score > BEST_SCORE:
            net.eval()
            BEST_SCORE = val_score
            model_save_path = os.path.join(train_log_path,"model")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(net.state_dict(),os.path.join(model_save_path,"model.pt"))
            # 保存预测结果
            if not os.path.exists(train_log_path+"/eval_pic"):
                os.makedirs(train_log_path+"/eval_pic")
            best_id = len(os.listdir(train_log_path+"/eval_pic"))+1
            eval_path = os.path.join(train_log_path,"eval_pic","best_"+str(best_id))
            with tqdm(total=dataset.getValStep()) as t2:
                with torch.no_grad():
                    for i, (img,label,edge) in enumerate(dataset.getValData()):
                        img = img.to(device)    
                        label = label.squeeze(dim = 1).cuda()
                        pred, border_pred=net(img)
                        predict=torch.argmax(pred, dim=1)
                        border_predict=torch.argmax(border_pred, dim=1)
                        # savePic(img.cpu()[0],label[0],predict[0],COLOR_DICT,eval_path,summary,epoch)
                        savePic2(img[0],label[0],edge[0],predict[0],border_predict[0],COLOR_DICT,eval_path,summary,epoch,tag="Test")
                        t2.update()


def eval(net,dataset,epoch,summary,model_name):
    print("evaling")
    eval_step = dataset.getValStep()
    val_epoch_score = []
    val_epoch_loss = []
    val_epoch_border_loss = []
    val_iou = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=eval_step) as t2:
            for i, (img,label,edge) in enumerate(dataset.getValData()):   
                img = img.to(device) 
                label = label.squeeze(dim = 1).cuda()
                edge = edge.squeeze(dim = 1).cuda()
                pred, border_pred=net(img)
                loss = SEG_LOSS(pred,label)
                border_loss = BORDER_LOSS(border_pred,edge)
                loss_all = loss + 0.2*border_loss
                predict=torch.argmax(pred, dim=1)
                #性能评估
                me = metric(predict,label,LABEL)
                iou = list(me.iou())
                val_iou.append(iou)
                score = me.miou()
                val_epoch_score.append(score)
                val_epoch_loss.append(loss_all.item())
                val_epoch_border_loss.append(border_loss.item())
                # savePic(img[0,:-1,:,:],label[0],predict[0],COLOR_DICT,train_log_path)
                t2.update()
    val_iou = np.nanmean(np.array(val_iou),axis=0)
    summary.add_scalar('val_loss',np.array(val_epoch_loss).mean(),epoch)
    summary.add_scalar('val_border_loss',np.array(val_epoch_border_loss).mean(),epoch)
    summary.add_scalar('val_score',np.array(val_epoch_score).mean(),epoch)

    for n in range(len(val_iou)):
        summary.add_scalar('val_iou_class'+str(n),val_iou[n],epoch)

    print('第{}轮结束,val_score:{}'.format(epoch,np.array(val_epoch_score).mean()))
    return np.array(val_epoch_score).mean(),np.array(val_epoch_loss).mean()





if __name__ == '__main__':
    # WHDLD数据集lable为单通道图。值为[0,1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    EPOCH=80
    TRAIN_TXT_PATH = r"datasets\Postdam\output\train_list.txt"
    VAL_TXT_PATH = r"datasets\Postdam\output\val_list.txt"
    SAVE_PATH=r"snapshots\train_source_postdam_model"
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1
    LR = 0.001
    
    COLOR_DICT = {  0 : [255, 0, 0] ,
                    1 : [0, 255, 0] ,
                    2 : [255, 255, 0],
                    3 : [0, 0, 255],
                    4 : [0, 255, 255],
                    5 : [255, 255, 255]
                    }

    LABEL=[0,1,2,3,4,5]
    NUM_CLASSES = len(COLOR_DICT.keys())
    BORDER_LOSS =  FocalLoss2d(gamma = 2.0)
    SEG_LOSS = nn.CrossEntropyLoss()
    NUM_CHANNELS = 3
    RESIZE = 128
    dataset = DataSet(TRAIN_TXT_PATH,VAL_TXT_PATH,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,RESIZE)

    train("borderUet")