from utils.dataset2d5 import SEGData
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from utils.showPic import showPic,savePic
from utils.metrics import metric
from networks.unet import UNet
from utils.loss2 import CrossEntropyLoss2d,FocalLoss2d




def getModel(model_name):
    switch = {
    "UNet":lambda:UNet(n_channels=NUM_CHANNELS,n_classes=NUM_CLASSES).to(device),
    }
    return switch[model_name]()

class DataSet:
    train_data = []
    val_data = []
    def __init__(self,train_txt_path,val_txt_path,train_batch_size,val_batch_size,resize,ID_TO_TRAINIED):
        for line in open(train_txt_path):
            self.train_data.append(line)
        for line in open(val_txt_path):
            self.val_data.append(line)
        self.resize = resize
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.id_to_trainied = ID_TO_TRAINIED
        # self.trainLoader = torch.utils.data.DataLoader(self.train_SEGData, num_workers=0, batch_size=train_batch_size, shuffle=False,drop_last=True)
        # self.valLoader = torch.utils.data.DataLoader(self.val_SEGData, num_workers=0, batch_size=val_batch_size, shuffle=False)
    def getTrainData(self):
        self.train_SEGData = SEGData(self.train_data,self.resize,self.id_to_trainied)
        return torch.utils.data.DataLoader(self.train_SEGData, num_workers=4, batch_size=self.train_batch_size, shuffle=False,drop_last=True)
    def getValData(self):
        self.val_SEGData = SEGData(self.val_data,self.resize,self.id_to_trainied )
        return torch.utils.data.DataLoader(self.val_SEGData, num_workers=4, batch_size=self.val_batch_size, shuffle=False)
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
            net.train()
            for i, (img,label) in enumerate(dataset.getTrainData()):
                img = img.to(device)
                label = label.squeeze(dim = 1).to(device)      # (12,1,128,128)   -> (12,128,128)   
                optimizer.zero_grad()   
                if model_name == "LANet":
                    pred,aux_out=net(img)
                    main_loss = CrossEntropyLoss2d()(pred, label)
                    aux_loss = CrossEntropyLoss2d()(aux_out, label)
                    loss = main_loss + 0.3*aux_loss
                    predict=torch.argmax(pred, dim=1)
                elif model_name=="PSPNet" or model_name== "DensePSPNet" or model_name == "ResNet":
                    predict, main_loss, aux_loss = net(img, label)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    pred = net(img)
                    loss=LOSS(pred,label) 
                    predict = torch.argmax(pred, dim=1)   
                    
                loss.backward()
                optimizer.step()
                #性能评估
                me = metric(predict,label,LABEL)
                train_score =me.miou()
                train_epoch_score.append(train_score)
                train_epoch_loss.append(loss.item())
                #设置tqdm输出
                t.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'])
                t.update(1)
            #保存训练日志
            summary.add_scalar('train_celoss',np.array(train_epoch_loss).mean(),epoch)
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
                    for i, (img,label) in enumerate(dataset.getValData()):
                        img = img.to(device)    
                        label = label.squeeze(dim = 1).cuda()
                        if model_name == "LANet":
                            pred,_=net(img) 
                            predict=torch.argmax(pred, dim=1)
                        elif model_name == "PSPNet" or model_name== "DensePSPNet" or model_name == "ResNet":
                            pred = net(img, label)
                            predict=torch.argmax(pred, dim=1)
                        else:
                            pred=net(img)
                            predict=torch.argmax(pred, dim=1)
                        savePic(img.cpu()[0],label[0],predict[0],eval_path,summary,epoch)
                        t2.update()


def eval(net,dataset,epoch,summary,model_name):
    print("evaling")
    eval_step = dataset.getValStep()
    val_epoch_score = []
    val_epoch_loss = []
    val_iou = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=eval_step) as t2:
            for i, (img,label) in enumerate(dataset.getValData()):   
                img = img.to(device) 
                label = label.squeeze(dim = 1).cuda()
                if model_name == "LANet":
                    pred,aux_out = net (img) 
                    main_loss = CrossEntropyLoss2d()(pred, label)
                    aux_loss = CrossEntropyLoss2d()(aux_out, label)
                    loss = main_loss + 0.3*aux_loss
                    predict=torch.argmax(pred, dim=1)  
                elif model_name == "PSPNet" or model_name== "DensePSPNet" or model_name == "ResNet":
                    pred = net(img, label)
                    loss = LOSS(pred,label)
                    predict=torch.argmax(pred, dim=1)
                else:
                    pred=net(img)
                    loss = LOSS(pred,label)
                    predict=torch.argmax(pred, dim=1)
                #性能评估
                me = metric(predict,label,LABEL)
                iou = list(me.iou())
                val_iou.append(iou)
                score = me.miou()
                val_epoch_score.append(score)
                val_epoch_loss.append(loss.item())
                # savePic(img[0,:-1,:,:],label[0],predict[0],COLOR_DICT,train_log_path)
                t2.update()
    val_iou = np.nanmean(np.array(val_iou),axis=0)
    summary.add_scalar('val_celoss',np.array(val_epoch_loss).mean(),epoch)
    summary.add_scalar('val_score',np.array(val_epoch_score).mean(),epoch)

    for n in range(len(val_iou)):
        summary.add_scalar('val_iou_class'+str(n),val_iou[n],epoch)

    print('第{}轮结束,val_score:{}'.format(epoch,np.array(val_epoch_score).mean()))
    return np.array(val_epoch_score).mean(),np.array(val_epoch_loss).mean()





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    EPOCH=80
    TRAIN_TXT_PATH = r"datasets\gta5\train_list.txt"
    VAL_TXT_PATH = r"datasets\gta5\val_list.txt"
    SAVE_PATH=r"snapshots\train_source_model"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 1
    LR = 0.001
    ID_TO_TRAINIED =  {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                    26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    LABEL=ID_TO_TRAINIED.values()
    NUM_CLASSES = len(ID_TO_TRAINIED)
    LOSS = nn.CrossEntropyLoss(ignore_index=255)
    NUM_CHANNELS = 3
    RESIZE = [512,256]
    dataset = DataSet(TRAIN_TXT_PATH,VAL_TXT_PATH,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,RESIZE,ID_TO_TRAINIED)

    train("UNet")