from networks.dfn.dfn import DFN
from torch import nn
import torch
from networks.dfn.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from networks.dfn.dataset import SEGData1
import os
import numpy as np
from tqdm import tqdm
from utils.metrics import metric
from utils.showPic import showPic,savePic1
from networks.dfn.init_func import init_weight, group_weight
from networks.dfn.lr_policy import PolyLR
from torch.utils.tensorboard import SummaryWriter



def getModel(model_name):
    switch = {
    "DFN":lambda:DFN(NUM_CLASSES, criterion=nn.CrossEntropyLoss(reduction='mean',ignore_index=255),
                    aux_criterion=SigmoidFocalLoss(ignore_label=255, gamma=2.0, alpha=0.25), 
                    alpha=0.1,
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d).to(device),
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
        self.train_SEGData = SEGData1(self.train_data,self.resize)
        return torch.utils.data.DataLoader(self.train_SEGData, num_workers=4, batch_size=self.train_batch_size, shuffle=False,drop_last=True)
    def getValData(self):
        self.val_SEGData = SEGData1(self.val_data,self.resize)
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
    init_weight(net.business_layer, nn.init.kaiming_normal_,
                net.norm_layer, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')

    base_lr = 7e-4
    params_list = []
    params_list = group_weight(params_list, net.backbone,
                               net.norm_layer, base_lr)
    for module in net.business_layer:
        params_list = group_weight(params_list, module, net.norm_layer,base_lr * 10)

    
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum= 0.9,
                                weight_decay=1e-4)
    train_step=dataset.getTrainStep()
    total_iteration = 100 * train_step
    lr_policy = PolyLR(base_lr, 0.9, total_iteration)


    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.8, patience = 6, verbose=True)
    
    for epoch in range(EPOCH):
        with tqdm(total=train_step) as t:
            t.set_description('Epoch %i' % epoch)
            train_epoch_loss=[]
            train_epoch_auxloss=[]
            net.train()
            for i, (img,label,edge) in enumerate(dataset.getTrainData()):
                img = img.to(device)
                label = label.squeeze(dim = 1).to(device)      # (12,1,128,128)   -> (12,128,128)   
                edge = edge.squeeze(dim = 1).to(device)
                optimizer.zero_grad()   
                loss, aux_loss = net(img, label, edge)
                loss.backward()
                optimizer.step()
                #性能评估
                train_epoch_loss.append(loss.item())
                train_epoch_auxloss.append(aux_loss.item())
                #
                current_idx = epoch * train_step + i
                lr = lr_policy.get_lr(current_idx)
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr
                for i in range(2, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr * 10
                #设置tqdm输出
                t.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'])
                t.update(1)
            #保存训练日志
            summary.add_scalar('train_loss',np.array(train_epoch_loss).mean(),epoch)
            summary.add_scalar('train_auxloss',np.array(train_epoch_auxloss).mean(),epoch)
            summary.add_scalar('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
        val_score = eval(net,dataset,epoch,summary,model_name)
        # scheduler.step(val_score)

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
                        edge = edge.squeeze(dim = 1).to(device)
                        logits1, logits2=net(data = img)
                        predict1=torch.argmax(logits1, dim=1)
                        logits2[logits2>=0.5]=1
                        logits2[logits2<0.5]=0
                        predict2=logits2[0]
                        twoPred = torch.cat([predict1,predict2], dim = 2)
                        savePic1(img.cpu()[0],label[0],twoPred[0],COLOR_DICT,eval_path,summary,epoch,"eval")
                        t2.update()


def eval(net,dataset,epoch,summary,model_name):
    print("evaling")
    eval_step = dataset.getValStep()
    val_epoch_score1 = []
    val_epoch_score2 = []
    val_iou1 = []
    val_iou2 = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=eval_step) as t2:
            for i, (img,label,edge) in enumerate(dataset.getValData()):   
                img = img.to(device) 
                label = label.squeeze(dim = 1).cuda()
                edge = edge.squeeze(dim = 1).to(device)
                logits1,logits2=net(data = img)
                predict1=torch.argmax(logits1, dim=1)
                predict2=torch.argmax(logits2, dim=1)
                #性能评估
                me1 = metric(predict1,label,LABEL)
                iou1 = list(me1.iou())
                val_iou1.append(iou1)
                score1 = me1.miou()
                val_epoch_score1.append(score1)

                me2 = metric(predict2,edge,LABEL)
                iou2 = list(me2.iou())
                val_iou2.append(iou2)
                score2 = me2.miou()
                val_epoch_score2.append(score2)
                # savePic(img[0,:-1,:,:],label[0],predict[0],COLOR_DICT,train_log_path)
                t2.update()
    val_iou1 = np.nanmean(np.array(val_iou1),axis=0)
    val_iou2 = np.nanmean(np.array(val_iou2),axis=0)
    summary.add_scalar('val_score_label',np.array(val_epoch_score1).mean(),epoch)
    summary.add_scalar('val_score_edge',np.array(val_epoch_score2).mean(),epoch)

    for n in range(len(val_iou1)):
        summary.add_scalar('val_iou_class_label'+str(n),val_iou1[n],epoch)
    for n in range(len(val_iou2)):
        summary.add_scalar('val_iou_class_edge'+str(n),val_iou2[n],epoch)

    print('第{}轮结束,val_score_label:{}'.format(epoch,np.array(val_epoch_score1).mean()))
    print('第{}轮结束,val_score_edge:{}'.format(epoch,np.array(val_epoch_score2).mean()))
    return np.array(val_epoch_score1).mean()





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    EPOCH=100
    TRAIN_TXT_PATH = r"datasets\buildings\train_list.txt"
    VAL_TXT_PATH = r"datasets\buildings\val_list.txt"
    SAVE_PATH=r"snapshots\DFN"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 1
    LR = 0.001
    COLOR_DICT = {0 : [0, 0, 0] ,
                  1 : [255, 0, 0]}
    LABEL=[0,1]
    NUM_CLASSES = len(LABEL)
    NUM_CHANNELS = 3
    RESIZE = [128,128]
    dataset = DataSet(TRAIN_TXT_PATH,VAL_TXT_PATH,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,RESIZE)

    train("DFN")