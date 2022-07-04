import sys
import os
import os.path as osp
from pprint import pprint
import timeit
from tqdm import tqdm
from utils.metrics import metric
from utils.loss2 import FocalLoss2d

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils.dataset2d5 import SEGData1
from networks.borderUnet import borderUnet
from networks.borderDiscriminator import borderDiscriminator
from utils.SurfaceLoss import SurfaceLoss,class2one_hot,one_hot2dist
from networks.discriminator import EightwayASADiscriminator
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import cityscapesDataSet
from options import gta5asa_opt
from utils.utils import colorize_mask
from utils.showPic import showPic,savePic2
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as f
args = gta5asa_opt.get_arguments()


class myDataSet(SEGData1):
    def __init__(self, dataset, resize,edge=False):
        super().__init__(dataset, resize)
        self.edge = edge
    def __getitem__(self, item):
        # 取出图片路径
        names = self.dataset[item].split(' ')
        img_data = names[0]
        label_data = names[1].replace('\n','')
        img = Image.open(img_data)
        label = Image.open(label_data)
        if self.edge:
            edge_data = names[2].replace('\n','')
            edge = Image.open(edge_data)
            img, label, edge = self.argu(img,label,edge)
            return img, label, edge
        else:
            img, label, edge = self.argu(img,label,None)
            return img, label, self.edge
    def argu(self,img,mask,edge):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.3,1), ratio=(1,1))
        img = f.resized_crop(img, i, j, h, w, self.resize, interpolation=Image.BILINEAR)
        img = transforms.ToTensor()(img)

        mask = f.resized_crop(mask, i, j, h, w, self.resize, interpolation=Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long) 
        if edge != None:
            edge = f.resized_crop(edge, i, j, h, w, self.resize, interpolation=Image.NEAREST)
            edge = torch.tensor(np.array(edge), dtype=torch.long) 
        return img,mask,edge

class DataSet:
    def __init__(self,train_txt_path,val_txt_path,train_batch_size,val_batch_size,resize,edge):
        self.train_data = []
        self.val_data = []
        for line in open(train_txt_path):
            self.train_data.append(line)
        for line in open(val_txt_path):
            self.val_data.append(line)
        self.resize = resize
        self.edge = edge
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
    def getTrainData(self):
        self.train_SEGData = myDataSet(self.train_data,self.resize,self.edge)
        return torch.utils.data.DataLoader(self.train_SEGData, num_workers=4, batch_size=self.train_batch_size, shuffle=False,drop_last=True)
    def getValData(self):
        self.val_SEGData = myDataSet(self.val_data,self.resize,self.edge)
        return torch.utils.data.DataLoader(self.val_SEGData, num_workers=4, batch_size=self.val_batch_size, shuffle=False,drop_last=True)
    def getTrainStep(self):
        return self.train_data.__len__()//self.train_batch_size
    def getValStep(self):
        return self.val_data.__len__()//self.val_batch_size

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    criterion = CrossEntropy2d().to(device)
    return criterion(pred, label)

def border_loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    criterion = FocalLoss2d().to(device)
    return criterion(pred, label)

def loss_cal_surfaceLoss(logits, label):
    label = label.long()
    label = class2one_hot(label, 2)

    pred = nn.Softmax()(logits)
    pred = class2one_hot(pred, 2)
    # pred = pred.cpu().detach().numpy()
    pred = one_hot2dist(pred)   #bcwh

    res = SurfaceLoss()(pred, label, None)
    return res

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.EPOCH, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.EPOCH, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""
    save_dir = osp.join(args.snapshot_dir, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    cudnn.enabled = True
    # Create network
    if args.backbone == "unet":
        model = borderUnet(n_classes=args.num_classes)
    if args.resume:
        print("Resuming from ==>>", args.resume)
        state_dict = torch.load(args.resume,map_location=torch.device(device))
        model.load_state_dict(state_dict)
    model.train()
    model.to(device)
    cudnn.benchmark = True

    # init D
    model_seg_D = EightwayASADiscriminator(num_classes=args.num_classes)
    # model_D = FCDiscriminator(num_classes=args.num_classes)
    model_seg_D.train()
    model_seg_D.to(device)

    model_fea_D = borderDiscriminator(in_channel = 256)
    model_fea_D.train()
    model_fea_D.to(device)

    log = open(os.path.join(save_dir,"model_structure.txt"), mode='a',encoding='utf-8')
    print("model: ", model,"\n", file=log)
    print("model_D: ", model_seg_D, file=log)
    log.close()

    pprint(vars(args))
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    src_dataset = DataSet(args.data_list[0],args.data_list[1],args.src_batch_size[0],args.src_batch_size[1],input_size,True)
    tar_dataset = DataSet(args.data_list_target[0],args.data_list_target[1],args.tar_batch_size[0],args.tar_batch_size[1],input_size_target,False)
    src_trainloader = src_dataset.getTrainData()
    tar_trainloader = tar_dataset.getTrainData()

    src_valloader = src_dataset.getValData()
    tar_valloader = tar_dataset.getValData()
    # implement model.optim_parameters(args) to handle different models' lr setting
    # optimizer = optim.SGD(model.optim_parameters(args),
    #                       lr=args.learning_rate, 
    #                       momentum=args.momentum, 
    #                       weight_decay=args.weight_decay)
    
    optimizer = optim.SGD(model.parameters(),
                    lr=args.learning_rate, 
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8, patience = 6, verbose=True)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_seg_D.parameters(), 
                            lr=args.learning_rate_D, 
                            betas=(0.9, 0.99))
    # scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min',factor=0.8, patience = 6, verbose=True)
    optimizer_D.zero_grad()

    optimizer_border_D = optim.Adam(model_fea_D.parameters(), 
                            lr=args.learning_rate_D, 
                            betas=(0.9, 0.99))
    # scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min',factor=0.8, patience = 6, verbose=True)
    optimizer_border_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1
    start = timeit.default_timer()
    best_iou = 0
    best_id = 0
    for epoch in range(args.EPOCH):
        print("training epoch:",epoch)
        # seg model
        loss_seg_value = 0
        loss_border_value = 0

        tar_seg_d_adv_loss_value = 0
        src_seg_d_loss_value = 0
        tar_seg_d_loss_value = 0
        

        tar_fea_d_adv_loss_value = 0
        src_fea_d_loss_value = 0
        tar_fea_d_loss_value = 0


        damping = 1-epoch/args.EPOCH
        lr = adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate_D(optimizer_D, epoch)

        # train G
        # don't accumulate grads in D
        for param in model_seg_D.parameters():
            param.requires_grad = False
        for param in model_fea_D.parameters():
            param.requires_grad = False
        # train with source
        #_, batch = next(trainloader_iter)
        tar_trainloader_iter = enumerate(tar_trainloader)
        for i,batch in enumerate(tqdm(src_trainloader)):
            src_img, labels, edge = batch
            src_img = src_img.cuda()
            labels = labels.cuda()
            edge = edge.cuda()
            src_pred, border_out, src_fea = model(src_img)
            pred = interp(src_pred)
            # seg loss
            loss_seg = loss_calc(pred, labels)
            # border loss
            loss_border = border_loss_calc(border_out,edge)
            loss_seg_value += loss_seg.item()
            loss_border_value += loss_border.item()
            model_loss = loss_seg + 0.2*loss_border
            model_loss.backward(retain_graph=True)
            


            # train with target
            _, batch = next(tar_trainloader_iter)
            tar_img ,tar_label, _= batch
            tar_img = tar_img.cuda()
            tar_label = tar_label.cuda()
            # 目标域输出中间特征
            pred_target, _, tar_fea = model(tar_img)  
            pred_target = interp_target(pred_target)
            # 训练分割网络，使得分割结果让辨别器鉴别为源域分割结果
            tar_seg_d_out = model_seg_D(F.softmax(pred_target, dim=1))
            tar_seg_d_adv_loss = bce_loss(tar_seg_d_out, torch.FloatTensor(tar_seg_d_out.data.size()).fill_(source_label).cuda())
            tar_seg_d_adv_loss = tar_seg_d_adv_loss * args.lambda_adv_target1 * damping
            tar_seg_d_adv_loss.backward(retain_graph=True)
            tar_seg_d_adv_loss_value += tar_seg_d_adv_loss.item()
            # 训练分割网络，使得中间特征输入鉴别器鉴别为源域结果
            tar_fea_d_out = model_fea_D(tar_fea)
            tar_fea_d_adv_loss = bce_loss(tar_fea_d_out,torch.FloatTensor(tar_fea_d_out.data.size()).fill_(source_label).cuda())
            tar_fea_d_adv_loss = tar_fea_d_adv_loss * args.lambda_adv_target1 * damping
            tar_fea_d_adv_loss.backward(retain_graph=True)
            tar_fea_d_adv_loss_value += tar_fea_d_adv_loss.item()

            # train D
            # bring back requires_grad
            for param in model_seg_D.parameters():
                param.requires_grad = True
            for param in model_fea_D.parameters():
                param.requires_grad = True
            # train with source
            src_pred = src_pred.detach()
            src_seg_d_out = model_seg_D(F.softmax(src_pred, dim=1))
            src_seg_d_loss = bce_loss(src_seg_d_out, torch.FloatTensor(src_seg_d_out.data.size()).fill_(source_label).to(device))
            src_seg_d_loss = src_seg_d_loss / 2
            src_seg_d_loss.backward(retain_graph=True)
            src_seg_d_loss_value += src_seg_d_loss.item()

            src_fea = src_fea.detach()
            src_fea_d_out = model_fea_D(src_fea)
            src_fea_d_loss = bce_loss(src_fea_d_out, torch.FloatTensor(src_fea_d_out.data.size()).fill_(source_label).to(device))
            src_fea_d_loss = src_fea_d_loss / 2
            src_fea_d_loss.backward(retain_graph=True)
            src_fea_d_loss_value += src_fea_d_loss.item()
            # # 损失求和
            # loss_D1_loss_D1_border = loss_D1 + loss_D1_border
            # loss_D1_loss_D1_border.backward()

            # train with target
            pred_target = pred_target.detach()
            tar_seg_d_out = model_seg_D(F.softmax(pred_target, dim=1))
            tar_seg_d_loss = bce_loss(tar_seg_d_out, torch.FloatTensor(tar_seg_d_out.data.size()).fill_(target_label).to(device))
            tar_seg_d_loss = tar_seg_d_loss / 2
            tar_seg_d_loss.backward(retain_graph=True)
            tar_seg_d_loss_value += tar_seg_d_loss.item()
            

            tar_fea = tar_fea.detach()
            tar_fea_d_out = model_fea_D(tar_fea)
            tar_fea_d_loss = bce_loss(tar_fea_d_out, torch.FloatTensor(tar_fea_d_out.data.size()).fill_(target_label).to(device))
            tar_fea_d_loss = tar_fea_d_loss / 2
            tar_fea_d_loss.backward()
            tar_fea_d_loss_value += tar_fea_d_loss.item()
            
            optimizer.step()
            optimizer_D.step()
            optimizer_border_D.step()

            optimizer.zero_grad()
            optimizer_D.zero_grad()
            optimizer_border_D.zero_grad()

            current = timeit.default_timer()
        # scheduler.step(loss_seg_value/(i+1))
        # scheduler_D.step(loss_D_value/(i+1))



        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars("DIS_SEG LOSS", {"tar_seg_d_adv_loss_value": tar_seg_d_adv_loss_value/(i+1), "src_seg_d_loss_value": src_seg_d_loss_value/(i+1),"tar_seg_d_loss_value":tar_seg_d_loss_value/(i+1)}, epoch)
        writer.add_scalars("DIS_FEA LOSS", {"tar_fea_d_adv_loss_value": tar_fea_d_adv_loss_value/(i+1), "src_fea_d_loss_value": src_fea_d_loss_value/(i+1),"tar_fea_d_loss_value":tar_fea_d_loss_value/(i+1)}, epoch)
        writer.add_scalars("SEG LOSS", {"SEG": loss_seg_value/(i+1)}, epoch)
        model.eval()
        IOU_SRC = []
        IOU_TAR = []
        for i,batch in enumerate(tqdm(src_valloader)):
            src_img, labels, edge = batch
            src_img = src_img.cuda()
            labels = labels.cuda()
            edge = edge.cuda()
            pred_pic_source, border_pred_pic_source, _ = model(src_img)
            pred_pic_source = torch.argmax(F.softmax(pred_pic_source,dim = 1),dim=1).squeeze()
            border_pred_pic_source = torch.argmax(F.softmax(border_pred_pic_source,dim = 1),dim=1).squeeze()

            me = metric(pred_pic_source,labels,args.label)
            iou = list(me.iou())
            IOU_SRC.append(iou)
            src_save_path = os.path.join(save_dir,"output_pic","epoch_"+str(epoch),"src")
            savePic2(src_img[0],labels[0],edge[0],pred_pic_source,border_pred_pic_source,args.color_dict,src_save_path,writer,epoch,"src")
        for i,batch in enumerate(tqdm(tar_valloader)):
            tar_img, labels,edge= batch
            tar_img = tar_img.cuda()
            labels = labels.cuda()
            pred_pic_target, border_pred_pic_target,_ = model(tar_img)
            pred_pic_target = torch.argmax(F.softmax(pred_pic_target, dim=1),dim = 1).squeeze()
            border_pred_pic_target = torch.argmax(F.softmax(border_pred_pic_target, dim=1),dim = 1).squeeze()
            me = metric(pred_pic_target,labels,args.label)
            iou = list(me.iou())
            IOU_TAR.append(iou)
            tar_save__path = os.path.join(save_dir,"output_pic","epoch_"+str(epoch),"tar")
            savePic2(tar_img.cpu()[0],labels[0],labels[0],pred_pic_target,border_pred_pic_target,args.color_dict,tar_save__path,writer,epoch,"tar")


        this_iou_tar = np.nanmean(np.array(IOU_TAR),axis=0)
        if this_iou_tar[1] > best_iou:
            best_iou = this_iou_tar[1]
            best_id = epoch
        print("The miou of this epoch is: ", this_iou_tar)
        print("The best miou is: ", best_iou,"\nThe best epoch is: ", best_id)

        print('taking snapshot ...')
        torch.save(model.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '.pth'))
        torch.save(model_seg_D.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '_seg_D.pth'))
        torch.save(model_fea_D.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '_fea_D.pth'))

        print('taking snapshot ...')
        torch.save(model.state_dict(), osp.join(save_dir, 'model' + str(args.num_steps_stop) + '.pth'))
        torch.save(model_seg_D.state_dict(), osp.join(save_dir, 'model' + str(args.num_steps_stop) + '_seg_D.pth'))
        torch.save(model_fea_D.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '_fea_D.pth'))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
