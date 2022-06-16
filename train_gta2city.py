import sys
import os
import os.path as osp
import random
from pprint import pprint
import timeit
from tkinter import Image
from matplotlib import transforms
from tqdm import tqdm
from utils.metrics import metric

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils.dataset2d5 import SEGData1
from networks.deeplab import Deeplab_Res101
from networks.unet import UNet
from utils.SurfaceLoss import SurfaceLoss,class2one_hot,one_hot2dist
from networks.discriminator import EightwayASADiscriminator
from networks.discriminator1 import FCDiscriminator
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import cityscapesDataSet
from options import gta5asa_opt
from utils.utils import colorize_mask
from utils.showPic import showPic,savePic1
from torch.utils.tensorboard import SummaryWriter
args = gta5asa_opt.get_arguments()

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

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    criterion = CrossEntropy2d().to(device)
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
    if args.backbone == 'resnet':
        model = Deeplab_Res101(num_classes=args.num_classes)
    elif args.backbone == "unet":
        model = UNet(n_classes=args.num_classes)
    if args.resume:
        print("Resuming from ==>>", args.resume)
        state_dict = torch.load(args.resume,map_location=torch.device(device))
        model.load_state_dict(state_dict)
    model.train()
    model.to(device)
    cudnn.benchmark = True

    # init D
    model_D = EightwayASADiscriminator(num_classes=args.num_classes)
    # model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D.train()
    model_D.to(device)

    log = open(os.path.join(save_dir,"model_structure.txt"), mode='a',encoding='utf-8')
    print("model: ", model,"\n", file=log)
    print("model_D: ", model_D, file=log)
    log.close()

    pprint(vars(args))
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    src_dataset = DataSet(args.data_list[0],args.data_list[1],args.src_batch_size[0],args.src_batch_size[1],input_size)
    tar_dataset = DataSet(args.data_list_target[0],args.data_list_target[1],args.tar_batch_size[0],args.tar_batch_size[1],input_size_target)
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

    optimizer_D = optim.Adam(model_D.parameters(), 
                            lr=args.learning_rate_D, 
                            betas=(0.9, 0.99))
    # scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min',factor=0.8, patience = 6, verbose=True)
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weight_bce_loss = WeightedBCEWithLogitsLoss()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1
    start = timeit.default_timer()
    best_iou = 0
    for epoch in range(args.EPOCH):
        print("training epoch:",epoch)
        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0
        damping = (1 - epoch/args.EPOCH)
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, epoch)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, epoch)

        # train G
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        # train with source
        #_, batch = next(trainloader_iter)
        tar_trainloader_iter = enumerate(tar_trainloader)
        for i,batch in enumerate(tqdm(src_trainloader)):
            src_img, labels = batch
            src_img = src_img.cuda()
            labels = labels.cuda()
            pred = model(src_img)
            pred = interp(pred)
            loss_seg = loss_calc(pred, labels)
            # loss_seg = loss_cal_surfaceLoss(pred,labels)
            loss_seg.backward()
            loss_seg_value += loss_seg.item()

            # train with target
            _, batch = next(tar_trainloader_iter)
            tar_img ,tar_label = batch
            tar_img = tar_img.cuda()
            tar_label = tar_label.cuda()
            pred_target = model(tar_img)
            pred_target = interp_target(pred_target)
            D_out = model_D(F.softmax(pred_target, dim=1))
            loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
            loss_adv = loss_adv_target * args.lambda_adv_target1 * damping
            loss_adv.backward()
            loss_adv_target_value += loss_adv_target.item()
            # train D
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True
            # train with source
            pred = pred.detach()
            D_out = model_D(F.softmax(pred, dim=1))
            loss_D1 = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss_D1 = loss_D1 / 2
            loss_D1.backward()
            loss_D_value += loss_D1.item()
            # train with target
            pred_target = pred_target.detach()
            D_out1 = model_D(F.softmax(pred_target, dim=1))
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
            loss_D1 = loss_D1 / 2
            loss_D1.backward()
            loss_D_value += loss_D1.item()
            optimizer.step()
            optimizer_D.step()
            current = timeit.default_timer()
        # scheduler.step(loss_seg_value/(i+1))
        # scheduler_D.step(loss_D_value/(i+1))

        print("loss_seg1 = {}  loss_adv1 = {}, loss_D1 = {}".format(loss_seg_value/(i+1),  loss_adv_target_value/(i+1), loss_D_value/(i+1)))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars("Loss", {"Seg": loss_seg_value/(i+1), "Adv": loss_adv_target_value/(i+1), "Disc": loss_D_value/(i+1)}, epoch)
        model.eval()
        IOU_SRC = []
        IOU_TAR = []
        for i,batch in enumerate(tqdm(src_valloader)):
            src_img, labels = batch
            src_img = src_img.cuda()
            labels = labels.cuda()
            pred_pic_source = model(src_img)
            pred_pic_source = torch.argmax(F.softmax(pred_pic_source,dim = 1),dim=1).squeeze()
            me = metric(pred_pic_source,labels,[0,1])
            iou = list(me.iou())
            IOU_SRC.append(iou)
            src_save_path = os.path.join(save_dir,"output_pic","epoch_"+str(epoch),"src")
            savePic1(src_img.cpu()[0],labels[0],pred_pic_source,args.color_dict,src_save_path,writer,epoch,"src")

        for i,batch in enumerate(tqdm(tar_valloader)):
            tar_img, labels = batch
            tar_img = tar_img.cuda()
            labels = labels.cuda()
            pred_pic_target = model(tar_img)
            pred_pic_target = torch.argmax(F.softmax(pred_pic_target, dim=1),dim = 1).squeeze()
            me = metric(pred_pic_target,labels,[0,1])
            iou = list(me.iou())
            IOU_TAR.append(iou)
            tar_save__path = os.path.join(save_dir,"output_pic","epoch_"+str(epoch),"tar")
            savePic1(tar_img.cpu()[0],labels[0],pred_pic_target,args.color_dict,tar_save__path,writer,epoch,"tar")


        this_iou_tar = np.nanmean(np.array(IOU_TAR),axis=0)
        if this_iou_tar[1] > best_iou:
            best_iou = this_iou_tar[1]
        print("The miou of this epoch is: ", this_iou_tar)
        print("The best miou is: ", best_iou,"\nThe best epoch is: ", epoch)

        print('taking snapshot ...')
        torch.save(model.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '.pth'))
        torch.save(model_D.state_dict(), osp.join(save_dir, 'model' + str(epoch) + '_D.pth'))

        print('taking snapshot ...')
        torch.save(model.state_dict(), osp.join(save_dir, 'model' + str(args.num_steps_stop) + '.pth'))
        torch.save(model_D.state_dict(), osp.join(save_dir, 'model' + str(args.num_steps_stop) + '_D.pth'))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
