from matplotlib import pyplot as plt
import os
from utils.colorful import colorful
from torchvision import transforms
from utils.utils import colorize_mask
import torch



def showPic(img,label,pre,color_dict,savePath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    num=len(os.listdir(savePath))
    num=num+1
    label = colorful(label,color_dict)
    pre = colorful(pre,color_dict)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img.permute(1,2,0))
    plt.title('img')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(label)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(pre)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(savePath+"/"+str(num)+".png")
    plt.close()
    #plt.show()

def savePic(img,lab,pre,savePath,summary, epoch):
    imgSavePath = os.path.join(savePath,"img")
    labSavePath = os.path.join(savePath,"lab")
    preSavePath = os.path.join(savePath,"pre")
    comparePath = os.path.join(savePath,"com")
    if not os.path.exists(imgSavePath):
        os.makedirs(imgSavePath)
    if not os.path.exists(labSavePath):
        os.makedirs(labSavePath)
    if not os.path.exists(preSavePath):
        os.makedirs(preSavePath)
    if not os.path.exists(comparePath):
        os.makedirs(comparePath)
    num=len(os.listdir(imgSavePath))
    num=num+1
    img_pic = transforms.ToPILImage()(img.squeeze(dim=0).cpu())
    # pre_pic = transforms.ToPILImage()(pre.squeeze(dim=0).cpu().type(torch.uint8))
    # lab_pic = transforms.ToPILImage()(lab.squeeze(dim=0).cpu().type(torch.uint8))
    pre_pic = colorize_mask(pre.cpu().detach().numpy())
    lab_pic = colorize_mask(lab.cpu().detach().numpy())
    img_pic.save(os.path.join(imgSavePath,str(num)+".png"))
    lab_pic.save(os.path.join(labSavePath,str(num)+".png"))
    pre_pic.save(os.path.join(preSavePath,str(num)+".png"))

    figure = plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img_pic)
    plt.title('img')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,2)
    plt.imshow(lab_pic)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,1,2)
    plt.imshow(pre_pic)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(comparePath+"/"+str(num)+".png")
    if(num<=20):
        summary.add_figure(str(num), figure = figure, global_step = epoch)
    plt.close()

def savePic1(img,lab,pre,COLOR_DICT,savePath,summary, epoch,tag):
    imgSavePath = os.path.join(savePath,"img")
    labSavePath = os.path.join(savePath,"lab")
    preSavePath = os.path.join(savePath,"pre")
    comparePath = os.path.join(savePath,"com")
    if not os.path.exists(imgSavePath):
        os.makedirs(imgSavePath)
    if not os.path.exists(labSavePath):
        os.makedirs(labSavePath)
    if not os.path.exists(preSavePath):
        os.makedirs(preSavePath)
    if not os.path.exists(comparePath):
        os.makedirs(comparePath)
    num=len(os.listdir(imgSavePath))
    num=num+1
    if (num == 50):
        return
    img_pic = transforms.ToPILImage()(img.cpu())
    pre_pic = transforms.ToPILImage()(pre.cpu().type(torch.uint8))
    lab_pic = transforms.ToPILImage()(lab.cpu().type(torch.uint8))
    pre_pic = colorful(pre_pic,COLOR_DICT)
    lab_pic = colorful(lab_pic,COLOR_DICT)
    img_pic.save(os.path.join(imgSavePath,str(num)+".png"))
    lab_pic.save(os.path.join(labSavePath,str(num)+".png"))
    pre_pic.save(os.path.join(preSavePath,str(num)+".png"))

    figure = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img_pic)
    plt.title('img')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(lab_pic)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(pre_pic)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(comparePath+"/"+str(num)+".png")
    if(num<=50):
        summary.add_figure(tag+"_"+str(num), figure = figure, global_step = epoch)
    plt.close()

def savePic2(img,lab,edge,pre,pre2,COLOR_DICT,savePath,summary, epoch,tag):
    imgSavePath = os.path.join(savePath,"img")
    labSavePath = os.path.join(savePath,"lab")
    edgeSavePath = os.path.join(savePath,"edge")
    preSavePath = os.path.join(savePath,"pre")
    pre2SavePath = os.path.join(savePath,"border_pre")
    comparePath = os.path.join(savePath,"com")
    if not os.path.exists(imgSavePath):
        os.makedirs(imgSavePath)
    if not os.path.exists(labSavePath):
        os.makedirs(labSavePath)
    if not os.path.exists(edgeSavePath):
        os.makedirs(edgeSavePath)
    if not os.path.exists(preSavePath):
        os.makedirs(preSavePath)
    if not os.path.exists(pre2SavePath):
        os.makedirs(pre2SavePath)
    if not os.path.exists(comparePath):
        os.makedirs(comparePath)
    num=len(os.listdir(imgSavePath))
    num=num+1
    if (num == 50):
        return
    img_pic = transforms.ToPILImage()(img.cpu())
    pre_pic = transforms.ToPILImage()(pre.cpu().type(torch.uint8))
    pre2_pic = transforms.ToPILImage()(pre2.cpu().type(torch.uint8))
    lab_pic = transforms.ToPILImage()(lab.cpu().type(torch.uint8))
    edge_pic = transforms.ToPILImage()(edge.cpu().type(torch.uint8))
    pre_pic = colorful(pre_pic,COLOR_DICT)
    pre2_pic = colorful(pre2_pic,COLOR_DICT)
    lab_pic = colorful(lab_pic,COLOR_DICT)
    edge_pic = colorful(edge_pic,COLOR_DICT)
    img_pic.save(os.path.join(imgSavePath,str(num)+".png"))
    lab_pic.save(os.path.join(labSavePath,str(num)+".png"))
    edge_pic.save(os.path.join(edgeSavePath,str(num)+".png"))
    pre_pic.save(os.path.join(preSavePath,str(num)+".png"))
    pre2_pic.save(os.path.join(pre2SavePath,str(num)+".png"))

    figure = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img_pic)
    plt.title('img')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,2)
    plt.imshow(lab_pic)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,3)
    plt.imshow(edge_pic)
    plt.title('edge_pic')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,3)
    plt.imshow(pre_pic)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,4)
    plt.imshow(pre2_pic)
    plt.title('predict2')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(comparePath+"/"+str(num)+".png")
    if(num<=50):
        summary.add_figure(tag+"_"+str(num), figure = figure, global_step = epoch)
    plt.close()

    
    