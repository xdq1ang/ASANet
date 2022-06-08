from train_source_model import DataSet
from utils.dataset2d5 import SEGData
from utils.showPic import showPic,savePic
from utils.metrics import metric
from matplotlib import pyplot as plt
from networks.unet import UNet
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import os
from torchvision import transforms

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVEPATH = r"snapshots\preTrainied_unet_ASANet"
DATA_ROOT_PATH = r"datasets\cityscapes"
TRAIN_TXT_PATH = r"datasets\cityscapes\train.txt"
VAL_TXT_PATH = r"datasets\cityscapes\val.txt"
TRAINIED_MODEL_PATH = r"snapshots\preTrainied_unet_ASANet\GTA5KLASA_20000.pth"
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 1
RESIZE = [256,512]
ID_TO_TRAINIED =  {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                    26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
LABEL = LABEL=ID_TO_TRAINIED.values()

class DataSet_cityscapes(DataSet):
    def __init__(self, train_txt_path, val_txt_path, train_batch_size, val_batch_size, resize, ID_TO_TRAINIED,data_root_path):
        super().__init__(train_txt_path, val_txt_path, train_batch_size, val_batch_size, resize, ID_TO_TRAINIED)
        self.data_root_path = data_root_path

    def getValData(self):
        self.val_SEGData = SEGData_cityscapes(self.val_data,self.resize,self.id_to_trainied,self.data_root_path)
        return torch.utils.data.DataLoader(self.val_SEGData, num_workers=4, batch_size=self.val_batch_size, shuffle=False)

class SEGData_cityscapes(SEGData):
    def __init__(self, dataset, resize, id_to_trainid,data_root_path):
        super().__init__(dataset, resize, id_to_trainid)
        self.data_root_path = data_root_path
        self.void_classes = [0, 1, 2, 3, 4, 5,
                             6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19,
                              20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(19)))
    def __getitem__(self, item):
        # 取出图片路径
        img_data = os.path.join(self.data_root_path,self.dataset[item].split(' ')[0])
        label_data = os.path.join(self.data_root_path,self.dataset[item].split(' ')[1].replace('\n',''))
        img = Image.open(img_data)
        label = Image.open(label_data)
        data = np.array(label)
        transform =transforms.Resize(size=self.resize,interpolation=0)
        img = transform(img)
        label = transform(label)

        label = self.encode_segmap(np.array(label, dtype=np.uint8))

        #转化为tensor
        img = transforms.ToTensor()(img)
        label = torch.tensor(np.array(label), dtype=torch.long) 
        return img, label
    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 255
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

def colorize_mask(mask):
    # mask: numpy array of the mask
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
def savePic(img,lab,pre,savePath):
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
    plt.close()

def eval(net,dataset,model_name,savePath):
    print("evaling")
    eval_step = dataset.getValStep()
    val_epoch_score = []
    val_iou = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=eval_step) as t2:
            for i, (img,label) in enumerate(dataset.getValData()):   
                img = img.to(device) 
                label = label.squeeze(dim = 1).cuda()
                if model_name == "LANet":
                    pred,aux_out = net (img) 
                    predict=torch.argmax(pred, dim=1)  
                elif model_name == "PSPNet" or model_name== "DensePSPNet" or model_name == "ResNet":
                    pred = net(img, label)
                    predict=torch.argmax(pred, dim=1)
                else:
                    pred=net(img)
                    predict=torch.argmax(pred, dim=1)
                #性能评估
                me = metric(predict,label,LABEL)
                iou = list(me.iou())
                val_iou.append(iou)
                score = me.miou()
                val_epoch_score.append(score)
                savePic(img.cpu()[0],label[0],predict[0],savePath)
                t2.update()
    val_iou = np.nanmean(np.array(val_iou),axis=0)



def main():
    dataset = DataSet_cityscapes(TRAIN_TXT_PATH,VAL_TXT_PATH,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,RESIZE,ID_TO_TRAINIED,DATA_ROOT_PATH)
    saved_state_dict = torch.load(TRAINIED_MODEL_PATH)
    net = UNet(n_channels=3,n_classes=len(LABEL)).to(device)
    net.load_state_dict(saved_state_dict)
    net.eval()
    eval(net,dataset,"UNet",SAVEPATH)

if __name__ == "__main__":
    main()