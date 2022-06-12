from webbrowser import BackgroundBrowser
import numpy as np
import pandas as pd
import os
import torch 
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)
from torchvision import transforms




def savePred(pred,eval_path):
    pred = torch.nn.Softmax(dim = 1)(pred)
    #pred = transforms.Resize(15)(pred)
    predSavePath = os.path.join(eval_path,"error_pic")
    if not os.path.exists(predSavePath):
        os.makedirs(predSavePath)
    name = str(len(os.listdir(predSavePath))+1)+".png"
    savePath = os.path.join(predSavePath,name)
    pred0 = pred[0][0].detach().numpy()
    pred1 = pred[0][1].detach().numpy()
    cha = np.abs(pred1 - pred0)
    fig = plt.figure(dpi=500)
    sns_plot = sns.heatmap(cha,xticklabels =False,yticklabels =False, cbar=False,annot=False,square=True,vmax=1,vmin=0)
    fig.savefig(savePath, bbox_inches='tight', pad_inches=0) # 减少边缘空白
    # plt.show()
    plt.close()


    
