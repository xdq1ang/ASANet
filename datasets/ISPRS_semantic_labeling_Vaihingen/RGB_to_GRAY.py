from turtle import color
from PIL import Image
import os
import numpy as np
from utils.colorList import color_dict
from utils.colorful import colorful

#colorDict_RGB, colorDict_GRAY = color_dict(r"ISPRS_semantic_labeling_Vaihingen\train_label",6)
colorDict_RGB = [[255,255,255],
                [255,0,0],
                [0,255,0], 
                [255,255,0], 
                [0,0,255], 
                [0,255,255]]
color_rgb  =  {0:[255,255,255],
                1:[255,0,0],
                2:[0,255,0], 
                3:[255,255,0], 
                4:[0,0,255], 
                5:[0,255,255]}
def getPath(path):
    pathList = os.listdir(path)
    return pathList

def convertGray(path,save_path,colorDict_RGB):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pathList = getPath(path)
    for each in pathList:
        img = Image.open(os.path.join(path,each))
        img_data = np.array(img)
        mask = np.zeros(img_data.shape[:-1])
        where0 = (img_data==colorDict_RGB[0]).all(axis=2)
        where1 = (img_data==colorDict_RGB[1]).all(axis=2)
        where2 = (img_data==colorDict_RGB[2]).all(axis=2)
        where3 = (img_data==colorDict_RGB[3]).all(axis=2)
        where4 = (img_data==colorDict_RGB[4]).all(axis=2)
        where5 = (img_data==colorDict_RGB[5]).all(axis=2)
        mask[where0==True]=0
        mask[where1==True]=1
        mask[where2==True]=2
        mask[where3==True]=3
        mask[where4==True]=4
        mask[where5==True]=5
        mask_path = os.path.join(save_path,each)
        mask_image=Image.fromarray(mask).convert("L")
        mask_image=colorful(mask_image,color_rgb)
        mask_image.save(mask_path)

convertGray(r"ISPRS_semantic_labeling_Vaihingen\train_label",r"ISPRS_semantic_labeling_Vaihingen\train_label_gray",colorDict_RGB)
