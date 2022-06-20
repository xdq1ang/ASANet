from turtle import color
from PIL import Image
import os
import numpy as np
from colorList import color_dict
from colorful import colorful

# colorDict_RGB, colorDict_GRAY = color_dict(r"label",6)
# print(colorDict_RGB)
colorDict_RGB = [[0,0,0],[255,255,255]]
color_rgb  =  { 0:[0,0,0],
                1:[96,128,192]}
def getPath(path):
    pathList = os.listdir(path)
    return pathList

def convertGray(path,save_path,colorDict_RGB):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pathList = getPath(path)
    for each in pathList:
        img = Image.open(os.path.join(path,each)).convert("RGB")
        img_data = np.array(img)
        mask = np.zeros(img_data.shape[:-1])
        where0 = (img_data==colorDict_RGB[0]).all(axis=2)
        where1 = (img_data==colorDict_RGB[1]).all(axis=2)

        mask[where0==True]=0
        mask[where1==True]=1
        mask_path = os.path.join(save_path,each)
        mask_image=Image.fromarray(mask).convert("L")
        mask_image=colorful(mask_image,color_rgb)
        mask_image.save(mask_path)

convertGray(r"label",r"label_gray",colorDict_RGB)
