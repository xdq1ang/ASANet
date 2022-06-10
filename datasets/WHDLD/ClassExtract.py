from PIL import  Image
import matplotlib as mpl
import numpy as np
import os
from tqdm import tqdm
from collections import  Counter
from colorful import colorful

#                                     提取森林
#                          陆地         建筑        森林          水体                
#colorDict_RGB=np.array([[94,65,47],[193,41,46],[88,129,51],[57,160,237]])

def ClassExtract(class_color,old_label_folder,new_label_folder):    
    # class_color:        待提取地物像素值
    # old_label_folder:   原始多分类label路径
    # new_label_folder    新label路径

    listdir=os.listdir(old_label_folder)
    for each in tqdm(listdir):
        each_path=os.path.join(old_label_folder,each)
        mask = Image.open(each_path)     
        mask_data = np.array(mask)
        new_mask=np.zeros_like(mask_data)                               #创建新标签（二分类）
        mask=mask_data==class_color#灰度值                                #返回被提取地物的位置信息
        count = Counter(mask.flatten())
        water_ratio = count[True]/(count[True]+count[False])
        # 过滤掉水体==0   水体>0.85的图片                                             
        if water_ratio < 0.02 or water_ratio > 0.95:
            continue                                                     
        new_mask[mask_data==class_color] = 1
        new_mask=colorful(Image.fromarray(np.uint8(new_mask)),color_dict = { 0:[0,0,0], 1:[255,0,0] })
        new_mask.save(os.path.join(new_label_folder,each))                #保存新标签



ClassExtract(1,r"ImagesPNG",r"buildings")


