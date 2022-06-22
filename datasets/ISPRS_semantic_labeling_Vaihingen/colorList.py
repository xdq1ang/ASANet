#创建颜色字典
from PIL import Image
import cv2
import os
import pathlib
import numpy as np

def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的所有文件名
    ImageNameList = os.listdir(labelFolder)  # ['105.tif', '20.tif', '77.tif',... '116.tif']
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(
                np.uint32)  # img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if (len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        #color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        color_RGB = [int(color[6: 9]),int(color[3: 6]),int(color[0: 3])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY
#标签映射到数字
def colorMapping(label_data, colorDict_RGB):
    label_data=np.array(label_data)

    #如果是灰度图则转换为三通道
    if len(label_data.shape)==2:
        label_data=cv2.cvtColor(label_data,cv2.COLOR_GRAY2BGR)
    
    classNum = colorDict_RGB.shape[0]
    w,h,c =label_data.shape
    new_label=np.zeros((w,h,classNum))
    for channel in range(classNum):
        if str(colorDict_RGB[channel])!="null":
            new_label[:,:,channel][(label_data==colorDict_RGB[channel])[:,:,0]]=1

    return new_label
colorDict_RGB, colorDict_GRAY = color_dict("ISPRS_semantic_labeling_Vaihingen\label_gray",10)
print(colorDict_RGB)
print(colorDict_GRAY)