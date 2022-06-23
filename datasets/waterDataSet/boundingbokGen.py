import os
import numpy as np
from itertools import groupby
from skimage import morphology,measure
from PIL import Image
from scipy import misc
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patch
 
# 因为一张图片里只有一种类别的目标，所以label图标记只有黑白两色
rgbmask = np.array([[0,0,0],[255,255,255]],dtype=np.uint8)
 
# 从label图得到 boundingbox 和图上连通域数量 object_num
def getboundingbox(image):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[image==255]=1
    # 删掉小于10像素的目标
    # mask_without_small = morphology.remove_small_objects(mask,min_size=10,connectivity=2)
    # 连通域标记
    label_image = measure.label(mask)
    #统计object个数
    object_num = len(measure.regionprops(label_image))
    boundingbox = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        boundingbox.append(region.bbox)
    return object_num, boundingbox


 
# 输出成图片查看得到boundingbox效果
imagedir = r'datasets\waterDataSet\label'
saveDir = r'datasets\waterDataSet\boundingbox'
if not os.path.exists(saveDir):
    os.mkdir(saveDir)
for root, _, fnames in sorted(os.walk(imagedir)):
    for fname in sorted(fnames):
        imagepath = os.path.join(root, fname)
        image = imageio.imread(imagepath)
        objectnum, bbox = getboundingbox(image)
        ImageID = fname.split('.')[0]
        
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        for box in bbox:
            rect = patch.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],edgecolor = 'r', linewidth = 1,fill = False)
            ax.add_patch(rect)
        plt.savefig(os.path.join(saveDir,ImageID+'.png'))