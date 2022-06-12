import numpy as np
from torchvision import transforms
import torch
from PIL import Image

def colorful(img,color_dict):
    #img:需要上色的图片(class PIL.Image)
    img = Image.fromarray(img.astype("uint8"))
    keys=color_dict.keys()
    palette=[[n,n,n] for n in range(max(keys)+1)]
    for kk,vv in color_dict.items():
        palette[kk]=vv
    palette=np.array(palette, dtype='uint8').flatten()
    img.putpalette(palette)
    return img