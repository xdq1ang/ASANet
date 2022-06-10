import numpy as np
from PIL import Image

def colorful(img,color_dict):
    #img:需要上色的图片(class PIL.Image)
    keys=color_dict.keys()
    palette=[[n,n,n] for n in range(max(keys)+1)]
    for kk,vv in color_dict.items():
        palette[kk]=vv
    palette=np.array(palette, dtype='uint8').flatten()
    img.putpalette(palette)
    return img

def getPalette(path,num_classes):
    img = Image.open(path)
    palette = np.array(img.getpalette()[:num_classes*3])
    paletteLen = len(palette)
    palette = palette.reshape(paletteLen//3,3)
    print(palette)

#getPalette(r"WHDLD\ImagesPNG\wh0001.png",7)