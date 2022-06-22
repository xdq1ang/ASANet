import numpy as np

def colorful(img,color_dict):
    #img:需要上色的图片(class PIL.Image)
    keys=color_dict.keys()
    palette=[[n,n,n] for n in range(max(keys)+1)]
    for kk,vv in color_dict.items():
        palette[kk]=vv
    palette=np.array(palette, dtype='uint8').flatten()
    img.putpalette(palette)
    return img