import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


root = r"datasets\buildings\label2_gray"
save = r"datasets\buildings\edge"
nameList = os.listdir(root)
for each in tqdm(nameList):
    img = cv2.imread(os.path.join(root,each))
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 150, 300)
    canny [canny==255] = 1
    Image.fromarray(canny).save(os.path.join(save,each))

