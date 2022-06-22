import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from PIL import Image

root = r"datasets\ISPRS_semantic_labeling_Vaihingen\image"
save = r"datasets\ISPRS_semantic_labeling_Vaihingen\edge"
nameList = os.listdir(root)
for each in tqdm(nameList):
    img = cv2.imread(os.path.join(root,each))
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 50, 200,apertureSize=3)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # Image.fromarray(canny).show()
    canny[canny==255]=1
    Image.fromarray(canny).save(os.path.join(save,each[:-3]+"png"))



# id255 = np.where(gt == 255)
# no255_gt = np.array(gt)
# no255_gt[id255] = 0
# cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
# cgt = cv2.dilate(cgt, self.edge_kernel)
# cgt[cgt == 255] = 1