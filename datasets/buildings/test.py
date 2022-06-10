from PIL import Image
import numpy as np

lab = Image.open(r"buildings\label2\1.png")
lab_data = np.array(lab)
print(lab_data.shape)