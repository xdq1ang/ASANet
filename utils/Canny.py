import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
img = cv2.imread(r'datasets\buildings\image\8.png')
# img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

# 高斯噪声
noise_img = skimage.util.random_noise(img.copy(), mode='gaussian', 
                                      seed=None, clip=True, mean=0, var=0.0016)
noise_img = np.uint8(noise_img*255)

# 拉普拉斯边缘检测
lap = cv2.Laplacian(noise_img, cv2.CV_64F) # 拉普拉斯边缘检测
lap = np.uint8(np.absolute(lap)) # 对lap去绝对值
# 对二值图像进行反转，黑白颠倒
lap = cv2.bitwise_not(lap)

# Canny边缘检测
canny = cv2.Canny(img, 150, 300)
# 对二值图像进行反转，黑白颠倒
canny = cv2.bitwise_not(canny)


# 展示图像
plt.subplot(221)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("(A)")

plt.subplot(222)
plt.imshow(noise_img,cmap=plt.cm.gray)
plt.title("(B)")

plt.subplot(223)
plt.imshow(lap, cmap=plt.cm.gray)
plt.title("(C)")

plt.subplot(224)
plt.imshow(canny, cmap=plt.cm.gray)
plt.title("(D)")

plt.tight_layout()
plt.savefig("fig.png")