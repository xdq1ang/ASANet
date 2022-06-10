from PIL import Image
import os


NameList = os.listdir("WHDLD\AugWater\images")
i = 0
for each in NameList:
    i+=1
    name = each[25: ]
    image_path = "WHDLD\AugWater\images\\"+"water_image_png_original_"+name
    label_path = "WHDLD\AugWater\labels\\"+"_groundtruth_(1)_water_image_png_"+name
    Image.open(image_path).save("WHDLD\AugWater\ReName\images\\"+str(i)+".png")
    Image.open(label_path).save("WHDLD\AugWater\ReName\labels\\"+str(i)+".png")



