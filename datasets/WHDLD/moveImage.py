from PIL import Image
import os


root = "WHDLD"

NameList = os.listdir("WHDLD\water_image")


i = 0
for each in NameList:
    i+=1
    image_name = each[0: -4]
    Image.open("WHDLD\water_image"+"\\"+each).save("WHDLD\water_image_png"+"\\"+image_name+".png")



