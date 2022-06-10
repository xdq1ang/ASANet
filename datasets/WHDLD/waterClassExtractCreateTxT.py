import os
import random

def create_list(data_path):
    root=data_path
    label_path = os.path.join(data_path,'water_label')
    data_names = os.listdir(label_path)
    random.shuffle(data_names)  # 打乱数据
    with open(os.path.join(data_path, 'train_list_water.txt'), 'w') as tf:
        with open(os.path.join(data_path, 'val_list_water.txt'), 'w') as vf:
            for idx, data_name in enumerate(data_names):
                data_name=data_name[:-3]+"png"
                img = os.path.join(root,'water_image_png', data_name)
                lab = os.path.join(root,'water_label', data_name.split('.')[0]+".png")######删除 _m
                if idx % 7 == 0:  # 90%的作为训练集
                    vf.write(img + ' ' + lab + '\n')
                else:
                    tf.write(img + ' ' + lab + '\n')
    print('数据列表生成完成')

data_path = r'WHDLD'
create_list(data_path)  # 生成数据列表