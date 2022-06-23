
import os
import random

def create_list(data_path):
    root=data_path
    #image_path = os.path.join(data_path, 'img')
    label_path = os.path.join(data_path, 'labelAug')
    data_names = os.listdir(label_path)
    random.shuffle(data_names)  # 打乱数据
    with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
        with open(os.path.join(data_path, 'val_list.txt'), 'w') as vf:
            for idx, data_name in enumerate(data_names):
                name=data_name[0:-4].split("label")
                image_name = name[0]+".jpeg"
                label_name = data_name
                img = os.path.join(root,'imgAug', image_name)
                lab = os.path.join(root,'labelAug',label_name)
                edge = os.path.join(root,'edge',label_name)
                if idx % 7 == 0:  # 90%的作为训练集
                    vf.write(img + ' ' + lab + ' ' + edge + '\n')
                else:
                    tf.write(img + ' ' + lab + ' ' + edge + '\n')
    print('数据列表生成完成')

data_path = r'datasets\waterDataSet'
create_list(data_path)  # 生成数据列表