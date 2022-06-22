
import os
import random

def create_list(data_path):
    root=data_path
    #image_path = os.path.join(data_path, 'img')
    label_path = os.path.join(data_path, 'label_gray')
    data_names = os.listdir(label_path)
    random.shuffle(data_names)  # 打乱数据
    with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
        with open(os.path.join(data_path, 'val_list.txt'), 'w') as vf:
            for idx, data_name in enumerate(data_names):
                name=data_name[:-3]
                img = os.path.join(root,'image', name+"jpg")
                lab = os.path.join(root,'label_gray',name+"png")######删除 _m
                edge = os.path.join(root,'edge',name+"png")######删除 _m
                if idx % 7 == 0:  # 90%的作为训练集
                    vf.write(img + ' ' + lab +' '+ edge + '\n')
                else:
                    tf.write(img + ' ' + lab +' '+ edge + '\n')
    print('数据列表生成完成')

data_path = r'datasets\ISPRS_semantic_labeling_Vaihingen'
create_list(data_path)  # 生成数据列表