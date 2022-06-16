
import os
import random

def create_list(data_path):
    root=data_path
    label_path = os.path.join(data_path, 'label2_gray')
    data_names = os.listdir(label_path)
    random.shuffle(data_names)  # 打乱数据
    with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
        with open(os.path.join(data_path, 'val_list.txt'), 'w') as vf:
            for idx, data_name in enumerate(data_names):
                name=data_name[:-3]
                img = os.path.join(root,'image', name+"png")
                lab = os.path.join(root,'label2_gray',name+"png")
                edge = os.path.join(root,'edge',name+"png")
                if idx % 5 == 0: 
                    vf.write(img + ' ' + lab + ' ' + edge + '\n')
                else:
                    tf.write(img + ' ' + lab + ' ' + edge + '\n')
    print('数据列表生成完成')

data_path = r'datasets\buildings'
create_list(data_path)  # 生成数据列表