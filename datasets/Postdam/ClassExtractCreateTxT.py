import os
import random

def create_list(data_path):
    trian_image_path = os.path.join(data_path,"train_image")
    trian_label_path = os.path.join(data_path,"train_label_gray")
    trian_dsm_path = os.path.join(data_path,"train_dsm")
    val_image_path = os.path.join(data_path,"val_image")
    val_dsm_path = os.path.join(data_path,"val_dsm")
    train_image_names = os.listdir(trian_image_path)
    val_image_names = os.listdir(val_image_path)
    with open(os.path.join(data_path, 'train_list.txt'), 'w') as tf:
        for idx, data_name in enumerate(train_image_names):
            data_name=data_name.split("_")[-1]
            train_data_name = "top_mosaic_09cm_"+data_name
            train_dsm_name = "dsm_09cm_matching_"+data_name
            img = os.path.join(trian_image_path, train_data_name)
            lab = os.path.join(trian_label_path, train_data_name)
            dsm = os.path.join(trian_dsm_path, train_dsm_name)
            tf.write(img + ' ' + dsm + ' ' + lab +'\n')

    with open(os.path.join(data_path, 'val_list.txt'), 'w') as tv:
        for idx, data_name in enumerate(val_image_names):
            data_name=data_name.split("_")[-1]
            val_data_name = "top_mosaic_09cm_"+data_name
            val_dsm_name = "dsm_09cm_matching_"+data_name
            img = os.path.join(val_image_path, val_data_name)
            dsm = os.path.join(val_dsm_path, val_dsm_name)
            tv.write(img + ' ' + dsm +'\n')
    print('数据列表生成完成')

data_path = r'ISPRS_semantic_labeling_Vaihingen'
create_list(data_path)  # 生成数据列表