set -ex
ls
export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='datasets\cityscapes' \
        --list-path='datasets\cityscapes_list\val.txt' \
        --log-dir=snapshots/klasa \
        --restore-from='pretrained\GTA5_init.pth'

CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='datasets\cityscapes' \
        --list-path='datasets\cityscapes_list\val.txt' \
        --log-dir=logs/GTA5ASA_ST \
        --restore-from='pretrained/GTA5ASA_ST_48.10.pth'

CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='datasets\cityscapes' \
        --list-path='datasets\cityscapes_list\val.txt' \
        --log-dir=logs/GTA5ASA_IT_ST \
        --restore-from='pretrained/GTA5ASA_IT_ST_49.10.pth'