def EXNUM(oldpath,savepath):
    with open(savepath,"w") as f:
        for line in open(oldpath):
            train=(line.split(' ')[0].split('\\')[2].split('.')[0])
            f.write(train+'\n')
EXNUM(r"waterDataSetAug\train_list.txt",r"waterDataSetAug\train.txt")
EXNUM(r"waterDataSetAug\val_list.txt",r"waterDataSetAug\val.txt")