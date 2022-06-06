import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from sklearn import metrics
import torch
#计算混淆矩阵
class metric():
    def __init__(self,y_pred,y_true,labels):
        
        self.y_pred=y_pred.cpu().reshape(1,-1).squeeze()
        self.y_true=y_true.cpu().reshape(1,-1).squeeze()
        self.eClasses = torch.unique(y_true)
        self.confusion_matrix=metrics.confusion_matrix(self.y_true,self.y_pred,labels=list(labels))
    #计算iou
    def iou(self):
        confusion_matrix=self.confusion_matrix
        intersection = np.diag(confusion_matrix)#交集
        union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
        IoU = intersection / union #交并比，即IoU
        return IoU
    #计算miou
    def miou(self):
        confusion_matrix=self.confusion_matrix
        intersection = np.diag(confusion_matrix)#交集
        union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
        IoU = intersection / union #交并比，即IoU
        MIoU = np.nanmean(IoU)#计算MIoU
        return MIoU
    
    def f1_recall(self):
        #计算 overall accuracy
        oa = np.diag(self.confusion_matrix).sum() /self.confusion_matrix.sum()
        #计算各类别 accuracy
        acc_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1)  
        # axis 0：gt, axis 1:prediction
        #计算各类别 precision和 recall
        precision_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1)
        recall_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 0)
        #计算各类别 f1-score
        f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
        #计算 mean f1-score
        mf1 = np.nanmean(f1_cls)
        return f1_cls,recall_cls

# y_true=np.array([[[[0,1,0],[0,0,0]]]])
# y_pred=np.array([[[[0,1,0],[0,0,1]]]])
# a=metric(y_pred[0][0],y_true)
# print("IOU:",a.iou())
# print("MIOU:",a.miou())
# print(y_pred.shape)