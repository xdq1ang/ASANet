from networks.dfn.dfn import DFN
from torch import nn
import torch
from networks.dfn.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d

criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=255)
aux_criterion = SigmoidFocalLoss(ignore_label=255, gamma=2.0, alpha=0.25)
model = DFN(10, criterion=criterion,
            aux_criterion=aux_criterion, alpha=0.1,
            pretrained_model=None,
            norm_layer=nn.BatchNorm2d)

model.eval()
image = torch.autograd.Variable(torch.randn(2, 3, 512, 512), volatile=True)
res1, res2 = model(data=image)
print (res1.size(), res2.size())