import torch

from backbone import *
from neck import *
from head import *
from yololoss import *



class Yolox(nn.Module):
    def __init__(self, num_cls=80, training=False, scale=4):
        super(Yolox, self).__init__()
        self.training = training
        self.slice = HalfSlice()
        self.slice2 = HalfSlice()
        self.backbone_neck = RecursiveBAndN(scale=scale, map_resol=4)
        self.head = Head(num_classes=num_cls, scale=scale)
        self.criteria = YOLOLoss(num_classes=num_cls, map_resol=4)

    def forward(self, img_tensor, target=None):
        img_tensor=self.slice(img_tensor)
        img_tensor=self.slice2(img_tensor)
        neck_output = self.backbone_neck(img_tensor)
        head_output = self.head(neck_output)
        if self.training:
            return self.criteria(head_output, labels=target)
        else:
            return head_output

if __name__=="__main__":
    m = Yolox(num_cls=1)
    x = torch.randn(1,3,640,640)
    for i in m(x):
        print(i.shape)