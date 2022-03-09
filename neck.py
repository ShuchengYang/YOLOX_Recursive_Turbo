import torch
from network_blocks import *
#delete
from backbone import *



class Neck(nn.Module):
    def __init__(self, scale=1):
        super(Neck, self).__init__()
        self.seq1 = CBL(384//scale, 128//scale, 1, 1, 0)
        self.seq2 = CBL(768//scale, 256//scale, 1, 1, 0)
        self.seq3 = CBL(1024//scale, 512//scale, 1, 1, 0)
        self.up1 = nn.Sequential(
            CBL(256//scale, 128//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            CBL(512//scale, 256//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, backbone_output):
        n3 = self.seq3(backbone_output[2])
        nm = self.up2(n3)
        n2 = self.seq2(torch.cat([backbone_output[1], nm], dim=1))
        nl = self.up1(n2)
        n1 = self.seq1(torch.cat([backbone_output[0], nl], dim=1))
        return [n1, n2, n3], nl, nm

class RecursiveBAndN(nn.Module):
    def __init__(self, scale=1, map_resol=1):
        super(RecursiveBAndN, self).__init__()
        self.scale = scale
        self.map_resol = map_resol
        self.first_flag = False
        self.nl = torch.zeros(1,128//scale,80//map_resol,80//map_resol)
        self.nm = torch.zeros(1,256//scale,40//map_resol,40//map_resol)
        self.backbone = Backbone(scale=scale)
        self.neck = Neck(scale=scale)

    def forward(self,x):
        if not self.first_flag or x.shape[0] != self.nl.shape[0]:
            self.nl = torch.zeros(x.shape[0], 128//self.scale, 80//self.map_resol, 80//self.map_resol).to(x.device)
            self.nm = torch.zeros(x.shape[0], 256//self.scale, 40//self.map_resol, 40//self.map_resol).to(x.device)
            self.device_flag = True
        backbone_output = self.backbone(x, self.nl, self.nm)
        neck_output, self.nl, self.nm = self.neck(backbone_output)
        return neck_output

if __name__=="__main__":
    model = RecursiveBAndN(scale=4, map_resol=4)
    x = torch.randn(1,3,160,160)
    y = model(x)
    for i in y:
        print(i.shape)
