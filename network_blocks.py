import torch
from torch import nn


class CBL(nn.Module):
    def __init__(self, *args):
        """args are for convolution step"""
        super(CBL, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(*args),
            nn.BatchNorm2d(args[1]),
            nn.LeakyReLU(negative_slope=1e-1)
        )

    def forward(self, x):
        return self.seq(x)


class ResUnit(nn.Module):
    def __init__(self, *args):
        """args[0] is num of in channels"""
        super(ResUnit, self).__init__()
        self.seq = nn.Sequential(
            CBL(args[0], args[0]//2, 1, 1, 0),
            CBL(args[0]//2, args[0], 3, 1, 1)
        )

    def forward(self, x):
        return x+self.seq(x)


class ResX(nn.Module):
    def __init__(self, n, *args):
        """n is for the number of res unit
           args and kwargs are for the down sampling cbl"""
        super(ResX, self).__init__()
        self.seq = nn.Sequential(
            CBL(*args),
        )
        for i in range(n):
            self.seq.add_module("res unit", ResUnit(args[1]))

    def forward(self, x):
        return self.seq(x)

#
# class CBL5(nn.Module):
#     def __init__(self, *args):
#         """
#         args[0] is num of input channels,
#         args[1] is of output channels
#         """
#         super(CBL5, self).__init__()
#         out2 = int(2*args[1])
#         self.seq = nn.Sequential(
#             CBL(args[0], args[1], 1, 1, 0),
#             CBL(args[1], out2, 3, 1, 1),
#             CBL(out2, args[1], 1, 1, 0),
#             CBL(args[1], out2, 3, 1, 1),
#             CBL(out2, args[1], 1, 1, 0)
#         )
#
#     def forward(self, x):
#         return self.seq(x)

class CBL3(nn.Module):
    def __init__(self, *args):
        """
        args[0] is num of input channels,
        args[1] is of output channels
        """
        super(CBL3, self).__init__()
        out2 = int(2*args[1])
        self.seq = nn.Sequential(
            CBL(args[0], args[1], 1, 1, 0),
            CBL(args[1], out2, 3, 1, 1),
            CBL(out2, args[1], 1, 1, 0),
        )

    def forward(self, x):
        return self.seq(x)


class DHFirst(nn.Module):
    def __init__(self, *args, scale=1):
        """
        args[0] is the input channels
        args[1] is the number of classes (abandoned)
        output all without sigmoid (will be added in train or inference stage)
        """
        super(DHFirst, self).__init__()
        self.stem = CBL(args[0], 256//scale, 1, 1, 0)
        self.box_conf_stem = nn.Sequential(
            CBL(256//scale, 256//scale, 3, 1, 1),
            CBL(256//scale, 256//scale, 3, 1, 1)
        )
        self.cls_conv = nn.Sequential(
            CBL(256//scale, 256//scale, 3, 1, 1),
            CBL(256//scale, 256//scale, 3, 1, 1),
        )


    def forward(self, neck_output):
        """neck_output is only an element of tensor list neck_outputs
        (batch, c, h, w)
        [(1,128,80,80), (1,256,40,40), (1,512,20,20)]
        """
        neck_output = self.stem(neck_output)
        cls_raw = self.cls_conv(neck_output)
        box_conf_raw = self.box_conf_stem(neck_output)
        return box_conf_raw, cls_raw

# class HalfSlice(nn.Module):
#     def __init__(self, in_channel=3):
#         super().__init__()
#         self.chnnl_adapt = CBL(in_channel, in_channel, 3,2,1)
#
#     def forward(self,x):
#         return self.chnnl_adapt(x)

class HalfSlice(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.norm = nn.LayerNorm(in_channel*4)
        self.chnnl_adapt = nn.Linear(in_channel*4, in_channel)

    def forward(self,x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B H W C
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.chnnl_adapt(x)
        x = x.view(B, H // 2, W // 2, -1).permute(0,3,1,2)
        return x


if __name__ == "__main__":
    x = torch.randn(1,256,40,40)
    m = DHFirst(256,scale=1)
    a, b = m(x)
    print(a.shape, b.shape)

