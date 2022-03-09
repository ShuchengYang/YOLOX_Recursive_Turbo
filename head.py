#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import torch

from backbone import Backbone
from neck import Neck
from network_blocks import *

class Head(nn.Module):
    def __init__(
        self,
        num_classes=80,
        scale=1,
        in_cha=[128, 256, 512],
    ):
        super().__init__()
        in_channels = [x//scale for x in in_cha]
        self.n_anchors = 1 #TODO ONLY USED IN INIT CAN BE DELETED
        self.num_classes = num_classes

        self.pre_heads = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()

        for i in range(len(in_channels)):
            self.pre_heads.append(DHFirst(in_channels[i],scale=scale))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256//scale),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256//scale),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256//scale),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )


    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, neck_outputs):
        outputs = []
        for i, x in enumerate(neck_outputs):
            self.pre_heads[i].to(x.device)
            self.reg_preds[i].to(x.device)
            self.obj_preds[i].to(x.device)
            self.cls_preds[i].to(x.device)
            reg_feat, cls_feat = self.pre_heads[i](x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            cls_output = self.cls_preds[i](cls_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs

if __name__== "__main__":

    b1 = Backbone()
    b2 = Backbone(scale=2)
    n1 = Neck()
    n2 = Neck(scale=2)
    h1 = Head()
    h2 = Head(scale=2)
    x = torch.randn(1,3,320,320)
    y = h1(n1(b1(x)))
    z = h2(n2(b2(x)))
    for i,j in zip(y,z):
        print(i.shape, j.shape)