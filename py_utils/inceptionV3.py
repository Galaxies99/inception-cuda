# Original Code by https://github.com/zt1112/pytorch_inceptionv3/blob/master/inception3.py
# Make critical modifications only for debug use.
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import onnx
import json
import onnx.numpy_helper as numpy_helper


def get_seperate_weights(model, name):
    [w] = [t for t in model.graph.initializer if t.name == name]
    return numpy_helper.to_array(w).copy()


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, transform_input=True):
        super(Inception3, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.group1 = nn.Linear(2048, num_classes)
    
    def set_params(self, dict):
        self.Conv2d_1a_3x3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer1']['c_1_w']))
        self.Conv2d_1a_3x3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer1']['c_1_b']))
        self.Conv2d_2a_3x3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer1']['c_2_w']))
        self.Conv2d_2a_3x3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer1']['c_2_b']))
        self.Conv2d_2b_3x3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer1']['c_3_w']))
        self.Conv2d_2b_3x3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer1']['c_3_b']))
        self.Conv2d_3b_1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer1']['c_4_w']))
        self.Conv2d_3b_1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer1']['c_4_b']))
        self.Conv2d_4a_3x3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer1']['c_5_w']))
        self.Conv2d_4a_3x3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer1']['c_5_b']))

        self.Mixed_5b.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way1_w']))
        self.Mixed_5b.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way1_b']))
        self.Mixed_5b.branch5x5_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way2_1_w']))
        self.Mixed_5b.branch5x5_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way2_1_b']))
        self.Mixed_5b.branch5x5_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way2_2_w']))
        self.Mixed_5b.branch5x5_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way2_2_b']))
        self.Mixed_5b.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_1_w']))
        self.Mixed_5b.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_1_b']))
        self.Mixed_5b.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_2_w']))
        self.Mixed_5b.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_2_b']))
        self.Mixed_5b.branch3x3dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_3_w']))
        self.Mixed_5b.branch3x3dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way3_3_b']))
        self.Mixed_5b.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way4_w']))
        self.Mixed_5b.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_1']['way4_b']))

        self.Mixed_5c.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way1_w']))
        self.Mixed_5c.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way1_b']))
        self.Mixed_5c.branch5x5_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way2_1_w']))
        self.Mixed_5c.branch5x5_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way2_1_b']))
        self.Mixed_5c.branch5x5_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way2_2_w']))
        self.Mixed_5c.branch5x5_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way2_2_b']))
        self.Mixed_5c.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_1_w']))
        self.Mixed_5c.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_1_b']))
        self.Mixed_5c.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_2_w']))
        self.Mixed_5c.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_2_b']))
        self.Mixed_5c.branch3x3dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_3_w']))
        self.Mixed_5c.branch3x3dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way3_3_b']))
        self.Mixed_5c.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way4_w']))
        self.Mixed_5c.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_2']['way4_b']))

        self.Mixed_5d.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way1_w']))
        self.Mixed_5d.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way1_b']))
        self.Mixed_5d.branch5x5_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way2_1_w']))
        self.Mixed_5d.branch5x5_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way2_1_b']))
        self.Mixed_5d.branch5x5_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way2_2_w']))
        self.Mixed_5d.branch5x5_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way2_2_b']))
        self.Mixed_5d.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_1_w']))
        self.Mixed_5d.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_1_b']))
        self.Mixed_5d.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_2_w']))
        self.Mixed_5d.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_2_b']))
        self.Mixed_5d.branch3x3dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_3_w']))
        self.Mixed_5d.branch3x3dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way3_3_b']))
        self.Mixed_5d.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way4_w']))
        self.Mixed_5d.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer2_3']['way4_b']))
        
        self.Mixed_6a.branch3x3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer3']['way1_w']))
        self.Mixed_6a.branch3x3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer3']['way1_b']))
        self.Mixed_6a.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_1_w']))
        self.Mixed_6a.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_1_b']))
        self.Mixed_6a.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_2_w']))
        self.Mixed_6a.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_2_b']))
        self.Mixed_6a.branch3x3dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_3_w']))
        self.Mixed_6a.branch3x3dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer3']['way2_3_b']))

        self.Mixed_6b.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way1_w']))
        self.Mixed_6b.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way1_b']))
        self.Mixed_6b.branch7x7_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_1_w']))
        self.Mixed_6b.branch7x7_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_1_b']))
        self.Mixed_6b.branch7x7_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_2_w']))
        self.Mixed_6b.branch7x7_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_2_b']))
        self.Mixed_6b.branch7x7_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_3_w']))
        self.Mixed_6b.branch7x7_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way2_3_b']))
        self.Mixed_6b.branch7x7dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_1_w']))
        self.Mixed_6b.branch7x7dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_1_b']))
        self.Mixed_6b.branch7x7dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_2_w']))
        self.Mixed_6b.branch7x7dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_2_b']))
        self.Mixed_6b.branch7x7dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_3_w']))
        self.Mixed_6b.branch7x7dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_3_b']))
        self.Mixed_6b.branch7x7dbl_4.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_4_w']))
        self.Mixed_6b.branch7x7dbl_4.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_4_b']))
        self.Mixed_6b.branch7x7dbl_5.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_5_w']))
        self.Mixed_6b.branch7x7dbl_5.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way3_5_b']))
        self.Mixed_6b.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way4_w']))
        self.Mixed_6b.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_1']['way4_b']))

        self.Mixed_6c.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way1_w']))
        self.Mixed_6c.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way1_b']))
        self.Mixed_6c.branch7x7_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_1_w']))
        self.Mixed_6c.branch7x7_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_1_b']))
        self.Mixed_6c.branch7x7_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_2_w']))
        self.Mixed_6c.branch7x7_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_2_b']))
        self.Mixed_6c.branch7x7_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_3_w']))
        self.Mixed_6c.branch7x7_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way2_3_b']))
        self.Mixed_6c.branch7x7dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_1_w']))
        self.Mixed_6c.branch7x7dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_1_b']))
        self.Mixed_6c.branch7x7dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_2_w']))
        self.Mixed_6c.branch7x7dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_2_b']))
        self.Mixed_6c.branch7x7dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_3_w']))
        self.Mixed_6c.branch7x7dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_3_b']))
        self.Mixed_6c.branch7x7dbl_4.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_4_w']))
        self.Mixed_6c.branch7x7dbl_4.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_4_b']))
        self.Mixed_6c.branch7x7dbl_5.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_5_w']))
        self.Mixed_6c.branch7x7dbl_5.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way3_5_b']))
        self.Mixed_6c.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way4_w']))
        self.Mixed_6c.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_2']['way4_b']))  

        self.Mixed_6d.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way1_w']))
        self.Mixed_6d.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way1_b']))
        self.Mixed_6d.branch7x7_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_1_w']))
        self.Mixed_6d.branch7x7_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_1_b']))
        self.Mixed_6d.branch7x7_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_2_w']))
        self.Mixed_6d.branch7x7_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_2_b']))
        self.Mixed_6d.branch7x7_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_3_w']))
        self.Mixed_6d.branch7x7_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way2_3_b']))
        self.Mixed_6d.branch7x7dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_1_w']))
        self.Mixed_6d.branch7x7dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_1_b']))
        self.Mixed_6d.branch7x7dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_2_w']))
        self.Mixed_6d.branch7x7dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_2_b']))
        self.Mixed_6d.branch7x7dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_3_w']))
        self.Mixed_6d.branch7x7dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_3_b']))
        self.Mixed_6d.branch7x7dbl_4.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_4_w']))
        self.Mixed_6d.branch7x7dbl_4.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_4_b']))
        self.Mixed_6d.branch7x7dbl_5.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_5_w']))
        self.Mixed_6d.branch7x7dbl_5.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way3_5_b']))
        self.Mixed_6d.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way4_w']))
        self.Mixed_6d.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_3']['way4_b']))  

        self.Mixed_6e.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way1_w']))
        self.Mixed_6e.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way1_b']))
        self.Mixed_6e.branch7x7_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_1_w']))
        self.Mixed_6e.branch7x7_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_1_b']))
        self.Mixed_6e.branch7x7_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_2_w']))
        self.Mixed_6e.branch7x7_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_2_b']))
        self.Mixed_6e.branch7x7_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_3_w']))
        self.Mixed_6e.branch7x7_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way2_3_b']))
        self.Mixed_6e.branch7x7dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_1_w']))
        self.Mixed_6e.branch7x7dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_1_b']))
        self.Mixed_6e.branch7x7dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_2_w']))
        self.Mixed_6e.branch7x7dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_2_b']))
        self.Mixed_6e.branch7x7dbl_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_3_w']))
        self.Mixed_6e.branch7x7dbl_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_3_b']))
        self.Mixed_6e.branch7x7dbl_4.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_4_w']))
        self.Mixed_6e.branch7x7dbl_4.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_4_b']))
        self.Mixed_6e.branch7x7dbl_5.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_5_w']))
        self.Mixed_6e.branch7x7dbl_5.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way3_5_b']))
        self.Mixed_6e.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way4_w']))
        self.Mixed_6e.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer4_4']['way4_b']))

        self.Mixed_7a.branch3x3_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way1_1_w']))
        self.Mixed_7a.branch3x3_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way1_1_b']))
        self.Mixed_7a.branch3x3_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way1_2_w']))
        self.Mixed_7a.branch3x3_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way1_2_b']))
        self.Mixed_7a.branch7x7x3_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_1_w']))
        self.Mixed_7a.branch7x7x3_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_1_b']))
        self.Mixed_7a.branch7x7x3_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_2_w']))
        self.Mixed_7a.branch7x7x3_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_2_b']))
        self.Mixed_7a.branch7x7x3_3.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_3_w']))
        self.Mixed_7a.branch7x7x3_3.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_3_b']))
        self.Mixed_7a.branch7x7x3_4.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_4_w']))
        self.Mixed_7a.branch7x7x3_4.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer5']['way2_4_b']))

        self.Mixed_7b.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way1_w']))
        self.Mixed_7b.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way1_b']))
        self.Mixed_7b.branch3x3_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way23_1_w']))
        self.Mixed_7b.branch3x3_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way23_1_b']))
        self.Mixed_7b.branch3x3_2a.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way2_2_w']))
        self.Mixed_7b.branch3x3_2a.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way2_2_b']))
        self.Mixed_7b.branch3x3_2b.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way3_2_w']))
        self.Mixed_7b.branch3x3_2b.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way3_2_b']))
        self.Mixed_7b.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way45_1_w']))
        self.Mixed_7b.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way45_1_b']))
        self.Mixed_7b.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way45_2_w']))
        self.Mixed_7b.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way45_2_b']))
        self.Mixed_7b.branch3x3dbl_3a.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way4_3_w']))
        self.Mixed_7b.branch3x3dbl_3a.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way4_3_b']))
        self.Mixed_7b.branch3x3dbl_3b.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way5_3_w']))
        self.Mixed_7b.branch3x3dbl_3b.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way5_3_b']))
        self.Mixed_7b.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way6_w']))
        self.Mixed_7b.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_1']['way6_b']))

        self.Mixed_7c.branch1x1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way1_w']))
        self.Mixed_7c.branch1x1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way1_b']))
        self.Mixed_7c.branch3x3_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way23_1_w']))
        self.Mixed_7c.branch3x3_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way23_1_b']))
        self.Mixed_7c.branch3x3_2a.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way2_2_w']))
        self.Mixed_7c.branch3x3_2a.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way2_2_b']))
        self.Mixed_7c.branch3x3_2b.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way3_2_w']))
        self.Mixed_7c.branch3x3_2b.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way3_2_b']))
        self.Mixed_7c.branch3x3dbl_1.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way45_1_w']))
        self.Mixed_7c.branch3x3dbl_1.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way45_1_b']))
        self.Mixed_7c.branch3x3dbl_2.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way45_2_w']))
        self.Mixed_7c.branch3x3dbl_2.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way45_2_b']))
        self.Mixed_7c.branch3x3dbl_3a.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way4_3_w']))
        self.Mixed_7c.branch3x3dbl_3a.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way4_3_b']))
        self.Mixed_7c.branch3x3dbl_3b.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way5_3_w']))
        self.Mixed_7c.branch3x3dbl_3b.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way5_3_b']))
        self.Mixed_7c.branch_pool.group1.weight= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way6_w']))
        self.Mixed_7c.branch_pool.group1.bias= nn.Parameter(torch.FloatTensor(dict['layer6_2']['way6_b']))
        self.group1.weight= nn.Parameter(torch.FloatTensor(dict['outputlayer']['fc_w']))
        self.group1.bias= nn.Parameter(torch.FloatTensor(dict['outputlayer']['fc_b']))

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0, :, :] = x[:, 0, :, :] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1, :, :] = x[:, 1, :, :] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2, :, :] = x[:, 2, :, :] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # N*3*299*299
        x = self.Conv2d_1a_3x3(x)
        # N*32*149*149
        x = self.Conv2d_2a_3x3(x)
        # N*32*147*147
        x = self.Conv2d_2b_3x3(x)
        # N*64*147*147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N*64*73*73
        x = self.Conv2d_3b_1x1(x)
        # N*80*73*73
        x = self.Conv2d_4a_3x3(x)
        # N*192*71*71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N*192*35*35
        x = self.Mixed_5b(x)
        # N*256*35*35
        x = self.Mixed_5c(x)
        # N*288*35*35
        x = self.Mixed_5d(x)
        # N*288*35*35
        x = self.Mixed_6a(x)
        # N*768*17*17
        x = self.Mixed_6b(x)
        # N*768*17*17
        x = self.Mixed_6c(x)
        # N*768*17*17
        x = self.Mixed_6d(x)
        # N*768*17*17
        x = self.Mixed_6e(x)
        # N*768*17*17
        # N*768*17*17
        x = self.Mixed_7a(x)
        # N*1280*8*8
        x = self.Mixed_7b(x)
        # N*2048*8*8
        x = self.Mixed_7c(x)
        # N*2048*8*8
        x = F.avg_pool2d(x, kernel_size=8)
        # N*2048*1*1
        x = x.view(x.size(0), -1)
        # N*2048
        x = self.group1(x)
        # N*num_classes
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.group1 = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)

    def forward(self, x):
        x = self.group1(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    net = Inception3()
    print('[1/5] Loading network data ...')
    model = onnx.load('../data/inceptionV3.onnx')
    dict = {
        "layer1": {
            "way1_w": 0.4580000042915344,
            "way1_b": -0.029999999329447746,
            "way2_w": 0.4480000138282776,
            "way2_b": -0.08799999952316284,
            "way3_w": 0.44999998807907104,
            "way3_b": -0.18799999356269836,
            "c_1_w": get_seperate_weights(model, "926"),
            "c_1_b": get_seperate_weights(model, "927"),
            "c_2_w": get_seperate_weights(model, "929"),
            "c_2_b": get_seperate_weights(model, "930"),
            "c_3_w": get_seperate_weights(model, "932"),
            "c_3_b": get_seperate_weights(model, "933"),
            "c_4_w": get_seperate_weights(model, "935"),
            "c_4_b": get_seperate_weights(model, "936"),
            "c_5_w": get_seperate_weights(model, "938"),
            "c_5_b": get_seperate_weights(model, "939")
        },
        "layer2_1": {
            "way1_w": get_seperate_weights(model, "941"),
            "way1_b": get_seperate_weights(model, "942"),
            "way2_1_w": get_seperate_weights(model, "944"),
            "way2_1_b": get_seperate_weights(model, "945"),
            "way2_2_w": get_seperate_weights(model, "947"),
            "way2_2_b": get_seperate_weights(model, "948"),
            "way3_1_w": get_seperate_weights(model, "950"),
            "way3_1_b": get_seperate_weights(model, "951"),
            "way3_2_w": get_seperate_weights(model, "953"),
            "way3_2_b": get_seperate_weights(model, "954"),
            "way3_3_w": get_seperate_weights(model, "956"),
            "way3_3_b": get_seperate_weights(model, "957"),
            "way4_w": get_seperate_weights(model, "959"),
            "way4_b": get_seperate_weights(model, "960")
        }, 
        "layer2_2": {
            "way1_w": get_seperate_weights(model, "962"),
            "way1_b": get_seperate_weights(model, "963"),
            "way2_1_w": get_seperate_weights(model, "965"),
            "way2_1_b": get_seperate_weights(model, "966"),
            "way2_2_w": get_seperate_weights(model, "968"),
            "way2_2_b": get_seperate_weights(model, "969"),
            "way3_1_w": get_seperate_weights(model, "971"),
            "way3_1_b": get_seperate_weights(model, "972"),
            "way3_2_w": get_seperate_weights(model, "974"),
            "way3_2_b": get_seperate_weights(model, "975"),
            "way3_3_w": get_seperate_weights(model, "977"),
            "way3_3_b": get_seperate_weights(model, "978"),
            "way4_w": get_seperate_weights(model, "980"),
            "way4_b": get_seperate_weights(model, "981")
        },
        "layer2_3": {
            "way1_w": get_seperate_weights(model, "983"),
            "way1_b": get_seperate_weights(model, "984"),
            "way2_1_w": get_seperate_weights(model, "986"),
            "way2_1_b": get_seperate_weights(model, "987"),
            "way2_2_w": get_seperate_weights(model, "989"),
            "way2_2_b": get_seperate_weights(model, "990"),
            "way3_1_w": get_seperate_weights(model, "992"),
            "way3_1_b": get_seperate_weights(model, "993"),
            "way3_2_w": get_seperate_weights(model, "995"),
            "way3_2_b": get_seperate_weights(model, "996"),
            "way3_3_w": get_seperate_weights(model, "998"),
            "way3_3_b": get_seperate_weights(model, "999"),
            "way4_w": get_seperate_weights(model, "1001"),
            "way4_b": get_seperate_weights(model, "1002")
        },        
        "layer3": {
            "way1_w": get_seperate_weights(model, "1004"),
            "way1_b": get_seperate_weights(model, "1005"),
            "way2_1_w": get_seperate_weights(model, "1007"),
            "way2_1_b": get_seperate_weights(model, "1008"),
            "way2_2_w": get_seperate_weights(model, "1010"),
            "way2_2_b": get_seperate_weights(model, "1011"),
            "way2_3_w": get_seperate_weights(model, "1013"),
            "way2_3_b": get_seperate_weights(model, "1014")
        },
        "layer4_1": {
            "way1_w": get_seperate_weights(model, "1016"),
            "way1_b": get_seperate_weights(model, "1017"),
            "way2_1_w": get_seperate_weights(model, "1019"),
            "way2_1_b": get_seperate_weights(model, "1020"),
            "way2_2_w": get_seperate_weights(model, "1022"),
            "way2_2_b": get_seperate_weights(model, "1023"),
            "way2_3_w": get_seperate_weights(model, "1025"),
            "way2_3_b": get_seperate_weights(model, "1026"),
            "way3_1_w": get_seperate_weights(model, "1028"),
            "way3_1_b": get_seperate_weights(model, "1029"),
            "way3_2_w": get_seperate_weights(model, "1031"),
            "way3_2_b": get_seperate_weights(model, "1032"),
            "way3_3_w": get_seperate_weights(model, "1034"),
            "way3_3_b": get_seperate_weights(model, "1035"),
            "way3_4_w": get_seperate_weights(model, "1037"),
            "way3_4_b": get_seperate_weights(model, "1038"),
            "way3_5_w": get_seperate_weights(model, "1040"),
            "way3_5_b": get_seperate_weights(model, "1041"),
            "way4_w": get_seperate_weights(model, "1043"),
            "way4_b": get_seperate_weights(model, "1044")
        },
        "layer4_2": {
            "way1_w": get_seperate_weights(model, "1046"),
            "way1_b": get_seperate_weights(model, "1047"),
            "way2_1_w": get_seperate_weights(model, "1049"),
            "way2_1_b": get_seperate_weights(model, "1050"),
            "way2_2_w": get_seperate_weights(model, "1052"),
            "way2_2_b": get_seperate_weights(model, "1053"),
            "way2_3_w": get_seperate_weights(model, "1055"),
            "way2_3_b": get_seperate_weights(model, "1056"),
            "way3_1_w": get_seperate_weights(model, "1058"),
            "way3_1_b": get_seperate_weights(model, "1059"),
            "way3_2_w": get_seperate_weights(model, "1061"),
            "way3_2_b": get_seperate_weights(model, "1062"),
            "way3_3_w": get_seperate_weights(model, "1064"),
            "way3_3_b": get_seperate_weights(model, "1065"),
            "way3_4_w": get_seperate_weights(model, "1067"),
            "way3_4_b": get_seperate_weights(model, "1068"),
            "way3_5_w": get_seperate_weights(model, "1070"),
            "way3_5_b": get_seperate_weights(model, "1071"),
            "way4_w": get_seperate_weights(model, "1073"),
            "way4_b": get_seperate_weights(model, "1074")
        },
        "layer4_3": {
            "way1_w": get_seperate_weights(model, "1076"),
            "way1_b": get_seperate_weights(model, "1077"),
            "way2_1_w": get_seperate_weights(model, "1079"),
            "way2_1_b": get_seperate_weights(model, "1080"),
            "way2_2_w": get_seperate_weights(model, "1082"),
            "way2_2_b": get_seperate_weights(model, "1083"),
            "way2_3_w": get_seperate_weights(model, "1085"),
            "way2_3_b": get_seperate_weights(model, "1086"),
            "way3_1_w": get_seperate_weights(model, "1088"),
            "way3_1_b": get_seperate_weights(model, "1089"),
            "way3_2_w": get_seperate_weights(model, "1091"),
            "way3_2_b": get_seperate_weights(model, "1092"),
            "way3_3_w": get_seperate_weights(model, "1094"),
            "way3_3_b": get_seperate_weights(model, "1095"),
            "way3_4_w": get_seperate_weights(model, "1097"),
            "way3_4_b": get_seperate_weights(model, "1098"),
            "way3_5_w": get_seperate_weights(model, "1100"),
            "way3_5_b": get_seperate_weights(model, "1101"),
            "way4_w": get_seperate_weights(model, "1103"),
            "way4_b": get_seperate_weights(model, "1104")
        },
        "layer4_4": {
            "way1_w": get_seperate_weights(model, "1106"),
            "way1_b": get_seperate_weights(model, "1107"),
            "way2_1_w": get_seperate_weights(model, "1109"),
            "way2_1_b": get_seperate_weights(model, "1110"),
            "way2_2_w": get_seperate_weights(model, "1112"),
            "way2_2_b": get_seperate_weights(model, "1113"),
            "way2_3_w": get_seperate_weights(model, "1115"),
            "way2_3_b": get_seperate_weights(model, "1116"),
            "way3_1_w": get_seperate_weights(model, "1118"),
            "way3_1_b": get_seperate_weights(model, "1119"),
            "way3_2_w": get_seperate_weights(model, "1121"),
            "way3_2_b": get_seperate_weights(model, "1122"),
            "way3_3_w": get_seperate_weights(model, "1124"),
            "way3_3_b": get_seperate_weights(model, "1125"),
            "way3_4_w": get_seperate_weights(model, "1127"),
            "way3_4_b": get_seperate_weights(model, "1128"),
            "way3_5_w": get_seperate_weights(model, "1130"),
            "way3_5_b": get_seperate_weights(model, "1131"),
            "way4_w": get_seperate_weights(model, "1133"),
            "way4_b": get_seperate_weights(model, "1134")
        },
        "layer5": {
            "way1_1_w": get_seperate_weights(model, "1136"),
            "way1_1_b": get_seperate_weights(model, "1137"),
            "way1_2_w": get_seperate_weights(model, "1139"),
            "way1_2_b": get_seperate_weights(model, "1140"),
            "way2_1_w": get_seperate_weights(model, "1142"),
            "way2_1_b": get_seperate_weights(model, "1143"),
            "way2_2_w": get_seperate_weights(model, "1145"),
            "way2_2_b": get_seperate_weights(model, "1146"),
            "way2_3_w": get_seperate_weights(model, "1148"),
            "way2_3_b": get_seperate_weights(model, "1149"),
            "way2_4_w": get_seperate_weights(model, "1151"),
            "way2_4_b": get_seperate_weights(model, "1152")
        },
        "layer6_1": {
            "way1_w": get_seperate_weights(model, "1154"),
            "way1_b": get_seperate_weights(model, "1155"),
            "way23_1_w": get_seperate_weights(model, "1157"),
            "way23_1_b": get_seperate_weights(model, "1158"),
            "way2_2_w": get_seperate_weights(model, "1160"),
            "way2_2_b": get_seperate_weights(model, "1161"),
            "way3_2_w": get_seperate_weights(model, "1163"),
            "way3_2_b": get_seperate_weights(model, "1164"),
            "way45_1_w": get_seperate_weights(model, "1166"),
            "way45_1_b": get_seperate_weights(model, "1167"),
            "way45_2_w": get_seperate_weights(model, "1169"),
            "way45_2_b": get_seperate_weights(model, "1170"),
            "way4_3_w": get_seperate_weights(model, "1172"),
            "way4_3_b": get_seperate_weights(model, "1173"),
            "way5_3_w": get_seperate_weights(model, "1175"),
            "way5_3_b": get_seperate_weights(model, "1176"),
            "way6_w": get_seperate_weights(model, "1178"),
            "way6_b": get_seperate_weights(model, "1179")
        },
        "layer6_2": {
            "way1_w": get_seperate_weights(model, "1181"),
            "way1_b": get_seperate_weights(model, "1182"),
            "way23_1_w": get_seperate_weights(model, "1184"),
            "way23_1_b": get_seperate_weights(model, "1185"),
            "way2_2_w": get_seperate_weights(model, "1187"),
            "way2_2_b": get_seperate_weights(model, "1188"),
            "way3_2_w": get_seperate_weights(model, "1190"),
            "way3_2_b": get_seperate_weights(model, "1191"),
            "way45_1_w": get_seperate_weights(model, "1193"),
            "way45_1_b": get_seperate_weights(model, "1194"),
            "way45_2_w": get_seperate_weights(model, "1196"),
            "way45_2_b": get_seperate_weights(model, "1197"),
            "way4_3_w": get_seperate_weights(model, "1199"),
            "way4_3_b": get_seperate_weights(model, "1200"),
            "way5_3_w": get_seperate_weights(model, "1202"),
            "way5_3_b": get_seperate_weights(model, "1203"),
            "way6_w": get_seperate_weights(model, "1205"),
            "way6_b": get_seperate_weights(model, "1206")
        },
        "outputlayer": {
            "fc_w": get_seperate_weights(model, "fc.weight"),
            "fc_b": get_seperate_weights(model, "fc.bias")
        }
    }
    print('[1/5] Network data loaded.')
    print('[2/5] Setting params ...')
    net.set_params(dict)
    print('[2/5] Param setted.')
    print('[3/5] Loading input and output data')
    with open('../data/inceptionInput.json', 'r') as fp:
        input = json.load(fp)
    with open('../data/inceptionOutput.json', 'r') as fp:
        output = json.load(fp);
    input = torch.FloatTensor(np.array(input['test0']).reshape(1, 3, 299, 299).astype(np.float32))
    # output = torch.FloatTensor(np.array(output['test0']).reshape(1, 1000).astype(np.float32))
    print('[3/5] Input and output data loaded.')
    print('[4/5] Inference ...')
    net_out = net(input).flatten()
    print(net_out[0], net_out[1], net_out[2], net_out[100])
    print('[4/5] Inference End.')

