# -*- coding: utf-8 -*-
"""
2D Unet-like architecture code in Pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.dsbn import DomainSpecificBatchNorm2d



def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def normalization(planes, norm='gn', num_domains=None):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        return y


class ConvU_Rec(nn.Module):
    def __init__(self, planes, norm='bn', activation='relu', num_domains=None):
        super(ConvU_Rec, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes//2, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes//2, norm, num_domains)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes//2, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm, num_domains)

        self.conv3 = nn.Conv2d(planes//2, planes//2, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes//2, norm, num_domains)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, domain_label=None):
        #layer 1 conv, bn, relu
        x = self.conv1(x)
        if domain_label is not None:
            x, _ = self.bn1(x, domain_label)
        else:
            x = self.bn1(x)
        x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        y = self.conv3(y)
        if domain_label is not None:
            y, _ = self.bn3(y, domain_label)
        else:
            y = self.bn3(y)
        y = self.activation(y)

        return y


class Unet2D(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet2D, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        y1_pred = self.seg1(y1)
        return y1_pred


class Unet2D_MT(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet2D_MT, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)
        self.rec1 = nn.Conv2d(2*n, c, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_rec=False):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        if is_rec:
            y1_pred = self.rec1(y1)
        else:
            y1_pred = self.seg1(y1)
        return y1_pred


class Encoder(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', activation='relu'):
        super(Encoder, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        return [x1, x2, x3, x4, x5]

class Decoder(nn.Module):
    def __init__(self, n=16, num_classes=2, norm='bn', activation='relu'):
        super(Decoder, self).__init__()
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.out1 = nn.Conv2d(2*n, num_classes, 3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feats):
        y4 = self.convu4(feats[-1], feats[-2])
        y3 = self.convu3(y4, feats[-3])
        y2 = self.convu2(y3, feats[-4])
        y1 = self.convu1(y2, feats[-5])
        y1_pred = self.out1(y1)
        return y1_pred

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n=16, norm='bn', activation='relu'):
        super(UNet, self).__init__()
        self.convd1 = ConvD(n_channels,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.out1 = nn.Conv2d(2*n, n_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        feats = [x1, x2, x3, x4, x5]
        y4 = self.convu4(feats[-1], feats[-2])
        y3 = self.convu3(y4, feats[-3])
        y2 = self.convu2(y3, feats[-4])
        y1 = self.convu1(y2, feats[-5])
        y1_pred = self.out1(y1)
        return y1_pred

        


class Rec_Decoder(nn.Module):
    def __init__(self, n=16, num_classes=2, norm='bn', activation='relu', num_domains=None):
        super(Rec_Decoder, self).__init__()
        self.convu4 = ConvU_Rec(16*n, norm, activation=activation, num_domains=num_domains)
        self.convu3 = ConvU_Rec(8*n, norm, activation=activation, num_domains=num_domains)
        self.convu2 = ConvU_Rec(4*n, norm, activation=activation, num_domains=num_domains)
        self.convu1 = ConvU_Rec(2*n, norm, activation=activation, num_domains=num_domains)

        self.out1 = nn.Conv2d(n, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, domain_label=None):
        y4 = self.convu4(x, domain_label=domain_label)
        y3 = self.convu3(y4, domain_label=domain_label)
        y2 = self.convu2(y3, domain_label=domain_label)
        y1 = self.convu1(y2, domain_label=domain_label)
        y1_pred = self.out1(y1)
        return y1_pred


class Unet2D_DS(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet2D_DS, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg5 = nn.Conv2d(16*n, num_classes, 3, padding=1)
        self.seg4 = nn.Conv2d(16*n, num_classes, 3, padding=1)
        self.seg3 = nn.Conv2d(8*n, num_classes, 3, padding=1)
        self.seg2 = nn.Conv2d(4*n, num_classes, 3, padding=1)
        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)

        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, deep_sup=False):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        if deep_sup:
            y5_pred = self.upscore5(self.seg5(x5))
            y4_pred = self.upscore4(self.seg4(y4))
            y3_pred = self.upscore3(self.seg3(y3))
            y2_pred = self.upscore2(self.seg2(y2))
            y1_pred = self.seg1(y1)
            return y1_pred, y2_pred, y3_pred, y4_pred, y5_pred
        else:
            y1_pred = self.seg1(y1)
            return y1_pred


class Unet2D_MS(nn.Module):
    def __init__(self, c=3, n=16, norm='bn', num_classes=2, activation='relu'):
        super(Unet2D_MS, self).__init__()
        self.convd1 = ConvD(c,     n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n,   2*n, norm, activation=activation)
        self.convd3 = ConvD(2*n, 4*n, norm, activation=activation)
        self.convd4 = ConvD(4*n, 8*n, norm, activation=activation)
        self.convd5 = ConvD(8*n,16*n, norm, activation=activation)
        
        self.convu4 = ConvU(16*n, norm, first=True, activation=activation)
        self.convu3 = ConvU(8*n, norm, activation=activation)
        self.convu2 = ConvU(4*n, norm, activation=activation)
        self.convu1 = ConvU(2*n, norm, activation=activation)

        self.seg5 = nn.Conv2d(16*n, num_classes, 3, padding=1)
        self.seg4 = nn.Conv2d(16*n, num_classes, 3, padding=1)
        self.seg3 = nn.Conv2d(8*n, num_classes, 3, padding=1)
        self.seg2 = nn.Conv2d(4*n, num_classes, 3, padding=1)
        self.seg1 = nn.Conv2d(2*n, num_classes, 3, padding=1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, multi_scale_output=False):
            x1 = self.convd1(x)
            x2 = self.convd2(x1)
            x3 = self.convd3(x2)
            x4 = self.convd4(x3)
            x5 = self.convd5(x4)

            y4 = self.convu4(x5, x4)
            y3 = self.convu3(y4, x3)
            y2 = self.convu2(y3, x2)
            y1 = self.convu1(y2, x1)

            if multi_scale_output:
                y5_pred = self.seg5(x5)
                y4_pred = self.seg4(y4)
                y3_pred = self.seg3(y3)
                y2_pred = self.seg2(y2)
                y1_pred = self.seg1(y1)
                return y1_pred, y2_pred, y3_pred, y4_pred, y5_pred
            else:
                y1_pred = self.seg1(y1)
                return y1_pred


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, n=16):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, n, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(n, 2*n, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(2*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(2*n, 4*n, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(4*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(4*n, 8*n, 4, padding=1),
                    nn.InstanceNorm2d(8*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(8*n, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)