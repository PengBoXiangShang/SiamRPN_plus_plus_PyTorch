import os

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.adj_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))

        #
        self.fusion_module_1 = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_2 = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.Box_Head = nn.Sequential(
            nn.Conv2d(256, 4 * 5, kernel_size=1, padding=0, stride=1))
        self.Cls_Head = nn.Sequential(
            nn.Conv2d(256, 2 * 5, kernel_size=1, padding=0, stride=1))

    def forward(self, examplar_feature_map, search_region_feature_map, BATCH_SIZE):
        # crop
        #print 'examplar_feature_map:'
        #print examplar_feature_map.size()
        cropped_examplar_feature_map = F.pad(
            examplar_feature_map, (-4, -4, -4, -4))
        #print 'cropped_examplar_feature_map'
        #print cropped_examplar_feature_map.size()
        adj_1_output = self.adj_1(cropped_examplar_feature_map)
        #print 'adj_1_output'
        #print adj_1_output.size()
        # ipdb.set_trace()
        # TODO
        adj_1_output = adj_1_output.reshape(-1, 7, 7)
        #print 'adj_1_output.reshape(-1,8,8)'
        #print adj_1_output.size()
        #ipdb.set_trace()
        adj_1_output = adj_1_output.unsqueeze(0).permute(1, 0, 2, 3)
        adj_2_output = self.adj_2(search_region_feature_map)
        adj_2_output = adj_2_output.reshape(-1, 31, 31)
        adj_2_output = adj_2_output.unsqueeze(0)

        adj_3_output = self.adj_3(cropped_examplar_feature_map)
        # TODO
        adj_3_output = adj_3_output.reshape(-1, 7, 7)
        adj_3_output = adj_3_output.unsqueeze(0).permute(1, 0, 2, 3)

        adj_4_output = self.adj_4(search_region_feature_map)
        adj_4_output = adj_4_output.reshape(-1, 31, 31)
        adj_4_output = adj_4_output.unsqueeze(0)

        depthwise_cross_reg = F.conv2d(
            adj_2_output, adj_1_output, bias=None, stride=1, padding=0, groups=adj_1_output.size()[0]).squeeze()

        depthwise_cross_cls = F.conv2d(
            adj_4_output, adj_3_output, bias=None, stride=1, padding=0, groups=adj_3_output.size()[0]).squeeze()
        # ipdb.set_trace()
        depthwise_cross_reg = depthwise_cross_reg.reshape(-1, 256, 25, 25)
        depthwise_cross_cls = depthwise_cross_cls.reshape(-1, 256, 25, 25)
        depthwise_cross_reg = self.fusion_module_1(depthwise_cross_reg)
        depthwise_cross_cls = self.fusion_module_2(depthwise_cross_cls)

        bbox_regression_prediction = self.Box_Head(depthwise_cross_reg)
        cls_prediction = self.Cls_Head(depthwise_cross_cls)
        return cls_prediction, bbox_regression_prediction
