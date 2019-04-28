from customized_resnet import *
from RPN import *


class SiamRPN(nn.Module):
    def __init__(self):
        super(SiamRPN, self).__init__()
        self.examplar_branch = resnet50()
        self.search_region_branch = resnet50()
        self.conv3_3_RPN = RPN()
        self.conv4_6_RPN = RPN()
        self.conv5_3_RPN = RPN()
        self.weighted_sum_layer_alpha = nn.Conv2d(
            30, 10, kernel_size=1, padding=0, groups=10)
        self.weighted_sum_layer_beta = nn.Conv2d(
            60, 20, kernel_size=1, padding=0, groups=20)

    def forward(self, examplar, search_region):
        _, examplar_conv_3_output, examplar_conv_4_output, examplar_conv_5_output = self.examplar_branch(
            examplar)
        _, search_region_conv_3_output, search_region_conv_4_output, search_region_conv_5_output = self.search_region_branch(search_region)
       
        conv3_3_cls_prediction, conv3_3_bbox_regression_prediction = self.conv3_3_RPN(
            examplar_conv_3_output, search_region_conv_3_output, examplar.size()[0])
        conv4_6_cls_prediction, conv4_6_bbox_regression_prediction = self.conv4_6_RPN(
            examplar_conv_4_output, search_region_conv_4_output, examplar.size()[0])
        conv5_3_cls_prediction, conv5_3_bbox_regression_prediction = self.conv5_3_RPN(
            examplar_conv_5_output, search_region_conv_5_output, examplar.size()[0])
        # ipdb.set_trace()
        stacked_cls_prediction = torch.cat((conv3_3_cls_prediction, conv4_6_cls_prediction, conv5_3_cls_prediction), 2).reshape(
            examplar.size()[0], 10, -1, 25, 25).reshape(examplar.size()[0], -1, 25, 25)
        stacked_regression_prediction = torch.cat(
            (conv3_3_bbox_regression_prediction, conv4_6_bbox_regression_prediction, conv5_3_bbox_regression_prediction), 2).reshape(examplar.size()[0], 20, -1, 25, 25).reshape(examplar.size()[0], -1, 25, 25)
        # ipdb.set_trace()
        fused_cls_prediction = self.weighted_sum_layer_alpha(
            stacked_cls_prediction)
        fused_regression_prediction = self.weighted_sum_layer_beta(
            stacked_regression_prediction)

        return fused_cls_prediction, fused_regression_prediction
