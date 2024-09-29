import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from net.model import EfficientNet
from utils.sam import SAM
from utils.supcon import SupConLoss


class Detector(nn.Module):

    def __init__(self, phase='train'):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4",
                                                advprop=True,
                                                num_classes=2)
        self.cel = nn.CrossEntropyLoss()
        self.mapl = nn.BCEWithLogitsLoss()
        self.consl = SupConLoss()

        self.map_head = RegressionMap(c_in=448)
        self.cls_head = ClsHead()

        self.optimizer = SAM(self.parameters(),
                             torch.optim.SGD,
                             lr=0.001,
                             momentum=0.9)
        self.optimizer_head = SAM(list(self.cls_head.parameters()) + list(self.map_head.parameters()),
            torch.optim.SGD,
            lr=0.01,
            momentum=0.9)
        self.phase = phase

    def forward(self, x):
        feature_map, feature_cls, _ = self.net(x)
        predict_map = self.map_head(feature_map)
        predict = self.cls_head(feature_cls)

        return predict, predict_map, feature_cls

    def training_step(self, x, target, mask):
        for i in range(2):
            pred_cls, pred_map, feature_cls = self(x)
            pred_map = pred_map.squeeze(1)
            if i == 0:
                pred_first = pred_cls
                pred_fist_map = pred_map
                pred_first_feature = feature_cls
            loss_cls = self.cel(pred_cls, target)
            loss_consl = self.consl(feature_cls, target)
            loss_map = self.mapl(pred_map, mask)

            loss_head = loss_cls + 0.1 * loss_map
            self.optimizer.zero_grad()
            self.optimizer_head.zero_grad()
            loss_head.backward(retain_graph=True)
            loss_consl.backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
                self.optimizer_head.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
                self.optimizer_head.second_step(zero_grad=True)

        return pred_first, pred_fist_map, pred_first_feature
    

class SeparableConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class RegressionMap(nn.Module):

    def __init__(self, c_in):
        super(RegressionMap, self).__init__()
        # First separable convolution layer
        self.sep_conv1 = SeparableConv2d(c_in,
                                         c_in,
                                         3,
                                         stride=1,
                                         padding=1,
                                         bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu1 = nn.ReLU(inplace=True)

        # Second separable convolution layer
        self.sep_conv2 = SeparableConv2d(c_in,
                                         1,
                                         3,
                                         stride=1,
                                         padding=1,
                                         bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Applying the first separable convolution, followed by batch normalization and ReLU activation
        x = self.sep_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Applying the second separable convolution, followed by batch normalization and Sigmoid activation
        x = self.sep_conv2(x)
        x = self.bn2(x)
        # x = self.sigmoid(x)

        return x


class ClsHead(nn.Module):

    def __init__(self):
        super(ClsHead, self).__init__()
        self._dropout = nn.Dropout(p=0.4 , inplace=False)
        self._fc = nn.Linear(in_features=1792, out_features=2, bias=True)

    def forward(self, x):
        feature_cls = x.flatten(start_dim=1)
        feature_cls = self._dropout(feature_cls)
        x = self._fc(feature_cls)
        return x
