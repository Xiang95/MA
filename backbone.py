import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from kitti_object import KittiObject
import knn
from mlp import MLP

kitti_data = KittiObject()


class BackBone(nn.Module):

    def __init__(self):
        super(BackBone,self).__init__()
        # image Block
        self.image_block = ImageBackbone
        # Fusion Block
        self.fusion_block = FusionBlock
        # Bev Block
        self.bev_block = BevBackbone
        # HEADER
        self.header = Header(num_classes)

    def forward(self, idx):
        # get input data
        img_fpn = self.image_block(kitti_data.get_image(idx))
        # a upsampling-layer x4 must be added, in oder to match the original size of Image
        img_fpn_upsampled = F.interpolate(img_fpn, scale_factor=4, mode='bilinear', align_corners=True)
        fused = self.fusion_block(img_fpn_upsampled)
        dense_bev = self.bev_block(fused)

        header_out = self.header(dense_bev)

        return header_out






#### Backbone-network for image #####

ImageBackbone = resnet_fpn_backbone(backbone_name='resnet18', pretrained=True).eval()


# Backbone-network for BEV


def conv3x3(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chn, dim_size, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_chn, dim_size, stride)
        self.bn1 = nn.BatchNorm2d(dim_size)
        self.conv2 = conv3x3(dim_size, dim_size * 1)
        self.bn2 = nn.BatchNorm2d(dim_size)
        self.activation = nn.ReLU(inplace=True)

        self.downsample = None
        if stride == 2:
            layers = []
            layers += [conv1x1(in_chn, dim_size, stride)]
            layers += [nn.BatchNorm2d(dim_size)]
            self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)

        return out


class BevBackbone(nn.Module):
    '''
        Based on Resnet
        one group of convolutional layers: 2 convolutions, stride ==1 Dimension ==32
        four groups of convolutional layers: 4, 8, 12, 12 convolutions, starting with stride==2 then stride ==1
                                             Dimension: 64, 128, 192, 256
    '''
    def __init__(self):
        super(BevBackbone, self).__init__()

        # first group
        self.in_chn = 32
        self.conv1 = nn.Conv2d(self.in_chn, self.in_chn, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_chn)
        self.conv2 = nn.Conv2d(self.in_chn, self.in_chn, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.in_chn)
        self.activation = nn.ReLU(inplace=True)

        # We use BasicBlocks here, but Deep Continuous Fusion uses ResBlock
        self.layer1 = self._make_layer(64, 4, stride=2)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(192, 12, stride=2)
        self.layer4 = self._make_layer(256, 12, stride=2)

        self.toplayer = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.lateral3 = nn.Conv2d(192, 256, kernel_size=1, stride=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)

    def _make_layer(self, dim_size, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_chn, dim_size, stride)]
            self.in_chn = dim_size * 1
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def _upsample_add_dimension_match(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)[:, :, :, :-1] + y

    def forward(self, x):
        c1 = self.activation(self.bn1(self.conv1(x)))
        c1 = self.activation(self.bn2(self.conv2(c1)))

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        ''' as specified in the paper, the final BEV feature output combines the last three residual groups' output, 
        not four '''

        p5 = self.toplayer(c5)
        p4 = self._upsample_add_dimension_match(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))

        return p3


class FusionBlock(nn.Module):
    def __init__(self):

    def forward(self, img_fpn, bev):
        lidar_backprojection_on_image_index, offset_3d = knn.knn_search(idx)
        image_feature = []
        for knn_group in lidar_backprojection_on_image_index[0]:
            knn_group_feature = []
            for pts_2d in knn_group:
                image_feature_single = img_fpn[:, pts_2d[0], pts_2d[1]]
                knn_group_feature.append(image_feature_single)
            image_feature.append(knn_group_feature)
        input_mlp = np.concatenate((image_feature, offset_3d), axis=2) # num_points * 9 * (256+3)
        output_mlp = MLP(input_mlp)
        return output_mlp

class Header(nn.Module):

    def __init__(self, num_classes):
        super(Header, self).__init__()

        self.anchor_orients = [0, np.pi/2]
        self.score_out = (num_classes + 1) * len(self.anchor_orients)
        # (dx, dy, dz, w, l, h, t) * 2 anchors
        self.bbox_out = 8 * len(self.anchor_orients)

        self.conv1 = nn.Conv2d(256, self.score_out + self.bbox_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        clsscore_bbox = self.conv1(x)
        cls_score, bbox = torch.split(clsscore_bbox, [self.score_out, self.bbox_out], dim=1)

        return cls_score, bbox


bev_stream = BEVBackbone()
head = Header(10)