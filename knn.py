# import torch
# from torchsummary import summary
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# import pcl
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# resnet18_fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False).eval()
#
# print(resnet18_fpn)
# #backbone = resnet18_fpn.to(device)
#
# #summary(backbone, (1, 3, 800, 800))

from kitti_util import *
from kitti_object import *
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
calib_filename = '1.txt'
velo_file = '1.bin'
image_file = '1.png'

calibration = Calibration(calib_filename)

def knn_search(velo_filename):
    # load and project velo to top view
    points = load_velo_scan(velo_filename)
    # keep velo in image FOV
    imagefov_velo, _, _ = get_lidar_in_image_fov(points, calibration, 0, 0, 1224, 370,return_more=True )

    # Lidar to BEV

    '''the value store in the array are the index of points which has the max. height in each channel(0-13) 
    the 14th channel was intended for something else'''

    bev, bev_with_quantized_index, lidar_3d = lidar_to_top(imagefov_velo)
    # knn search on bev, bev last channel was maneulle added, so we need to delete the last channel
    bev_non_zero_index = np.stack(np.nonzero(bev_with_quantized_index[:, :, 0:13]), axis=-1)
    # only for non zero points
    tree = KDTree(bev_non_zero_index)
    # 8 neighbours
    dist, ind = tree.query(bev_non_zero_index, k=9)  # k=3 nearest neighbors where k1 = identity

    '''knn is an array points_number * 9 * 3, 
    which indicate the position of velo_points in the imagefov_velo array'''
    # toDo
    # how to get the element of a n-Dimensional Array using arrays
    knn = bev_non_zero_index[ind]
    index = knn[0]
    bev_knn = bev.item(tuple(index[0]))
    # get the to back project 3d points
    to_project_points = imagefov_velo[bev_knn]
    # caculate the 3d offset euclidean distance
    '''to_project_points: N * 9 *3 array
        distance = sqr((x-x1)2+(y-y1)2+(z-z1)2)
    '''
    offset_3d = 0
    return to_project_points, offset_3d


def knn_back_projection(to_project_points):
    # get the index of lidar on image fov
    lidar2image_index = get_lidar_in_image_fov(to_project_points, calibration, 0, 0, 1224, 370)
    return lidar2image_index


def receive_image_feature(lidar2image_index, image_file):
    image = load_image(image_file)
    image_feature = image[lidar2image_index]
    return image_feature







def main():
    to_project_points, offset_3d = knn_search(velo_file)
    lidar2image_index = knn_back_projection(to_project_points)
    imagefeatrue = receive_image_feature(lidar2image_index, image_file)