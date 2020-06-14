
import datagen
from kitti_util import *
from kitti_object import *
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt

kitti_data = KittiObject()


def knn_search():
    # TODO the shape of pointcloud in many files are weird. cant reshape to n*4
    # load and project velo to top view
    points_4d = kitti_data.get_lidar(1000)
    calibration = kitti_data.get_calibration()
    points_3d = points_4d[:, :-1]

    # keep velo in image FOV
    # TODO not enough points on the image plane
    imagefov_velo, lidar_index_in_fov = get_lidar_index_in_image_fov(points_3d, calibration, 0, 0, 1224, 370,
                                                                     return_more=True)
    imagefov_velo_4d = np.hstack((imagefov_velo, np.zeros((imagefov_velo.shape[0], 1))))
    ''' Lidar to BEV
        the value store in the array are the index of points which has the max. height in each channel(0-33) 
        the 33th channel was intended for something else'''
    bev, bev_with_quantized_index, lidar_4d = lidar_to_top(imagefov_velo_4d)
    # filter the empty points on bev_plane
    bev_non_zero_index = np.stack(np.nonzero(bev_with_quantized_index[:, :, :-1]), axis=-1)
    index = np.nonzero(bev_with_quantized_index)
    # knn search on non_zero_bev, bev last channel was manuel added, so we need to delete the last channel
    tree = KDTree(bev_non_zero_index)
    # 8 neighbours
    dist, ind = tree.query(bev_non_zero_index, k=9)  # k=3 nearest neighbors where k1 = identity

    '''knn is an array points_number * 9 * 3, 
    which indicate the position of velo_points in the imagefov_velo array'''
    knn = bev_non_zero_index[ind]
    # get the to backproject 3d points
    lidar_backprojection, lidar_backprojection_on_image_index = bev_to_pc(knn, bev_with_quantized_index,
                                                                          lidar_4d, calibration)
    # pts_2d_group_on_image, pts_3d_backprojection = backprojection(bev, knn, calibration)
    offset_3d = calculate_3d_offset(lidar_backprojection)
    return lidar_backprojection_on_image_index, offset_3d


def calculate_3d_offset(lidar_backprojection):
    offset_3d_all = []
    for pts_group in lidar_backprojection:
        offset_3d_group = []
        for pts in range(len(pts_group)):
            squared_offset_3d = np.sum((pts_group[0]-pts_group[pts])**2, axis=0)
            offset_3d = np.sqrt(squared_offset_3d)
            offset_3d_group.append(offset_3d)
        offset_3d_all.append(offset_3d_group)
    return offset_3d_all


def retrieve_image_feature(pts_2d_group_on_image, ):
    image = load_image(image_file)
    image_feature = image[pts_2d_group_on_image]
    return image_feature


def main():
    to_project_points, offset_3d = knn_search()

main()