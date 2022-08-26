import numbers
import os
import yaml
import pickle
import argparse
import numpy as np
from os.path import join, exists
from sklearn.neighbors import KDTree
from utils.data_process import DataProcessing as DP

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', default=None, help='source dataset path [default: None]')
parser.add_argument('--dst_path', default=None, help='destination dataset path [default: None]')
parser.add_argument('--grid_size', type=float, default=0, help='Subsample Grid Size [default: 0.06]')
parser.add_argument('--yaml_config', default='utils/synlidar.yaml', help='semantic-kitti.yaml path')
parser.add_argument('--stacking', default='True', help='if perform stacking on minor classes')
FLAGS = parser.parse_args()


data_config = FLAGS.yaml_config
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = FLAGS.grid_size
dataset_path = FLAGS.src_path
output_path = FLAGS.dst_path + '_' + str(FLAGS.grid_size)
seq_list = np.sort(os.listdir(dataset_path))
for seq_id in seq_list:
    print('sequence' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)
    seq_path_out = join(output_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    # create some path for output file
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

    if int(seq_id) < 13: # all data are labeled
        label_path = join(seq_path, 'labels')
        label_path_out = join(seq_path_out, 'labels')
        os.makedirs(label_path_out) if not exists(label_path_out) else None
        scan_list = np.sort(os.listdir(pc_path))
        if FLAGS.stacking == 'True':
            i = 0
            while i < len(scan_list):
                num_points = 0
                stacked_point = None
                stacked_label = None
                while i < len(scan_list) and num_points < 150000:
                    scan_id = scan_list[i]
                # load points and labels for point one scan id
                    points = DP.load_pc_kitti(join(pc_path, scan_id))
                    labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
                    # mask major class: cars, road, sidewalk, vegetation, building
                    cond = (labels == 1) | (labels == 9) | (labels == 11) | (labels == 15) | (labels == 13)
                    cond = [not c for c in cond]
                    print("before: ", points.shape, labels.shape)
                    points, labels = points[cond], labels[cond]
                    print("after: ", points.shape, labels.shape)
                    if num_points == 0:
                        stacked_label = labels
                        stacked_point = points
                    else:
                        stacked_point = np.append(stacked_point,points, axis = 0)
                        stacked_label = np.append(stacked_label, labels, axis = 0)
                    num_points += len(points)
                    i += 1
                search_tree = KDTree(stacked_point)
                print("Final stack num_points:", stacked_point.shape)
                KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
                np.save(join(pc_path_out, scan_id)[:-4], stacked_point)
                np.save(join(label_path_out, scan_id)[:-4], stacked_label)
                with open(KDTree_save, 'wb') as f:
                    pickle.dump(search_tree, f)
        elif FLAGS.stacking == 'replicate':
            for scan_id in scan_list:
                stacked_point = None
                stacked_label = None
                num_points = 0
                print(scan_id)
                # load points and labels for point one scan id
                points = DP.load_pc_kitti(join(pc_path, scan_id))
                labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
                # sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
                cond = (labels == 1) | (labels == 9) | (labels == 11) | (labels == 15) | (labels == 13)
                cond = [not c for c in cond]
                print("before: ", points.shape, labels.shape)
                points, labels = points[cond], labels[cond]
                print("after: ", points.shape, labels.shape)
                while num_points < 150000:
                    if num_points == 0:
                        stacked_label = labels
                        stacked_point = points
                    else:
                        stacked_point = np.append(stacked_point, points, axis = 0)
                        stacked_label = np.append(stacked_label, labels, axis = 0)
                    num_points += len(points)
                    
                sub_points, sub_labels = stacked_point, stacked_label
                # build KDtree for search
                search_tree = KDTree(sub_points)
                KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
                np.save(join(pc_path_out, scan_id)[:-4], sub_points)
                np.save(join(label_path_out, scan_id)[:-4], sub_labels)
                with open(KDTree_save, 'wb') as f:
                    pickle.dump(search_tree, f)

        else:
            for scan_id in scan_list:
                print(scan_id)
                # load points and labels for point one scan id
                points = DP.load_pc_kitti(join(pc_path, scan_id))
                labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
                sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
                sub_points, sub_labels = points, labels
                # build KDtree for search
                search_tree = KDTree(sub_points)

                KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
                np.save(join(pc_path_out, scan_id)[:-4], sub_points)
                np.save(join(label_path_out, scan_id)[:-4], sub_labels)
                with open(KDTree_save, 'wb') as f:
                    pickle.dump(search_tree, f)

                if seq_id == '08': # use seq 8 as validation
                    proj_path = join(seq_path_out, 'proj')
                    os.makedirs(proj_path) if not exists(proj_path) else None
                    proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)
                    proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                    with open(proj_save, 'wb') as f:
                        pickle.dump([proj_inds], f)
    else:
        proj_path = join(seq_path_out, 'proj')
        os.makedirs(proj_path) if not exists(proj_path) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))
            sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)
