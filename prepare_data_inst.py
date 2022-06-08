# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOM_TYPES = {
    'conferenceRoom': 0,
    'copyRoom': 1,
    'hallway': 2,
    'office': 3,
    'pantry': 4,
    'WC': 5,
    'auditorium': 6,
    'storage': 7,
    'lounge': 8,
    'lobby': 9,
    'openspace': 10,
}

INV_OBJECT_LABEL = {
    0: 'ceiling',
    1: 'floor',
    2: 'wall',
    3: 'beam',
    4: 'column',
    5: 'window',
    6: 'door',
    7: 'chair',
    8: 'table',
    9: 'bookcase',
    10: 'sofa',
    11: 'board',
    12: 'clutter',
}

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()} # OBJECT_LABEL = {'ceiling':0, 'floor':1, 'wall',2, 'beam':3 ... }


def object_name_to_label(object_class):
    r"""convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL['clutter'])
    return object_label


# modify from https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/s3dis.py  # noqa
def read_s3dis_format(area_id: str,
                      room_name: str,
                      data_root: str = './',
                      label_out: bool = True,
                      verbose: bool = False):
    r"""
    extract data from a room folder
    """
    room_type = room_name.split('_')[0] # room_type = 'hallway', etc
    room_label = ROOM_TYPES[room_type] # room_label = 2
    room_dir = osp.join(data_root, area_id, room_name) # ./Stanford3dDataset_v1.2/Area_1/conferenceRoom1/
    raw_path = osp.join(room_dir, f'{room_name}.txt') # raw_path => './Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.txt'

    room_ver = pd.read_csv(raw_path, sep=' ', header=None).values # room_ver.shape = [n_pts,6] 의 numpy 형태로 저장
  
    try:
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype='float32') # xyz.shape = [n_pts,3]
        rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype='uint8') # rgb.shape = [n_pts,3]
    except:
        return None

    if not label_out: # Default는 True이므로 실행 x
        return xyz, rgb
    n_ver = len(room_ver) # n_ver = n_pts
    del room_ver # memory때문에 xyz, rgb로 이미 따로 저장했으므로 삭제
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz) # 가장 가까운 distance, index = nn.kneighbors
    semantic_labels = np.zeros((n_ver, ), dtype='int64') # n_ver만큼의 길이의 nparray 선언 / semantic_labels = [0, 0, 0, 0, 0, 0, 0, ..]
    room_label = np.asarray([room_label]) # room_label = [2]
    instance_labels = np.ones((n_ver, ), dtype='int64') * -100 # instance_labels = [-100, -100, -100, -100, -100, -100, ...]
    objects = glob.glob(osp.join(room_dir, 'Annotations', '*.txt')) # ./Stanford3dDataset_v1.2/Area_1/conferenceRoom1/Annotation/beam_1.txt , beam_2.txt etc..
    i_object = 1
    for single_object in objects: # area_id/room_name에 있는 모든 object를 iterate
        object_name = os.path.splitext(os.path.basename(single_object))[0] # 확장자를 제거함 object_name = beam_1 (os.path.basename은 앞에 경로를 삭제)
        if verbose:
            print(f'adding object {i_object} : {object_name}')
        object_class = object_name.split('_')[0] # beam
        object_label = object_name_to_label(object_class) # beam이면 object_label = 3
        obj_ver = pd.read_csv(single_object, sep=' ', header=None).values # beam_1.txt 값을 numpy로 받아옴
        _, obj_ind = nn.kneighbors(obj_ver[:, 0:3]) # beam_1.txt에 주어진 x,y,z중 가장 가까운 neighbor들의 index를 obj_ind에 반환
        semantic_labels[obj_ind] = object_label # semantic_labels(n_pts의 길이를 갖는 1차원 numpy) = [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0.. ] => 각 point가 beam에 속하면 3
        # if object_label < 3: # background object
        #     continue
        instance_labels[obj_ind] = i_object # [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 .. ]
        i_object = i_object + 1

        # 즉, semantic_label은 n_pts 길이의 1차원 numpy. 같은 class면 같은 one-hot vector값.
        # instance_label은 모든 다른 객체들을 다른 값으로 one-hot encoding

    return (
        xyz,
        rgb,
        semantic_labels,
        instance_labels,
        room_label,
    )
    # xyz = [n_pts,3], rgb = [n_pts,3], semantic_labels = [n_pts], instance_labels = [n_pts]


def get_parser():
    parser = argparse.ArgumentParser(description='s3dis data prepare')
    parser.add_argument(
        '--data-root', type=str, default='./Stanford3dDataset_v1.2', help='root dir save data')
    parser.add_argument(
        '--save-dir', type=str, default='./preprocess', help='directory save processed data')
    parser.add_argument(
        '--patch', action='store_true', help='patch data or not (just patch at first time running)') # action = 'store_true' : 인자를 적으면 True, else False 반환
    parser.add_argument('--align', action='store_true', help='processing aligned dataset or not')
    parser.add_argument('--verbose', action='store_true', help='show processing room name or not')

    args_cfg = parser.parse_args()

    return args_cfg


# patch -ruN -p0 -d  raw < s3dis.patch
if __name__ == '__main__':
    args = get_parser() # args.data_root, args.save_root 등등..
    data_root = args.data_root
    # processed data output dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True) # S3DIS/preprocess/ 폴더 생성
    if args.patch: # default는 False
        if args.align:  # processing aligned s3dis dataset
            os.system(
                f"patch -ruN -p0 -d  {data_root} < {osp.join(osp.dirname(__file__), 's3dis_align.patch')}"  # noqa
            )
            # rename to avoid room_name conflict
            if osp.exists(osp.join(data_root, 'Area_6', 'copyRoom_1', 'copy_Room_1.txt')):
                os.rename(
                    osp.join(data_root, 'Area_6', 'copyRoom_1', 'copy_Room_1.txt'),
                    osp.join(data_root, 'Area_6', 'copyRoom_1', 'copyRoom_1.txt'))
        else:
            os.system(
                f"patch -ruN -p0 -d  {data_root} < {osp.join(osp.dirname(__file__), 's3dis.patch')}"
            )

    area_list = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    # area_list = ['Area_1']
    # area_list = ['Area_2']
    # area_list = ['Area_3']
    # area_list = ['Area_4']
    # area_list = ['Area_5']
    # area_list = ['Area_6']

    for area_id in area_list:
        print(f'Processing: {area_id}')
        area_dir = osp.join(data_root, area_id) # ./Stanford3dDataset_v1.2/Area_1
        # get the room name list for each area
        room_name_list = os.listdir(area_dir) # room_name_list = ['hallway_1','hallway_2' ,'hallway_3',..]

        try:
            room_name_list.remove(f'{area_id}_alignmentAngle.txt')
            room_name_list.remove('.DS_Store')
        except:  # noqa
            pass
        for room_name in room_name_list:
            scene = f'{area_id}_{room_name}' #scene = Area_1_hallway_1
            if args.verbose:
                print(f'processing: {scene}')
            save_path = osp.join(save_dir, scene + '_inst_nostuff.pth') # ./preprocess/Area1_1_hallway_1_inst_nostuff.pth 형태로 저장경로 설정
            if osp.exists(save_path):
                continue
            if room_name == '.DS_Store':
                continue
            if read_s3dis_format(area_id, room_name, data_root) == None:
                continue
            else: # read_s3dis_format method 실행
                (xyz, rgb, semantic_labels, instance_labels,
                room_label) = read_s3dis_format(area_id, room_name, data_root)
                rgb = (rgb / 127.5) - 1 # rgb value range = [-1,1]
                torch.save((xyz, rgb, semantic_labels, instance_labels, room_label, scene), save_path) # pth파일 형태로 저장

# parser = argparse.ArgumentParser(description='Argparse tutorial')
# parser.add_argument('--data-root',type=str)
# parser.add_argument('--age',type=int)
# parser.add_argument('-foo',type=int,default=5)
# parser.add_argument('--name',action='store_true')

# args = parser.parse_args()

# print(args.name)