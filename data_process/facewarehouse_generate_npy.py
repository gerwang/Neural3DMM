import scipy.io as sio
import struct
import numpy as np
import os
from sklearn.decomposition import PCA
import openmesh

n_bs = 0
n_vert = 0
n_face = 0
n_people = 150
path_prefix = '/home/jingwang/Data/data/FaceWarehouse/'
result_path = '/home/jingwang/Data/data/FaceWarehouse/preprocessed'
n_training_pose = 20

n_test = 10
n_train = n_people - n_test


def load_bs_file(bs_path):
    global n_bs, n_vert, n_face
    with open(bs_path, 'rb') as fin:
        n_bs, n_vert, n_face = struct.unpack('iii', fin.read(4 * 3))
        # 46 11510 11400
        neu_vert = np.fromfile(fin, dtype=np.float32,
                               count=(n_bs + 1) * n_vert * 3)
        return neu_vert


def load_training_pose(idx):
    ten = []
    template = os.path.join(path_prefix, 'Tester_{}'.format(
        idx), 'TrainingPose', 'pose_{}.obj')
    for i in range(n_training_pose):
        file_path = template.format(i)
        mesh = openmesh.read_trimesh(file_path)
        ten.append(mesh.points())
    ten = np.array(ten)
    return ten


def load_personalized_blendshape(idx):
    file_path = os.path.join(
        path_prefix, 'Tester_{}'.format(idx), 'Blendshape', 'shape.bs')
    tmp = load_bs_file(file_path)
    tmp = tmp.reshape(n_bs+1, n_vert, 3)
    return tmp


if __name__ == '__main__':
    ten_train = []
    ten_test = []
    id_test = list(range(1, n_test+1))
    id_train = list(range(n_test+1, n_people+1))
    for i in range(1, n_people + 1):
        train_pose = load_training_pose(i)
        blendshape = load_personalized_blendshape(i)
        if i in id_test:
            ten_test.append(train_pose)
            ten_test.append(blendshape)
        else:
            ten_train.append(train_pose)
            ten_train.append(blendshape)
        print('done {}'.format(i))
    ten_test = np.concatenate(ten_test, axis=0)
    ten_train = np.concatenate(ten_train, axis=0)
    print(ten_test.shape, ten_train.shape)
    np.save(os.path.join(result_path, 'train.npy'), ten_train)
    np.save(os.path.join(result_path, 'test.npy'), ten_test)
