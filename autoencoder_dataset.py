import os

import numpy as np
import torch
from torch.utils.data import Dataset


class autoencoder_dataset(Dataset):

    # no reason to use a dummy node
    def __init__(self, root_dir, points_dataset, shapedata, normalization=True, dummy_node=True):

        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_' + points_dataset + '.npy'))
        self.buffer = torch.empty((len(self.paths), shapedata.n_vertex + 1, shapedata.n_features), dtype=torch.float32,
                                  device=torch.device('cuda'))
        self.loaded = torch.zeros(len(self.paths), dtype=torch.int8)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.loaded[idx] != 0:
            verts = self.buffer[idx]
        else:
            basename = self.paths[idx]
            if hasattr(basename, 'decode'):  # python2
                basename = basename.decode('utf-8')

            verts_init = np.load(os.path.join(self.root_dir, 'points' + '_' + self.points_dataset, basename + '.npy'))
            if self.normalization:
                verts_init = verts_init - self.shapedata.mean
                verts_init = verts_init / self.shapedata.std
            verts_init[np.where(np.isnan(verts_init))] = 0.0

            verts_init = verts_init.astype('float32')
            if self.dummy_node:
                verts = np.zeros((verts_init.shape[0] + 1, verts_init.shape[1]), dtype=np.float32)
                verts[:-1, :] = verts_init
                verts_init = verts
            verts = torch.tensor(verts_init, device=torch.device('cuda'))
            self.buffer[idx] = verts
            self.loaded[idx] = 1

        sample = {'points': verts}

        return sample


class AutoEncoderDatasetWithTag(autoencoder_dataset):
    def __init__(self, root_dir, points_dataset, shapedata, tags, normalization=True, dummy_node=True):
        super().__init__(root_dir, points_dataset, shapedata, normalization, dummy_node)
        self.tags = torch.as_tensor(tags)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['tag'] = self.tags[idx]
        return sample


class DeviceDataset(Dataset):
    def __init__(self, np_data, np_tags, shapedata, device, n_id, n_exp, dummy_node=True):
        self.shapedata = shapedata
        self.device = device
        if dummy_node:
            self.data = torch.zeros((np_data.shape[0], np_data.shape[1] + 1, np_data.shape[2]), dtype=torch.float32,
                                    device=device)
            self.data[:, :-1, :] = torch.as_tensor(np_data)
        else:
            self.data = torch.as_tensor(np_data).to(device)
        self.dummy_node = dummy_node
        self.tags = torch.as_tensor(np_tags)
        self.n_id = n_id
        self.n_exp = n_exp

    def get_idx(self, id_idx, exp_idx):
        return id_idx * self.n_exp + exp_idx

    def __len__(self):  # prevbug: __len__ vs. len
        return self.data.shape[0]

    def __getitem__(self, idx):
        verts = self.data[idx]
        sample = {
            'points': verts,
            'id_targets': self.get_id_target(idx),
            'exp_targets': self.get_exp_target(idx),
            'stacked_mean': self.get_mean_target()
        }
        return sample

    def get_id_target(self, idx):
        id_t = self.tags[idx, 0]
        exp_t = self.tags[idx, 1]
        if id_t == -1:
            if exp_t != 0:
                raise RuntimeError('invalid exp')
            target = self.data[idx]
        else:
            target = self.data[self.get_idx(id_t, 0)]  # id starts at 1, but 0 is for mean face
        return target

    def get_exp_target(self, idx):
        exp_t = self.tags[idx, 1]
        target = self.data[self.get_idx(0, exp_t)]
        return target

    def get_mean_target(self):
        target = self.data[self.get_idx(0, 0)]
        return target
