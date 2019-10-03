import os

import numpy as np
import torch
from torch.utils.data import Dataset


class autoencoder_dataset(Dataset):

    # no reason to use a dummy node
    def __init__(self, root_dir, n_sample, shapedata, normalization=True, dummy_node=True):

        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.n_sample = n_sample
        self.dummy_node = dummy_node

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        basename = os.path.join(self.root_dir, '{}.npy'.format(idx))

        verts_init = np.load(basename)
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init / self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))] = 0.0

        verts_init = verts_init.astype(np.float32)
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0] + 1, verts_init.shape[1]), dtype=np.float32)
            verts[:-1, :] = verts_init
            verts_init = verts
        verts = torch.as_tensor(verts_init)

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
    def __init__(self, np_data, shapedata, device, dummy_node=True):
        self.shapedata = shapedata
        self.device = device
        if dummy_node:
            self.data = torch.zeros((np_data.shape[0], np_data.shape[1] + 1, np_data.shape[2]), dtype=torch.float32,
                                    device=device)
            self.data[:, :-1, :] = torch.as_tensor(np_data)
        else:
            self.data = torch.as_tensor(np_data).to(device)
        self.dummy_node = dummy_node

    def __len__(self):  # prevbug: __len__ vs. len
        return self.data.shape[0]

    def __getitem__(self, idx):
        verts = self.data[idx]
        sample = {
            'points': verts
        }
        return sample
