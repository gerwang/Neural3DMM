from torch.utils.data import Dataset
import torch
import numpy as np
import os


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
            verts = torch.Tensor(verts_init).cuda()
            self.buffer[idx] = verts
            self.loaded[idx] = 1

        sample = {'points': verts}

        return sample
