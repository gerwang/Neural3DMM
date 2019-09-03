from torch.utils.data import Dataset
import torch
import numpy as np
import os
import math


class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, normalization = True, dummy_node = True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        
        verts_init = np.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.npy'))
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init/self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))]=0.0
        
        verts_init = verts_init.astype('float32')
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
            verts[:-1,:] = verts_init
            verts_init = verts
        verts = torch.Tensor(verts_init)
        

        sample = {'points': verts}

        return sample

class cached_autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, device, normalization = True, dummy_node = True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))
        self.device = device
        self.cache = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        basename = str(self.paths[idx])
        
        verts_init = np.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.npy'))
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init/self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))]=0.0
        
        verts_init = verts_init.astype('float32')
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
            verts[:-1,:] = verts_init
            verts_init = verts
        verts = torch.Tensor(verts_init).to(self.device)
        

        sample = {'points': verts}
        
        self.cache[idx] = sample
        return sample

def getRowCol(idx, n):
    neg_b = 2*n-1
    four_ac = 8*(idx+1)
    x = (neg_b-math.sqrt(neg_b**2-four_ac))/2.0
    x = int(math.ceil(x))-1
    y = idx - x*(2*n-x-1)/2
    return x,n-1-y


class PairedExampleDataSet(Dataset):

    def __init__(self, train_file_path, n_exp):
        self.models = np.load(train_file_path)
        self.n_exp = n_exp
        assert self.models.shape[0] % n_exp == 0
        self.n_id = int(self.models.shape[0]/n_exp)

        def c2(n):
            return n*(n-1)/2

        self.n_id_combine = c2(self.n_id)
        self.n_exp_combine = c2(self.n_exp)
        self.n_total = self.n_id_combine*self.n_exp_combine
    
    def __len__(self):
        return self.n_total

    def get_tensor(self, id_idx, exp_idx):
        idx = id_idx*self.n_exp_combine+exp_idx
        return self.models[idx]

    def __getitem__(self, idx):
        id_comb_idx = idx/self.n_exp_combine
        exp_comb_idx = idx % self.n_exp_combine
        idA, idB = getRowCol(id_comb_idx,self.n_id)
        exp1, exp2 = getRowCol(exp_comb_idx, self.n_exp)
        return np.concatenate([self.get_tensor(idA, exp1), self.get_tensor(idA, exp2),
                               self.get_tensor(idB, exp1), self.get_tensor(idB, exp2)])