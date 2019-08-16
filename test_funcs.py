import torch
import copy
from tqdm import tqdm
import numpy as np
import os
import heapq

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, worst_face_num, worst_path, mm_constant = 1000, save_recons=True, samples_dir=None):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)

    min_heap = []

    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
                
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:,:-1]
                x = tx[:,:-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            
            old_x_recon = x_recon
            old_x = x

            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            per_l2_loss = torch.sqrt(torch.sum((x_recon - x)**2,dim=2))

            for j in range(per_l2_loss.shape[0]):
                payload = (torch.mean(per_l2_loss[j]), i *
                           per_l2_loss.shape[0]+j, old_x[j], old_x_recon[j])
                # print(payload[2].shape, payload[3].shape)
                # print('pushing',payload)
                if len(min_heap) < worst_face_num:
                    heapq.heappush(min_heap, payload)
                else:
                    heapq.heappushpop(min_heap, payload)
            l2_loss+= torch.mean(per_l2_loss)*x.shape[0]/float(len(dataloader_test.dataset))

        msh = []
        msh_recon = []
        msh_ind = []
        for loss, idx, x, x_recon in min_heap:
            print('loss for ', idx, loss)
            msh_ind.append(idx)
            msh.append(x.detach().cpu().numpy())
            msh_recon.append(x_recon.detach().cpu().numpy())
        msh = np.array(msh)
        msh_recon = np.array(msh_recon) # prevbug msh_ind msh_recon
        shapedata.save_meshes(os.path.join(worst_path, 'input'), msh, msh_ind)
        shapedata.save_meshes(os.path.join(
            worst_path, 'worst'), msh_recon, msh_ind)

            
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()

    return predictions, l1_loss, l2_loss
