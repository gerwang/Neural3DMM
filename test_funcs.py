import copy

import torch
from tqdm import tqdm

from train_funcs import dict_to_device


# evaluate other metric on test dataset

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.as_tensor(shapedata.mean)
    shapedata_std = torch.as_tensor(shapedata.std)
    keys_of_interest = ['ori_rec', 'id_rec', 'exp_rec', 'id_cycle_rec', 'exp_cycle_rec']
    total_dict = {}

    def get_recon(x):
        if dataloader_test.dataset.dummy_node:
            x = x[:, :-1]
        x = x.detach().cpu()
        return x

    def get_denormalized(x):
        x = x * shapedata_std + shapedata_mean
        return x

    def get_dict_recon(dic):
        res = {}
        for key in keys_of_interest:
            res[key] = get_recon(dic[key])
        return res

    def concat_dict(total_dict, dic):
        if total_dict == {}:
            return copy.deepcopy(dic)
        else:
            return {key: torch.cat([total_dict[key], dic[key]], dim=0) for key in keys_of_interest}

    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            sample_dict = dict_to_device(sample_dict, device)
            tx = sample_dict['points']
            tx_dict = model(tx)
            tx_dict = get_dict_recon(tx_dict)
            x_recon = tx_dict['ori_rec']
            x = tx[:, :-1].cpu()

            total_dict = concat_dict(total_dict, tx_dict)

            l1_loss += torch.mean(torch.abs(x_recon - x)) * x.shape[0] / float(len(dataloader_test.dataset))

            x_recon = get_denormalized(x_recon)
            x = get_denormalized(x)

            l2_loss += mm_constant * torch.mean(torch.sqrt(torch.sum((x_recon - x) ** 2, dim=2))) * x.shape[0] / float(
                len(dataloader_test.dataset))

        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()

    return total_dict, l1_loss, l2_loss
