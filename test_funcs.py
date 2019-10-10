import copy

import torch
import torch.nn as nn
from tqdm import tqdm


def get_optim_points(model, current, target, loss_fn, z, n_iter=1000):
    params = []
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            params.append(p)
    z = nn.Parameter(z)
    optim = torch.optim.Adam([z], lr=5e-2)
    for i in range(n_iter):
        optim.zero_grad()
        current = model.decode(z.unsqueeze(0)).squeeze()
        loss = loss_fn(current, target)
        # print('iter {}, loss {}'.format(i, loss.item()))
        loss.backward()
        optim.step()
    for p in params:
        p.requires_grad_(True)
    return current, z


def loss_l1(outputs, targets):
    L = torch.abs(outputs - targets).mean()
    return L


def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            tx_dict = model(tx)
            prediction = tx_dict['rec']
            prediction = torch.cat([
                get_optim_points(model, prediction[i], tx[i], loss_l1, tx_dict['mu'][i])[0].unsqueeze(0)
            ], dim=0)
            if i == 0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions, prediction], 0)

            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:, :-1]
                x = tx[:, :-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss += torch.mean(torch.abs(x_recon - x)) * x.shape[0] / float(len(dataloader_test.dataset))

            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss += torch.mean(torch.sqrt(torch.sum((x_recon - x) ** 2, dim=2))) * x.shape[0] / float(
                len(dataloader_test.dataset))

        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()

    return predictions, l1_loss, l2_loss
