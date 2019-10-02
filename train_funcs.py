import os

import torch
from tqdm import tqdm


def dict_to_device(input_dict, device):
    res = {}
    for key, value in input_dict.items():
        res[key] = value.to(device)
    return res


def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn,
                                 bsize, start_epoch, n_epochs, eval_freq, scheduler=None,
                                 writer=None, save_recons=True, shapedata=None,
                                 metadata_dir=None, samples_dir=None, checkpoint_path=None):
    total_steps = start_epoch * len(dataloader_train)

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tloss = []
        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
            sample_dict = dict_to_device(sample_dict, device)

            tx = sample_dict['points']
            cur_bsize = tx.shape[0]

            tx_dict = model(tx)

            loss, loss_exp, loss_exp_cycle, loss_id, loss_id_cycle, loss_ori = compute_overall_loss(sample_dict,
                                                                                                    loss_fn, tx_dict)

            loss.backward()
            optim.step()

            tloss.append(cur_bsize * loss.item())
            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss', loss.item(), total_steps)
                writer.add_scalar('loss/loss/loss_ori', loss_ori.item(), total_steps)
                writer.add_scalar('loss/loss/loss_id', loss_id.item(), total_steps)
                writer.add_scalar('loss/loss/loss_exp', loss_exp.item(), total_steps)
                writer.add_scalar('loss/loss/loss_id_cycle', loss_id_cycle.item(), total_steps)
                writer.add_scalar('loss/loss/loss_exp_cycle', loss_exp_cycle.item(), total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'], total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b, sample_dict in enumerate(tqdm(dataloader_val)):
                sample_dict = dict_to_device(sample_dict, device)
                tx = sample_dict['points']
                cur_bsize = tx.shape[0]

                tx_dict = model(tx)
                loss, loss_exp, loss_exp_cycle, loss_id, loss_id_cycle, loss_ori = compute_overall_loss(sample_dict,
                                                                                                        loss_fn,
                                                                                                        tx_dict)

                vloss.append(cur_bsize * loss.item())

        if scheduler:
            scheduler.step()

        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss', epoch_tloss, epoch)
        if len(dataloader_val.dataset) > 0:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss, epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch, epoch_tloss, epoch_vloss))
        else:
            print('epoch {0} | tr {1} '.format(epoch, epoch_tloss))
        model = model.cpu()

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(metadata_dir, checkpoint_path + '.pth.tar'))

        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                        'autoencoder_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(metadata_dir, checkpoint_path + '%s.pth.tar' % (epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir, 'input_epoch_{0}'.format(epoch)),
                                          msh, mesh_ind)
                mesh_ind = [0]
                msh = tx_dict['ori_rec'][mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()
                shapedata.save_meshes(os.path.join(samples_dir, 'epoch_{0}'.format(epoch)),
                                      msh, mesh_ind)

    print('~FIN~')


def compute_overall_loss(sample_dict, loss_fn, tx_dict):
    ori_rec = tx_dict['ori_rec']  # should be same as tx
    id_rec = tx_dict['id_rec']  # should be same as
    exp_rec = tx_dict['exp_rec']
    id_cycle_rec = tx_dict['id_cycle_rec']
    exp_cycle_rec = tx_dict['exp_cycle_rec']
    ori_targets = sample_dict['points']
    id_targets = sample_dict['id_targets']
    exp_targets = sample_dict['exp_targets']
    id_cycle_targets = exp_cycle_targets = sample_dict['stacked_mean']
    loss_ori = loss_fn(ori_rec, ori_targets)
    loss_id = loss_fn(id_rec, id_targets)
    loss_exp = loss_fn(exp_rec, exp_targets)
    loss_id_cycle = loss_fn(id_cycle_rec, id_cycle_targets)
    loss_exp_cycle = loss_fn(exp_cycle_rec, exp_cycle_targets)
    loss = loss_ori + loss_id + loss_exp + loss_id_cycle + loss_exp_cycle
    return loss, loss_exp, loss_exp_cycle, loss_id, loss_id_cycle, loss_ori
