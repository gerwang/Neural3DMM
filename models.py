import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True):
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.conv = nn.Linear(in_c * spiral_size, out_c, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(bsize * num_pts * spiral_size)  # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1, 1).repeat([1, num_pts * spiral_size]).view(
            -1).long()  # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(bsize * num_pts,
                                                        spiral_size * feats)  # [bsize*numpt, spiral*feats]

        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=x.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class FakeArray(object):
    def __init__(self, master, name, n, buffer_list=None):
        self.master = master
        self.n = n
        self.name = name
        self.template = name + '_%d'
        if buffer_list is not None:
            for i in range(self.n):
                self.master.register_buffer(self.template % i, buffer_list[i])

    def __len__(self):
        return self.n

    def __getitem__(self, x):
        x %= self.n
        if x < 0:
            x += self.n
        return getattr(self.master, self.template % x)


class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, id_latent_size, exp_latent_size, sizes, spiral_sizes,
                 spirals, D, U, activation='elu'):
        super(SpiralAutoencoder, self).__init__()
        self.id_latent_size = id_latent_size
        self.exp_latent_size = exp_latent_size
        self.latent_size = id_latent_size + exp_latent_size
        self.sizes = sizes
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.spirals = FakeArray(self, 'spirals', len(spirals), spirals)
        self.D = FakeArray(self, 'D', len(D), D)
        self.U = FakeArray(self, 'U', len(U), U)
        self.activation = activation

        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes) - 1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i + 1],
                                        activation=self.activation))
            input_size = filters_enc[0][i + 1]

        self.conv = nn.ModuleList(self.conv)

        self.fc_mu_enc = nn.Linear((sizes[-1] + 1) * input_size, self.latent_size)
        self.fc_logvar_enc = nn.Linear((sizes[-1] + 1) * input_size, self.latent_size)
        self.fc_latent_dec = nn.Linear(self.latent_size, (sizes[-1] + 1) * filters_dec[0][0])

        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes) - 1):
            if i != len(spiral_sizes) - 2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                             activation=self.activation))
                input_size = filters_dec[0][i + 1]

                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation=self.activation))
                    input_size = filters_dec[1][i + 1]
            else:
                if filters_dec[1][i + 1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation=self.activation))
                    input_size = filters_dec[0][i + 1]
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[1][i + 1],
                                                 activation='identity'))
                    input_size = filters_dec[1][i + 1]
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2 - i], filters_dec[0][i + 1],
                                                 activation='identity'))
                    input_size = filters_dec[0][i + 1]

        self.dconv = nn.ModuleList(self.dconv)

    def encode(self, x):
        bsize = x.size(0)
        # S = self.spirals
        S = FakeArray(self, 'spirals', len(self.spirals))
        # D = self.D
        D = FakeArray(self, 'D', len(self.D))

        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
        x = x.view(bsize, -1)
        mu = self.fc_mu_enc(x)
        logvar = self.fc_logvar_enc(x)
        if self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z):
        bsize = z.size(0)
        # S = self.spirals
        S = FakeArray(self, 'spirals', len(self.spirals))
        # U = self.U
        U = FakeArray(self, 'U', len(self.U))

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)
        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = torch.matmul(U[-1 - i], x)
            x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i + 1]:
                x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
                j += 1
        return x

    def split_id_exp(self, z):
        id_z = z[:, :self.id_latent_size]
        exp_z = z[:, self.id_latent_size:]
        return id_z, exp_z

    def forward(self, x):
        bsize = x.size(0)
        z, mu, logvar = self.encode(x)

        id_z, exp_z = self.split_id_exp(z)

        id_mean = torch.zeros_like(id_z)
        exp_mean = torch.zeros_like(exp_z)

        id_only_z = torch.cat([id_z, exp_mean], dim=1)
        exp_only_z = torch.cat([id_mean, exp_z], dim=1)

        ori_rec = self.decode(z)
        id_rec = self.decode(id_only_z)
        exp_rec = self.decode(exp_only_z)

        id_cycle_z, _, _ = self.encode(id_rec)
        exp_cycle_z, _, _ = self.encode(exp_rec)

        id_cycle_id_z, id_cycle_exp_z = self.split_id_exp(id_cycle_z)
        exp_cycle_id_z, exp_cycle_exp_z = self.split_id_exp(exp_cycle_z)

        id_cycle_nothing_z = torch.cat([id_mean, id_cycle_exp_z], dim=1)
        exp_cycle_nothing_z = torch.cat([exp_cycle_id_z, exp_mean], dim=1)

        id_cycle_rec = self.decode(id_cycle_nothing_z)
        exp_cycle_rec = self.decode(exp_cycle_nothing_z)

        return {
            'ori_rec': ori_rec,
            'id_rec': id_rec,
            'exp_rec': exp_rec,
            'id_cycle_rec': id_cycle_rec,
            'exp_cycle_rec': exp_cycle_rec,
            'mu': mu,
            'logvar': logvar
        }
        # I feel the routine is over-complex: We KNOW mean's id latent and exp latent
