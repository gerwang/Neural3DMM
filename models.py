import torch
import torch.nn as nn

import pdb


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encode_fc1 = nn.Linear(11511 * 3, 1024)
        self.encode_fc2 = nn.Linear(1024, 128)
        self.decode_fc1 = nn.Linear(128, 1024)
        self.decode_fc2 = nn.Linear(1024, 11511 * 3)
        self.activation = nn.ELU()
        self._only_encode = False
        self._only_decode = False

    def only_encode(self, status):
        self._only_encode = status
    
    def only_decode(self, status):
        self._only_decode = status

    def encode(self, x):
        x = self.encode_fc1(x)
        x = self.activation(x)
        x = self.encode_fc2(x)
        return x

    def decode(self, x):
        x = self.decode_fc1(x)
        x = self.activation(x)
        x = self.decode_fc2(x)
        return x

    def forward(self, x):
        if self._only_decode:
            x = self.decode(x)
            return x
        old_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.encode(x)
        z = x
        if self._only_encode:
            return z
        x = self.decode(x)
        x = x.reshape(old_shape)
        return x


class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True):
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.conv = nn.Linear(in_c*spiral_size, out_c, bias=bias)

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

        # [1d array of batch,vertx,vertx-adj]
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size)
        batch_index = torch.arange(bsize, device=x.device).view(-1, 1).repeat(
            [1, num_pts*spiral_size]).view(-1).long()  # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(
            bsize*num_pts, spiral_size*feats)  # [bsize*numpt, spiral*feats]

        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=x.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class FakeArray(object):
    def __init__(self, master, n, template):
        self.master = master
        self.n = n
        self.template = template

    def __getitem__(self, x):
        x %= self.n
        if x < 0:
            x += self.n
        return getattr(self.master, self.template % x)


class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, activation='elu'):
        super(SpiralAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes

        self.spirals = spirals
        self.spiral_template = 'spirals_%d'
        for i in range(len(self.spirals)):
            self.register_buffer(self.spiral_template % i, self.spirals[i])

        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes

        self.D = D
        self.D_template = 'D_%d'
        for i in range(len(self.D)):
            self.register_buffer(self.D_template % i, self.D[i])
        self.U = U
        self.U_template = 'U_%d'
        for i in range(len(self.U)):
            self.register_buffer(self.U_template % i, self.U[i])

        self.activation = activation

        conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                       activation=self.activation))
                input_size = filters_enc[1][i]

            conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                   activation=self.activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(conv)
        # self.conv_template = 'conv_%d'
        # for i in range(len(self.conv)):
        #     self.add_module(self.conv_template % i, self.conv[i])

        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(
            latent_size, (sizes[-1]+1)*filters_dec[0][0])

        dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                        activation=self.activation))
                input_size = filters_dec[0][i+1]

                if filters_dec[1][i+1]:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[1][i+1],
                                            activation=self.activation))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                            activation=self.activation))
                    input_size = filters_dec[0][i+1]
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[1][i+1],
                                            activation='identity'))
                    input_size = filters_dec[1][i+1]
                else:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                            activation='identity'))
                    input_size = filters_dec[0][i+1]

        self.dconv = nn.ModuleList(dconv)
        # self.dconv_template = 'dconv_%d'
        # for i in range(len(self.dconv)):
        #     self.add_module(self.dconv_template % i, self.dconv[i])
        self.only_decode = False
        self.only_encode = False

    def encode(self, x):
        bsize = x.size(0)
        S = FakeArray(self, len(self.spirals), self.spiral_template)
        D = FakeArray(self, len(self.D), self.D_template)
        conv = self.conv

        j = 0
        for i in range(len(self.spiral_sizes)-1):
            # print(x.device, S[i].repeat(bsize,1,1).device, conv[j].conv.weight.device)
            x = conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
        x = x.view(bsize, -1)
        return self.fc_latent_enc(x)

    def decode(self, z):
        bsize = z.size(0)
        S = FakeArray(self, len(self.spirals), self.spiral_template)
        U = FakeArray(self, len(self.U), self.U_template)
        dconv = self.dconv

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1]+1, -1)
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i], x)
            x = dconv[j](x, S[-2-i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i+1]:
                x = dconv[j](x, S[-2-i].repeat(bsize, 1, 1))
                j += 1
        return x

    def forward(self, x):
        if self.only_encode:
            z = self.encode(x)
            return z
        if self.only_decode:
            z = x
            x = self.decode(z)
            return x
        z = self.encode(x)
        x = self.decode(z)
        return x


class SpiralAutoencoderVariationalLoss(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, activation='elu'):
        super(SpiralAutoencoderVariationalLoss, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes

        self.spirals = spirals
        self.spiral_template = 'spirals_%d'
        for i in range(len(self.spirals)):
            self.register_buffer(self.spiral_template % i, self.spirals[i])

        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes

        self.D = D
        self.D_template = 'D_%d'
        for i in range(len(self.D)):
            self.register_buffer(self.D_template % i, self.D[i])
        self.U = U
        self.U_template = 'U_%d'
        for i in range(len(self.U)):
            self.register_buffer(self.U_template % i, self.U[i])

        self.activation = activation

        conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                       activation=self.activation))
                input_size = filters_enc[1][i]

            conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                   activation=self.activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(conv)
        # self.conv_template = 'conv_%d'
        # for i in range(len(self.conv)):
        #     self.add_module(self.conv_template % i, self.conv[i])

        self.fc_latent_enc_mu = nn.Linear(
            (sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_enc_logvar = nn.Linear(
            (sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(
            latent_size, (sizes[-1]+1)*filters_dec[0][0])

        dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                        activation=self.activation))
                input_size = filters_dec[0][i+1]

                if filters_dec[1][i+1]:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[1][i+1],
                                            activation=self.activation))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                            activation=self.activation))
                    input_size = filters_dec[0][i+1]
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[1][i+1],
                                            activation='identity'))
                    input_size = filters_dec[1][i+1]
                else:
                    dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                            activation='identity'))
                    input_size = filters_dec[0][i+1]

        self.dconv = nn.ModuleList(dconv)
        # self.dconv_template = 'dconv_%d'
        # for i in range(len(self.dconv)):
        #     self.add_module(self.dconv_template % i, self.dconv[i])

        self._only_encode = False
        self._only_decode = False
    
    def only_encode(self, status):
        self._only_encode = status
    
    def only_decode(self, status):
        self._only_decode = status

    def encode(self, x):
        bsize = x.size(0)
        S = FakeArray(self, len(self.spirals), self.spiral_template)
        D = FakeArray(self, len(self.D), self.D_template)
        conv = self.conv

        j = 0
        for i in range(len(self.spiral_sizes)-1):
            # print(x.device, S[i].repeat(bsize,1,1).device, conv[j].conv.weight.device)
            x = conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
        x = x.view(bsize, -1)
        return self.fc_latent_enc_mu(x), self.fc_latent_enc_logvar(x)

    def decode(self, z):
        bsize = z.size(0)
        S = FakeArray(self, len(self.spirals), self.spiral_template)
        U = FakeArray(self, len(self.U), self.U_template)
        dconv = self.dconv

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1]+1, -1)
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i], x)
            x = dconv[j](x, S[-2-i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i+1]:
                x = dconv[j](x, S[-2-i].repeat(bsize, 1, 1))
                j += 1
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+eps*std

    def forward(self, x):
        if self._only_decode:
            x = self.decode(x)
            return x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self._only_encode:
            return z
        x = self.decode(z)
        return x, mu, logvar
