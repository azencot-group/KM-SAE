import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import numpy as np
from torch import nn
import torchvision.transforms as T
from koopman_utils import get_unique_num, get_sorted_indices, static_dynamic_split
from utils import t_to_np, np_to_t


class KoopmanCNN(nn.Module):

    def __init__(self, args):
        super(KoopmanCNN, self).__init__()

        self.args = args

        self.encoder = encNet(self.args)
        self.drop = torch.nn.Dropout(self.args.dropout)
        self.dynamics = KoopmanLayer(args)
        self.decoder = decNet(self.args)

        self.loss_func = nn.MSELoss()

        self.names = ['total', 'rec', 'predict_ambient', 'predict_latent', 'eigs']

    def forward(self, X, train=True):
        # input noise added for stability of the Koopman matrix calculation
        if train and self.args.noise in ["input"]:
            blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3))
            X = torch.concat([torch.concat([blurrer(x).unsqueeze(0) for x in X], dim=0) for _ in range(1)])

        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)

        # latent both noise - another option to stabilize the numeric calculation of the Koopman matrix
        if train and self.args.noise in ["latent_both"]:
            Z = Z + 0.25 * torch.rand(Z.shape).to(Z.device)

        Z2, Ct = self.dynamics(Z)
        Z = self.drop(Z)

        # latent reconstruction noise
        if train and self.args.noise in ["latent_rec"]:
            Z = Z + 0.25 * torch.rand(Z.shape).to(Z.device)

        X_dec = self.decoder(Z)
        X_dec2 = self.decoder(Z2)

        return X_dec, X_dec2, Z, Z2, Ct

    def decode(self, Z):
        X_dec = self.decoder(Z)

        return X_dec

    def loss(self, X, outputs):
        X_dec, X_dec2, Z, Z2, Ct = outputs

        # PENALTIES
        a1 = self.args.w_rec
        a2 = self.args.w_pred
        a3 = self.args.w_pred
        a4 = self.args.w_eigs

        # reconstruction
        E1 = self.loss_func(X, X_dec)

        # Koopman losses
        E2, E3, E4 = self.dynamics.loss(X_dec, X_dec2, Z, Z2, Ct)

        # LOSS
        loss = a1 * E1 + a2 * E2 + a3 * E3 + a4 * E4

        return loss, E1, E2, E3, E4

    def swap(self, Zp, I, J, U):
        # swap J factors with shuffle I,
        import copy

        # swap certain features
        Zp_tmp = copy.deepcopy(Zp)
        Zp_tmp[:, :, J] = Zp_tmp[I][:, :, J]
        Z_tmp = Zp_tmp @ U

        self.eval()
        with torch.no_grad():
            X_tmp = self.decode(np_to_t(np.real(Z_tmp))).squeeze()

        return X_tmp

    def factorial_swap(self, classifier, X, Zp, I, J, U):
        # swap J factors with shuffle I, and eval accuracy
        get_lbl = lambda pred: np.argmax(t_to_np(pred), axis=1)
        get_acc = lambda lbl1, lbl2: np.sum(lbl1 == lbl2) / len(lbl2)

        classifier.eval()
        with torch.no_grad():
            # action, skin, pant, top, hair
            preds = classifier(X[I])

        lbls = list(map(get_lbl, preds))

        import copy

        # swap certain features
        Zp_tmp = copy.deepcopy(Zp)
        Zp_tmp[:, :, J] = Zp_tmp[I][:, :, J]
        Z_tmp = Zp_tmp @ U

        self.eval()
        with torch.no_grad():
            X_tmp = self.decode(np_to_t(np.real(Z_tmp))).squeeze()

        classifier.eval()
        with torch.no_grad():
            preds_tmp = classifier(X_tmp)
        lbls_tmp = list(map(get_lbl, preds_tmp))

        accs = list(map(get_acc, lbls_tmp, lbls))
        return accs, lbls_tmp, lbls

    def forward_fixed_ma_for_classification(self, X, fix_motion, conj_pick=True, pick_type='norm'):
        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)
        Z2, Ct = self.dynamics(Z)
        Z = self.drop(Z)

        Z_old_shape = Z.shape

        # swap a single pair in batch
        bsz, fsz = X.shape[0:2]

        # swap contents of samples in indices
        X = t_to_np(X)
        Z = t_to_np(Z.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)
        swapped_Z = torch.zeros(Z.shape)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # static/dynamic split
        if pick_type == 'real':
            I = np.argsort(np.real(D))
        elif pick_type == 'norm':
            I = np.argsort(np.abs(D))
        else:
            raise Exception("no such method")

        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, self.args.static_size)

        for ii in range(bsz):
            iir = np.random.randint(bsz)
            while iir == ii:
                iir = np.random.randint(bsz)
            S1, Z1 = X[ii].squeeze(), Z[ii].squeeze()
            S2, Z2 = X[iir].squeeze(), Z[iir].squeeze()

            # project onto V
            Zp1, Zp2 = Z1 @ V, Z2 @ V

            # Zp* is in t x k
            Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
            Z2d, Z2s = Zp2[:, Id] @ U[Id], Zp2[:, Is] @ U[Is]

            if fix_motion:
                # we fix dynamics thus, use same d for our sample
                swapped_Z[ii] = torch.from_numpy(np.real(Z1d + Z2s)).to(self.args.device)
            else:
                swapped_Z[ii] = torch.from_numpy(np.real(Z2d + Z1s)).to(self.args.device)

        ZNs = torch.from_numpy(Z).to(self.args.device)
        Z = swapped_Z.to(self.args.device)

        X_dec_sample = self.decoder(Z.reshape(Z_old_shape))
        X_dec = self.decoder(ZNs.reshape(Z_old_shape))

        return X_dec_sample, X_dec

    def forward_sample_for_classification(self, X, fix_motion, pick_type='real'):
        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)
        Z2, Ct = self.dynamics(Z)
        # Z = self.drop(Z)

        Z_old_shape = Z.shape

        # swap a single pair in batch
        bsz, fsz = X.shape[0:2]

        # swap contents of samples in indices
        X = t_to_np(X)
        Z = t_to_np(Z.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)
        swapped_Z = torch.zeros(Z.shape)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # static/dynamic split
        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, self.args.static_size)

        convex_size = 5

        for ii in range(bsz):
            S1, Z1 = X[ii].squeeze(), Z[ii].squeeze()

            A = np.random.rand(convex_size)  # random coefs
            Ac = np.exp(A) / sum(np.exp(A))  # normalize sum to one
            Ac = np.expand_dims(Ac, axis=1) @ np.ones((1, convex_size))

            J = np.random.permutation(Z.shape[0])[:convex_size]
            Zc = Z[J, 0]
            Z2 = np.sum((Ac @ t_to_np(Zc)), axis=0)  # the convex combination

            # project onto V
            Zp1, Zp2 = Z1 @ V, Z2 @ V

            # Zp* is in t x k
            Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
            Z2d, Z2s = Zp2[Id] @ U[Id], np.repeat((Zp2[Is] @ U[Is])[None], fsz, 0)

            if fix_motion:
                # we fix dynamics thus, use same d for our sample
                swapped_Z[ii] = torch.from_numpy(np.real(Z1d + Z2s)).to(self.args.device)
            else:
                swapped_Z[ii] = torch.from_numpy(np.real(Z2d + Z1s)).to(self.args.device)

        ZNs = torch.from_numpy(Z).to(self.args.device)
        Z = swapped_Z.to(self.args.device)

        X_dec_sample = self.decoder(Z.reshape(Z_old_shape))
        X_dec = self.decoder(ZNs.reshape(Z_old_shape))

        return X_dec_sample, X_dec

    def forward_sample_for_classification2(self, X, fix_motion, pick_type='real', duplicate=False):
        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)
        Z2, Ct = self.dynamics(Z)

        # swap a single pair in batch
        bsz, fsz = X.shape[0:2]

        # swap contents of samples in indices
        Z = t_to_np(Z.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # static/dynamic split
        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, self.args.static_size)

        convex_size = 2

        Js = [np.random.permutation(bsz) for _ in range(convex_size)]  # convex_size permutations
        # J = np.random.permutation(bsz)              # bsz
        # J2 = np.random.permutation(bsz)

        A = np.random.rand(bsz, convex_size)  # bsz x 2
        A = A / np.sum(A, axis=1)[:, None]

        Zp = Z @ V

        # prev code
        # Zp1 = [a * z for a, z in zip(A[:, 0], Zp[J2])]
        # Zp2 = [a * z for a, z in zip(A[:, 1], Zp[J])]

        # bsz x time x feats
        # Zpc = np.array(Zp1) + np.array(Zp2)

        # Edit

        import functools
        Zpi = [np.array([a * z for a, z in zip(A[:, c], Zp[j])]) for c, j in enumerate(Js)]
        Zpc = functools.reduce(lambda a, b: a + b, Zpi)

        Zp2 = copy.deepcopy(Zp)
        # swap static info
        if fix_motion:
            if duplicate:
                Zp2[:, :, Is] = np.repeat(np.expand_dims(np.mean(Zpc[:, :, Is], axis=1), axis=1), 8, axis=1)
            else:
                Zp2[:, :, Is] = Zpc[:, :, Is]
        # swap dynamic info
        else:
            Zp2[:, :, Id] = Zpc[:, :, Id]

        Z2 = np.real(Zp2 @ U)

        X2_dec = self.decoder(torch.from_numpy(Z2).to(self.args.device))
        X_dec = self.decoder(torch.from_numpy(Z).to(self.args.device))

        return X2_dec, X_dec

    def forward_sample_for_classification2_specific_I(self, If, Ic, X, fix_motion, pick_type='real', duplicate=False):
        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)
        Z2, Ct = self.dynamics(Z)

        # swap a single pair in batch
        bsz, fsz = X.shape[0:2]

        # swap contents of samples in indices
        Z = t_to_np(Z.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # static/dynamic split
        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, self.args.static_size)

        convex_size = 2

        Js = [np.random.permutation(bsz) for _ in range(convex_size)]  # convex_size permutations

        A = np.random.rand(bsz, convex_size)  # bsz x 2
        A = A / np.sum(A, axis=1)[:, None]

        Zp = Z @ V

        import functools
        Zpi = [np.array([a * z for a, z in zip(A[:, c], Zp[j])]) for c, j in enumerate(Js)]
        Zpc = functools.reduce(lambda a, b: a + b, Zpi)

        Zp2 = copy.deepcopy(Zp)
        # swap static info
        if fix_motion:
            if duplicate:
                Zp2[:, :, Is] = np.repeat(np.expand_dims(np.mean(Zpc[:, :, Is], axis=1), axis=1), 8, axis=1)
            else:
                Zp2[:, :, Is] = Zpc[:, :, Is]
        # swap dynamic info
        else:
            Zp2[:, :, Id] = Zpc[:, :, Id]

        Z2 = np.real(Zp2 @ U)

        X2_dec = self.decoder(torch.from_numpy(Z2).to(self.args.device))
        X_dec = self.decoder(torch.from_numpy(Z).to(self.args.device))

        return X2_dec, X_dec


class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.net(input)


class upconv(nn.Module):
    def __init__(self, nin, nout):
        super(upconv, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.net(input)


class encNet(nn.Module):

    def __init__(self, args):
        super(encNet, self).__init__()

        self.args = args

        self.n_frames = args.n_frames
        self.n_channels = args.n_channels
        self.n_height = args.n_height
        self.n_width = args.n_width
        self.conv_dim = args.conv_dim
        self.k_dim = args.k_dim
        self.hidden_dim = args.hidden_dim

        self.c1 = conv(self.n_channels, self.conv_dim)
        self.c2 = conv(self.conv_dim, self.conv_dim * 2)
        self.c3 = conv(self.conv_dim * 2, self.conv_dim * 4)
        self.c4 = conv(self.conv_dim * 4, self.conv_dim * 8)
        self.c5 = nn.Sequential(
            nn.Conv2d(self.conv_dim * 8, self.k_dim, 4, 1, 0),
            nn.BatchNorm2d(self.k_dim),
            nn.Tanh()
        )

        if args.rnn in ["encoder", "both"]:
            self.lstm = nn.LSTM(self.k_dim, self.k_dim, batch_first=True, bias=True,
                                bidirectional=False)

    def forward(self, x):
        x = x.reshape(-1, self.n_channels, self.n_height, self.n_width)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)

        # lstm
        if self.args.rnn in ["encoder", "both"]:
            h5 = self.lstm(h5.reshape(-1, self.n_frames, self.k_dim))[0].reshape(-1, self.hidden_dim, 1, 1)

        return h5


class decNet(nn.Module):
    def __init__(self, args):
        super(decNet, self).__init__()

        self.args = args

        self.n_frames = args.n_frames
        self.n_channels = args.n_channels
        self.n_height = args.n_height
        self.n_width = args.n_width
        self.conv_dim = args.conv_dim
        self.koopman_dim = args.k_dim
        self.lstm_hidden_size = args.hidden_dim

        if args.lstm_dec_bi:
            self.koopman_dim = self.koopman_dim * 2

        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(self.koopman_dim, self.conv_dim * 8, 4, 1, 0),
            nn.BatchNorm2d(self.conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc2 = upconv(self.conv_dim * 8, self.conv_dim * 4)
        self.upc3 = upconv(self.conv_dim * 4, self.conv_dim * 2)
        self.upc4 = upconv(self.conv_dim * 2, self.conv_dim)
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(self.conv_dim, self.n_channels, 4, 2, 1),
            nn.Sigmoid()
        )

        if args.rnn in ["decoder", "both"]:
            self.lstm = nn.LSTM(self.lstm_hidden_size, self.koopman_dim, batch_first=True, bias=True,
                                bidirectional=args.lstm_dec_bi)

    def forward(self, x):
        # lstm
        if self.args.rnn in ["decoder", "both"]:
            x = self.lstm(x.reshape(-1, self.n_frames, self.lstm_hidden_size))[0].reshape(-1, self.koopman_dim, 1, 1)

        d1 = self.upc1(x)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(-1, self.n_frames, self.n_channels, self.n_height, self.n_width)

        return output


class KoopmanLayer(nn.Module):

    def __init__(self, args):
        super(KoopmanLayer, self).__init__()

        self.run = None
        self.args = args
        self.n_frames = args.n_frames
        self.k_dim = args.k_dim

        # eigen values arguments
        self.static = args.static_size
        self.mode = args.static_mode
        self.eigs_tresh = args.eigs_thresh ** 2
        self.dynamic_loss_mode = args.dynamic_mode

        # loss functions
        self.loss_func = nn.MSELoss()
        self.dynamic_threshold_loss = nn.Threshold(args.dynamic_thresh, 0)
        # self.sp_b_thresh = nn.Threshold(args.sp_b_thresh, 0)

    def forward(self, Z):
        # Z is in b * t x c x 1 x 1
        Zr = Z.squeeze().reshape(-1, self.n_frames, self.k_dim)

        if self.training and self.args.noise in ["latent"]:
            Zr = Zr + 0.003 * torch.rand(Zr.shape).to(Zr.device)

        # split
        X, Y = Zr[:, :-1], Zr[:, 1:]

        # solve linear system (broadcast)
        # Ct = torch.linalg.pinv(X.reshape(-1, self.k_dim)) @ Y.reshape(-1, self.k_dim)
        Ct = torch.linalg.lstsq(X.reshape(-1, self.k_dim), Y.reshape(-1, self.k_dim)).solution

        # predict (broadcast)
        Y2 = X @ Ct
        Z2 = torch.cat((X[:, 0].unsqueeze(dim=1), Y2), dim=1)

        assert (torch.sum(torch.isnan(Y2)) == 0)

        return Z2.reshape(Z.shape), Ct

    def loss(self, X_dec, X_dec2, Z, Z2, Ct):

        # predict ambient
        E1 = self.loss_func(X_dec, X_dec2)

        # predict latent
        E2 = self.loss_func(Z, Z2)

        # Koopman operator constraints (disentanglement)
        D = torch.linalg.eigvals(Ct)

        Dn = torch.real(torch.conj(D) * D)
        Dr = torch.real(D)
        Db = torch.sqrt((Dr - torch.ones(len(Dr)).to(Dr.device)) ** 2 + torch.imag(D) ** 2)

        # ----- static loss ----- #
        Id, new_static_number = None, None
        if self.mode == 'norm':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dns = torch.index_select(Dn, 0, Is)
            E3_static = self.loss_func(Dns, torch.ones(len(Dns)).to(Dns.device))

        elif self.mode == 'real':
            I = torch.argsort(Dr)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Drs = torch.index_select(Dr, 0, Is)
            E3_static = self.loss_func(Drs, torch.ones(len(Drs)).to(Drs.device))

        elif self.mode == 'ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            E3_static = self.loss_func(Dbs, torch.zeros(len(Dbs)).to(Dbs.device))

        elif self.mode == 'space_ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            # E3_static = torch.mean(self.sp_b_thresh(Dbs))

        elif self.mode == 'none':
            E3_static = torch.zeros(1).to(self.args.device)

        # report unique number
        if self.run:
            self.run['general/static_eigen_vals_number'].log(new_static_number)

        if self.dynamic_loss_mode == 'strict':
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = self.loss_func(Dnd, self.eigs_tresh * torch.ones(len(Dnd)).to(Dnd.device))

        elif self.dynamic_loss_mode == 'thresh' and self.mode == 'none':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_loss_mode == 'thresh':
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_loss_mode == 'ball':
            Dbd = torch.index_select(Db, 0, Id)
            E3_dynamic = torch.mean(
                (Dbd < self.args.ball_thresh).float() * ((torch.ones(len(Dbd))).to(Dbd.device) * 2 - Dbd))

        elif self.dynamic_loss_mode == 'real':
            Drd = torch.index_select(Dr, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Drd))

        if self.dynamic_loss_mode == 'none':
            E3 = E3_static
        else:
            E3 = E3_static + E3_dynamic

        return E1, E2, E3
