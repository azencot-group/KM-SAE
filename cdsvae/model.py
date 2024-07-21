import numpy as np
import torch
import torch.nn as nn

# ---------------- encoder -----------------------
from cdsvae.utils import reparameterize


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


# ---------------- decoder -----------------------
"""
# Using transpose conv as the block to up-sample
"""


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class decoder_convT(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_convT, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


# ---------------- model -----------------------
class CDSVAE(nn.Module):
    def __init__(self, opt):
        super(CDSVAE, self).__init__()
        self.s_dim = opt.s_dim  # content
        self.d_dim = opt.d_dim  # motion
        self.g_dim = opt.g_dim  # frame/image feature
        self.channels = opt.channels  # image channel
        self.hidden_dim = opt.rnn_size
        self.s_rnn_layers = opt.s_rnn_layers
        self.frames = opt.frames

        self.encoder = encoder(self.g_dim, self.channels)
        self.decoder = decoder_convT(self.d_dim + self.s_dim, self.channels)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.d_prior_lstm_ly1 = nn.LSTMCell(self.d_dim, self.hidden_dim)
        self.d_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.d_prior_mean = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_prior_logvar = nn.Linear(self.hidden_dim, self.d_dim)

        # ----- Posterior of content and motion
        # content and motion features share one bi-lstm
        self.d_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.s_mean = nn.Linear(self.hidden_dim * 2, self.s_dim)
        self.s_logvar = nn.Linear(self.hidden_dim * 2, self.s_dim)

        # motion features from the next RNN
        self.d_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.d_mean = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_logvar = nn.Linear(self.hidden_dim, self.d_dim)

    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.d_lstm(conv_x)
        # get f:
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        s_mean = self.s_mean(lstm_out_f)
        s_logvar = self.s_logvar(lstm_out_f)
        s_post = reparameterize(s_mean, s_logvar, random_sampling=True)

        # pass to one direction rnn
        features, _ = self.d_rnn(lstm_out)
        d_mean = self.d_mean(features)
        d_logvar = self.d_logvar(features)
        d_post = reparameterize(d_mean, d_logvar, random_sampling=True)

        if isinstance(x, list):
            s_mean_list = [s_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.d_lstm(conv_x)
                # get f:
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                s_mean = self.s_mean(lstm_out_f)
                s_mean_list.append(s_mean)
            s_mean = s_mean_list
        # s_mean is list if triple else not
        return s_mean, s_logvar, s_post, d_mean, d_logvar, d_post

    # ------ sample z from learned LSTM prior base on previous postior, teacher forcing for training  ------
    def sample_motion_prior_train(self, d_post, random_sampling=True):
        d_out = None
        d_means = None
        d_logvars = None
        batch_size = d_post.shape[0]

        d_t = torch.zeros(batch_size, self.d_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.d_prior_lstm_ly1(d_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.d_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            d_mean_t = self.d_prior_mean(h_t_ly2)
            d_logvar_t = self.d_prior_logvar(h_t_ly2)
            d_prior = reparameterize(d_mean_t, d_logvar_t, random_sampling)
            if d_out is None:
                d_out = d_prior.unsqueeze(1)
                d_means = d_mean_t.unsqueeze(1)
                d_logvars = d_logvar_t.unsqueeze(1)
            else:
                d_out = torch.cat((d_out, d_prior.unsqueeze(1)), dim=1)
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1)
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1)
            d_t = d_post[:, i, :]
        return d_means, d_logvars, d_out

    # ------ sample z purely from learned LSTM prior with arbitrary frames------
    def sample_motion_prior(self, n_sample, n_frame, random_sampling=True):
        d_out = None  # This will ultimately store all d_s in the format [batch_size, frames, d_dim]
        d_means = None
        d_logvars = None
        batch_size = n_sample

        d_t = torch.zeros(batch_size, self.d_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.d_prior_lstm_ly1(d_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.d_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            d_mean_t = self.d_prior_mean(h_t_ly2)
            d_logvar_t = self.d_prior_logvar(h_t_ly2)
            d_t = reparameterize(d_mean_t, d_logvar_t, random_sampling)
            if d_out is None:
                # If d_out is none it means d_t is d_1, hence store it in the format [batch_size, 1, d_dim]
                d_out = d_t.unsqueeze(1)
                d_means = d_mean_t.unsqueeze(1)
                d_logvars = d_logvar_t.unsqueeze(1)
            else:
                # If d_out is not none, d_t is not the initial z and hence append it to the previous d_ts collected in d_out
                d_out = torch.cat((d_out, d_t.unsqueeze(1)), dim=1)
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1)
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1)
        return d_means, d_logvars, d_out

    def forward(self, x):
        s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = self.encode_and_sample_post(x)
        d_mean_prior, d_logvar_prior, d_prior = self.sample_motion_prior_train(d_post, random_sampling=True)

        s_expand = s_post.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_post, s_expand), dim=2)
        recon_x = self.decoder(zf)
        return s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post, d_mean_prior, d_logvar_prior, d_prior, \
               recon_x

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    # fixed content and sample motion for classification disagreement scores
    def forward_fixed_content_for_classification(self, x):
        s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = self.encode_and_sample_post(x)
        # d_mean_prior, d_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        d_mean_prior, d_logvar_prior, d_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        s_expand = s_mean.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_mean_prior, s_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        zf = torch.cat((d_mean_post, s_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification(self, x):
        # d_mean_prior, d_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = self.encode_and_sample_post(x)

        # s_expand = s_mean.unsqueeze(1).expand(-1, self.frames, self.s_dim)

        s_prior = reparameterize(torch.zeros(s_mean.shape).cuda(), torch.zeros(s_logvar.shape).cuda(),
                                 random_sampling=True)
        # s_prior = reparameterize(s_mean, torch.zeros(s_logvar.shape).cuda(), random_sampling=True)
        s_expand = s_prior.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_mean_post, s_expand), dim=2)
        # zf = torch.cat((d_post, s_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        s_expand = s_post.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_post, s_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_fixed_content_for_classification_tr(self, x):
        s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = self.encode_and_sample_post(x)
        # d_mean_prior, d_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        d_mean_prior, d_logvar_prior, d_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        s_expand = s_mean.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_mean_prior, s_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification_tr(self, x):
        # d_mean_prior, d_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = self.encode_and_sample_post(x)

        # s_expand = s_mean.unsqueeze(1).expand(-1, self.frames, self.s_dim)

        s_prior = reparameterize(torch.zeros(s_mean.shape).cuda(), torch.zeros(s_logvar.shape).cuda(),
                                 random_sampling=True)
        # s_prior = reparameterize(s_mean, torch.zeros(s_logvar.shape).cuda(), random_sampling=True)
        s_expand = s_prior.unsqueeze(1).expand(-1, self.frames, self.s_dim)
        zf = torch.cat((d_mean_post, s_expand), dim=2)
        # zf = torch.cat((d_post, s_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    def forward_exchange(self, x):
        s_mean, s_logvar, s, d_mean_post, d_logvar_post, d = self.encode_and_sample_post(x)

        a = s[np.arange(0, s.shape[0], 2)]
        b = s[np.arange(1, s.shape[0], 2)]
        s_mix = torch.stack((b, a), dim=1).view((-1, s.shape[1]))

        s_expand = s_mix.unsqueeze(1).expand(-1, self.frames, self.s_dim)

        zf = torch.cat((d, s_expand), dim=2)
        recon_x = self.decoder(zf)
        return s_mean, s_logvar, s, None, None, d, None, None, recon_x

    def swap(self, x):
        s_mean, s_logvar, s, d_mean_post, d_logvar_post, d = self.encode_and_sample_post(x)
        s1, d1 = s_mean[0][None, :].expand(self.frames, s.shape[-1]), d_mean_post[0]
        s2, d2 = s_mean[1][None, :].expand(self.frames, s.shape[-1]), d_mean_post[1]

        s1d2 = torch.cat((d2, s1), dim=1)
        s2d1 = torch.cat((d1, s2), dim=1)
        sd = torch.stack([s1d2, s2d1])
        recon_x = self.decoder(sd)

        return recon_x

# ---------------- classifier -----------------------
class classifier_Sprite_all(nn.Module):
    def __init__(self, opt):
        super(classifier_Sprite_all, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_top = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_pant = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_hair = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_action = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 9))

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_action(lstm_out_f), self.cls_skin(lstm_out_f), self.cls_pant(lstm_out_f), \
               self.cls_top(lstm_out_f), self.cls_hair(lstm_out_f)
