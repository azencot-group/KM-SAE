import math

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


# -------------------------- computational utils -------------------------- #


def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu) ** 2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def compute_mi(latent_sample, latent_dist):
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pd_qd_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         None,
                                                                         is_mss=False)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()

    return mi_loss


def _get_log_pd_qd_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    # print("latent_sample:", latent_sample.shape)
    # print("latent_dist:", len(latent_dist), latent_dist[0].shape, latent_dist[1].shape)
    # print("is_mss:", is_mss)

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    # zeros = torch.zeros_like(latent_sample)
    # log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    # log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    # return log_pz, log_qz, log_prod_qzi, log_q_zCx
    return None, log_qz, None, log_q_zCx


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')


def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def calculate_mws(batch_size, d_post, d_post_logvar, d_post_mean, mi_fz, n_frame, opt, s, s_logvar, s_mean, d_dim):
    # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
    # batch_size x batch_size x s_dim
    _logq_s_tmp = log_density(s.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, opt.s_dim),
                              # [8, 128, 1, 256]
                              s_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.s_dim),
                              # [8, 1, 128, 256]
                              s_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size,
                                                                               opt.s_dim))  # [8, 1, 128, 256]
    # n_frame x batch_size x batch_size x s_dim
    _logq_d_tmp = log_density(d_post.transpose(0, 1).view(n_frame, batch_size, 1, d_dim),  # [8, 128, 1, 32]
                              d_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, d_dim),  # [8, 1, 128, 32]
                              d_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, d_dim))  # [8, 1, 128, 32]
    _logq_fd_tmp = torch.cat((_logq_s_tmp, _logq_d_tmp), dim=3)  # [8, 128, 128, 288]
    logq_f = (logsumexp(_logq_s_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    logq_z = (logsumexp(_logq_d_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    logq_fz = (logsumexp(_logq_fd_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    # n_frame x batch_size
    mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()
    return mi_fz


def kl_loss_calc(d_post_logvar, d_post_mean, d_prior_logvar, d_prior_mean, s_logvar, s_mean):
    # ----- calculate KL of f ----- #
    s_mean = s_mean.view((-1, s_mean.shape[-1]))  # [128, 256]
    s_logvar = s_logvar.view((-1, s_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
    # ----- calculate KL of z ----- #
    d_post_var = torch.exp(d_post_logvar)  # [128, 8, 32]
    d_prior_var = torch.exp(d_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(d_prior_logvar - d_post_logvar +
                            ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var) - 1)
    return kld_f, kld_z, s_logvar, s_mean


# -------------------------- model utils -------------------------- #

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def load_checkpoint(model, checkpoint_name):
    print("Loading Checkpoint from '{}'".format(checkpoint_name))
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])


def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


# -------------------------- evaluation utils -------------------------- #


def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h


def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis=1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h


def inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d
