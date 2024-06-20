import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from utils import load_checkpoint, reorder


def define_args():
    parser = argparse.ArgumentParser(description="Sprites SKD")

    # general
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    # data
    parser.add_argument("--dataset_path", default='./dataset/')
    parser.add_argument("--dataset", default='Sprites')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N')

    # model
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn', type=str, default='both',
                        help='encoder decoder LSTM strengths. Can be: "none", "encoder","decoder", "both"')
    parser.add_argument('--k_dim', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=80, help='the hidden dimension of the output decoder lstm')
    parser.add_argument('--lstm_dec_bi', type=bool, default=False)  # nimrod added

    # loss params
    parser.add_argument('--w_rec', type=float, default=15.0)
    parser.add_argument('--w_pred', type=float, default=1.0)
    parser.add_argument('--w_eigs', type=float, default=1.0)

    # eigen values system params
    parser.add_argument('--static_size', type=int, default=7)
    parser.add_argument('--static_mode', type=str, default='ball', choices=['norm', 'real', 'ball'])
    parser.add_argument('--dynamic_mode', type=str, default='real',
                        choices=['strict', 'thresh', 'ball', 'real', 'none'])

    # thresholds
    parser.add_argument('--ball_thresh', type=float, default=0.6)  # related to 'ball' dynamic mode
    parser.add_argument('--dynamic_thresh', type=float, default=0.5)  # related to 'thresh', 'real'
    parser.add_argument('--eigs_thresh', type=float, default=0.5)  # related to 'norm' static mode loss

    # other
    parser.add_argument('--noise', type=str, default='none', help='adding blur to the sample (in the pixel space')

    return parser


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


# hyperparameters
parser = define_args()
args = parser.parse_args()

# data parameters
args.n_frames = 8
args.n_channels = 3
args.n_height = 64
args.n_width = 64

# set PRNG seed
args.device = set_seed_device(args.seed)

# create model
from model import KoopmanCNN

model = KoopmanCNN(args).to(device=args.device)

# load the model
checkpoint_name = '../weights/sprites_weights.model'
load_checkpoint(model, checkpoint_name)
model.eval()

# load data
data = np.load('../dataset/batch1.npy', allow_pickle=True).item()
data2 = np.load('../dataset/batch2.npy', allow_pickle=True).item()
x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
x2 = reorder(data2['images'])

X = x.to(args.device)
X2 = x2.to(args.device)

# first batch
outputs = model(X)
X_dec_te, Ct_te, Z = outputs[0], outputs[-1], outputs[2]

# second batch
outputs2 = model(X2)
X_dec_te2, Ct_te2, Z2 = outputs2[0], outputs2[-1], outputs2[2]

# --------------- Performing 2-factor swap --------------- #
from koopman_utils import swap

""" Given 2 samples in a batch. It Swap their static and dynamic features"""
indices = [0, 1]
swap(model, X_dec_te, Z, Ct_te, indices, args.static_size, plot=True, pick_type=args.static_mode)

# --------------- Performing multi-factor swap --------------- #
""" Sprites have 4 factors of variation in the static subspace(appearance of the character):
    hair, shirt, skin and pants colors. In multi-factor swapping therein, we show how we swap each of the 4^4
    combinations from a target character to a source character"""

from koopman_utils import swap_by_index

indices = (2, 12)

# 1_1 skin
static_indexes = [32, 33]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 1_2 hair (2, 12)
static_indexes = [38, 39, 35]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 1_3 pants (2, 12)
static_indexes = [28]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 1_4 top (2, 12)
static_indexes = [34, 35, 29, 36]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_1 skin hair (2, 12)
static_indexes = [38, 39, 35, 32, 33]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_2 skin pants (2, 12)
static_indexes = [28, 32, 33]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_3 skin and top (2, 12)
static_indexes = [34, 35, 29, 36, 37, 32, 33]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_4 hair and pants (2, 12)
static_indexes = [39, 38, 35, 28]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_5 hair and top (2, 12)
static_indexes = [38, 39, 34, 35, 31, 29]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_6 pants and top (2, 12)
static_indexes = [34, 35, 36, 28]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 3_1 pants hair top (2, 12)
static_indexes = [28, 38, 39, 35, 34]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 3_2 pants hair skin (2, 12)
static_indexes = [32, 33, 38, 39, 28]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 3_3 pants skin top (2, 12)
static_indexes = [29, 34, 36, 37, 28, 33, 32, 30, 24]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# 2_4 skin top hair (2, 12)
static_indexes = [32, 33, 34, 29, 36, 38, 39, 35]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)

# full
static_indexes = [33, 32, 37, 36, 39, 38, 35, 34, 31, 28]
dynamic_indexes = np.delete(np.arange(Ct_te.shape[0]), static_indexes)
swap_by_index(model, X_dec_te2, Z2, Ct_te2, indices, static_indexes, dynamic_indexes, plot=True)
