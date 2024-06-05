import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.utils.data
import torch.nn.init
import numpy as np

import argparse
from tqdm import tqdm

from model import KoopmanCNN
import torch.optim as optim

from utils import load_dataset
from torch.utils.data import DataLoader


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
    parser.add_argument('--arch', type=str, default='KoopmanCNN', choices=['KoopmanCNN'])
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
    parser.add_argument('--eigs_thresh', type=float, default=.5)  # related to 'norm' static mode loss

    # other
    parser.add_argument('--noise', type=str, default='none', help='adding blur to the sample (in the pixel space')

    parser.add_argument('--train_classifier', type=bool, default=False)
    parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')
    parser.add_argument('--type_gt', type=str, default='action')

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


def create_model(args):
    return KoopmanCNN(args)


def create_checkpoint_name(args):
    checkpoint_name = f'./weights/sprites/' \
                      f'nepochs={args.epochs}' + \
                      f'rnn={args.rnn}' + \
                      f'rnn_bi={args.lstm_dec_bi}' + \
                      f'bsz={args.batch_size}' + \
                      f'_conv={args.conv_dim}' \
                      f'_lr={args.lr}' \
                      f'_wd={args.weight_decay}' \
                      f'_dropout={args.dropout}' + \
                      f'_wrec={args.w_rec}' \
                      f'_wpred={args.w_pred}' \
                      f'_weigs={args.w_eigs}' + \
                      f'_kdim={args.k_dim}' \
                      f'_hdim={args.hidden_dim}' \
                      f'_static={args.static_size}' \
                      f'_s_mode={args.static_mode}' + \
                      f'_d_mode={args.dynamic_mode}' + \
                      f'_eig_th={args.eigs_thresh}' \
                      f'_b_th={args.ball_thresh}' \
                      f'_d_th={args.dynamic_thresh}' \
                      f'_noise={args.noise}' \
                      f'.model'

    return checkpoint_name


def save_checkpoint(epoch, checkpoints):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': epoch_losses_test},
        checkpoints)


def load_checkpoint(model, optimizer, checkpoint_name):
    try:
        print("Loading Checkpoint from '{}'".format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_losses_test = checkpoint['losses']
        print("Resuming Training From Epoch {}".format(start_epoch))
        return start_epoch, epoch_losses_test
    except:
        print("No Checkpoint Exists At '{}'.Start Fresh Training".format(checkpoint_name))
        return 0, []


def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def agg_losses(LOSSES, losses):
    if not LOSSES:
        LOSSES = [[] for _ in range(len(losses))]
    for jj, loss in enumerate(losses):
        LOSSES[jj].append(loss.item())
    return LOSSES


def log_losses(epoch, losses_tr, losses_te, names):
    losses_avg_tr, losses_avg_te = [], []

    for loss in losses_tr:
        losses_avg_tr.append(np.mean(loss))

    for loss in losses_te:
        losses_avg_te.append(np.mean(loss))

    loss_str_tr = 'Epoch {}, TRAIN: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_tr):
        loss_str_tr += '{}={:.3e}, \t'.format(names[jj], loss)
    print(loss_str_tr)

    loss_str_te = 'Epoch {}, TEST: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_te):
        loss_str_te += '{}={:.3e}, \t'.format(names[jj], loss)
    print(loss_str_te)

    return losses_avg_tr[0], losses_avg_te[0]


def train(args):

    args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.checkpoint_path = checkpoint_name

    for epoch in range(start_epoch, args.epochs):
        print("Running Epoch : {}".format(epoch + 1))

        model.train()
        losses_agg_tr, losses_agg_te = [], []
        for i, data in tqdm(enumerate(train_loader, 1)):
            X = reorder(data['images']).to(args.device)

            optimizer.zero_grad()
            outputs = model(X)

            losses = model.loss(X, outputs)
            losses[0].backward()
            optimizer.step()

            losses_agg_tr = agg_losses(losses_agg_tr, losses)

        model.eval()
        with torch.no_grad():
            print('Evaulating the model')
            for i, data in tqdm(enumerate(test_loader, 1)):
                X = reorder(data['images']).to(args.device)

                outputs = model(X)
                losses = model.loss(X, outputs)

                losses_agg_te = agg_losses(losses_agg_te, losses)

        # log losses
        loss_avg_tr, loss_avg_te = log_losses(epoch, losses_agg_tr, losses_agg_te, model.names)
        epoch_losses_test.append(loss_avg_te)

        # save model checkpoint
        save_checkpoint(epoch, checkpoint_name)

    print("Training is complete")


if __name__ == '__main__':
    # hyperparameters
    parser = define_args()
    args = parser.parse_args()

    # data parameters
    args.n_frames = 8
    args.n_channels = 3
    args.n_height = 64
    args.n_width = 64

    print(args)

    # set PRNG seed
    args.device = set_seed_device(args.seed)

    # load data
    train_data, test_data = load_dataset(args)
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=args.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=args.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    # create model
    model = create_model(args).to(device=args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # create model checkpoint name
    checkpoint_name = create_checkpoint_name(args)

    # load the model
    start_epoch, epoch_losses_test = load_checkpoint(model, optimizer, checkpoint_name)

    print("number of model parameters: {}".format(sum(param.numel() for param in model.parameters())))
    train(args)
