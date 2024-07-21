import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch.utils.data
import torch.nn.init
import numpy as np
import argparse
import torch.nn.functional as F

from model import KoopmanCNN
from utils import load_dataset, load_checkpoint
from torch.utils.data import DataLoader
from cdsvae.model import classifier_Sprite_all


def define_args():
    parser = argparse.ArgumentParser(description="Sprites SKD")

    # general
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    # data
    parser.add_argument("--dataset_path", default='./data')
    parser.add_argument("--dataset", default='Sprites')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N')

    # model
    parser.add_argument('--arch', type=str, default='KoopmanCNN', choices=['KoopmanCNN'])
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn', type=str, default='both',
                        help='encoder decoder LSTM strengths. Can be: "none", "encoder","decoder", "both"')
    parser.add_argument('--k_dim', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=40, help='the hidden dimension of the output decoder lstm')
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

    # parameters for the classifier
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')

    return parser


def check_cls(cdsvae, classifier, test_loader, run, run_type):
    for epoch in range(1):
        print("Epoch", epoch)
        cdsvae.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            if run_type == "action":
                recon_x_sample, recon_x = cdsvae.forward_sample_for_classification2(x, fix_motion=True)
            else:
                recon_x_sample, recon_x = cdsvae.forward_sample_for_classification2(x, fix_motion=False)

            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

                pred1 = F.softmax(pred_action1, dim=1)
                pred2 = F.softmax(pred_action2, dim=1)
                pred3 = F.softmax(pred_action3, dim=1)

            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(np.argmax(label_D.detach().cpu().numpy(), axis=1))

            def count_D(pred, label, mode=1):
                return (pred // mode) == (label // mode)

            # action
            acc0_sample = (np.argmax(pred_action2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_D.cpu().numpy(), axis=1)).mean()
            # skin
            acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 0].cpu().numpy(), axis=1)).mean()
            # pant
            acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 1].cpu().numpy(), axis=1)).mean()
            # top
            acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 2].cpu().numpy(), axis=1)).mean()
            # hair
            acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 3].cpu().numpy(), axis=1)).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample

        print(
            'Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
                mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)

        acc = (label_gt == label2_all).mean()
        # kl = KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
        index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        # IS = inception_score(pred2_selected)
        # H_yx = entropy_Hyx(pred2_selected)
        # H_y = entropy_Hy(pred2_selected)

        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, 0, 0, 0, 0))


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


def test():
    losses_agg_te = []
    model.eval()
    with torch.no_grad():
        print('Evaulating the model disentanglement')
        check_cls(model, classifier, test_loader, None, 'action')
        check_cls(model, classifier, test_loader, None, 'aaction')


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

    # --- load model ---
    checkpoint_name = 'weights/sprites_weights.model'
    load_checkpoint(model, checkpoint_name)
    model.eval()

    # ----- load classifier ----- #
    classifier = classifier_Sprite_all(args)
    args.resume = 'cdsvae/sprite_judge.tar'
    loaded_dict = torch.load(args.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    print('loader classifier from: ', args.resume)
    print("number of model parameters: {}".format(sum(param.numel() for param in model.parameters())))
    test()
