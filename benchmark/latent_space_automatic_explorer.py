from typing import List

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os, sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from cdsvae.model import classifier_Sprite_all
from koopman_utils import get_sorted_indices, static_dynamic_split
from test_trained_model import check_cls_specific_indexes
from utils import load_checkpoint, reorder, t_to_np, load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def define_args():
    parser = argparse.ArgumentParser(description="Sprites SKD")

    # general
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    # data
    parser.add_argument("--dataset_path", default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML/')
    parser.add_argument("--dataset", default='Sprites')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N')

    # model
    parser.add_argument('--arch', type=str, default='KoopmanCNN')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--rnn', type=str, default='both',
                        help='encoder decoder LSTM strengths. Can be: "none", "encoder","decoder", "both"')
    parser.add_argument('--k_dim', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=40,
                        help='the hidden dimension of the output decoder lstm')
    parser.add_argument('--lstm_dec_bi', type=bool, default=False)  # nimrod added

    # loss params
    parser.add_argument('--w_rec', type=float, default=15.0)
    parser.add_argument('--w_pred', type=float, default=1.0)
    parser.add_argument('--w_eigs', type=float, default=1.0)

    # eigen values system params
    parser.add_argument('--static_size', type=int, default=8)
    parser.add_argument('--static_mode', type=str, default='norm', choices=['norm', 'real', 'ball'])
    parser.add_argument('--dynamic_mode', type=str, default='ball',
                        choices=['strict', 'thresh', 'ball', 'real', 'none'])

    # thresholds
    parser.add_argument('--ball_thresh', type=float, default=0.6)  # related to 'ball' dynamic mode
    parser.add_argument('--dynamic_thresh', type=float, default=0.5)  # related to 'thresh', 'real'
    parser.add_argument('--eigs_thresh', type=float, default=.5)  # related to 'norm' static mode loss

    # other
    parser.add_argument('--noise', type=str, default='none', help='adding blur to the sample (in the pixel space')

    # parameters for the classifier
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')

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


def explore_latent_space_and_return_mapping(latent_codes: List[np.ndarray], attributes: List[List[int]],
                                            classifier_type: str = 'linear') -> List[List[int]]:
    """
    This function takes a list of latent codes, and for each label in the attribute list, it trains a classifier
    to predict the label from the latent code. For each latent code coordinate, it collects its relative importance
    score for each label. Then it assigns each coordinate to the label with the highest importance score.

    Args:
    - latent_codes: A list of tensors, each tensor is a latent code.
    - attributes: A list of lists, where each list contains the attributes of the corresponding latent code.
    - classifier_type: The type of classifier to use, either 'linear' (Logistic Regression) or 'decision_tree'.

    Returns:
    - mapping: A list of lists where each sublist corresponds to the most important label for each latent code coordinate.
    """

    # Convert latent codes to a numpy array
    latent_codes_np = np.array(latent_codes)

    # Convert attributes to numpy array
    attributes_np = np.array(attributes)

    num_latent_dims = latent_codes_np.shape[1]
    num_labels = attributes_np.shape[1]

    importance_scores = np.zeros((num_latent_dims, num_labels))

    for label_idx in range(num_labels):
        # Select the classifier
        if classifier_type == 'linear':
            classifier = LogisticRegression()
        elif classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(max_depth=8)
        else:
            raise ValueError("Unsupported classifier_type. Choose either 'linear' or 'decision_tree'.")

        # Train the classifier on the latent codes to predict the current label
        classifier.fit(latent_codes_np, attributes_np[:, label_idx])
        # print the accuracy
        print(f"Accuracy for label {label_idx}: {classifier.score(latent_codes_np, attributes_np[:, label_idx])}")

        # Collect the importance scores
        if classifier_type == 'linear':
            # For linear classifier, importance is the absolute value of the coefficients
            normalized_coefs = (np.abs(classifier.coef_) / np.sum(np.abs(classifier.coef_))).sum(axis=0)
            importance_scores[:, label_idx] = normalized_coefs
        elif classifier_type == 'decision_tree':
            normalized_coefs = classifier.feature_importances_ / np.sum(classifier.feature_importances_)
            # For decision tree, importance is the feature_importances_ attribute
            importance_scores[:, label_idx] = normalized_coefs

    # Assign each coordinate to the label with the highest importance score
    mapping = np.argmax(importance_scores, axis=1)

    # if all importance scores are zero, assign the coordinate -1
    mapping[np.all(importance_scores == 0, axis=1)] = -1

    # Convert the mapping to a list of lists
    return mapping.tolist()


if __name__ == "__main__":
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
    train_data, test_data = load_dataset(args)
    # reduce the size of test data into k samples
    k = 2000
    idx = np.random.choice(2664, k, replace=False)
    test_data.data = test_data.data[idx]
    test_data.A_label = test_data.A_label[idx]
    test_data.D_label = test_data.D_label[idx]
    test_data.N = k
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=1024,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    x, label_A, label_D = reorder(torch.tensor(test_data.data)), test_data.A_label[:, 0], test_data.D_label[:, 0]

    # # todo - batch the latent codes to produce all of them instead of taking k samples
    # take a random subset of k samples
    # k = 2664
    # idx = np.random.choice(x.shape[0], k, replace=False)
    # x = x[idx]
    # label_A = label_A[idx]
    # label_D = label_D[idx]

    x = x.to(args.device)

    # todo - iteratively pass the data in batches and save the outputs
    outputs = model(x)
    _, Ct_te, Z = outputs[0], outputs[-1], outputs[2]

    #  --- extract latent codes ---
    Z = t_to_np(Z.reshape(x.shape[0], 8, 40))  # 8 its the time steps in this dataset and 40 is the latent dimension
    C = t_to_np(Ct_te)

    # eig
    D, V = np.linalg.eig(C)
    U = np.linalg.inv(V)

    # project onto V
    ZL = np.real((Z @ V).mean(axis=1))  # create a single latent code scaler for each sample

    # transfer label_a and label_d from one hot encoding to a single label

    # --- get the static indices --- #
    # labels = np.argmax(label_A, axis=-1)
    I = get_sorted_indices(D, 'ball')
    # Id, Is = static_dynamic_split(D, I, 'ball', 8)
    # ZL = ZL[:, Is]

    # transfer label_a and label_d from one hot encoding to a single label
    label_A = np.argmax(label_A, axis=-1)
    label_D = np.argmax(label_D, axis=1)
    # append label_d to label_a
    labels = np.column_stack((label_A, label_D))

    # classifier is 'linear' / 'decision_tree'
    static_mappings = explore_latent_space_and_return_mapping(latent_codes=list(ZL),
                                                              attributes=[list(a) for a in list(labels)],
                                                              classifier_type='decision_tree')

    # a dictionary between an indices and its associated label
    map_idx_to_label = {}
    # set the dynamic
    # for idx in I:
    #     map_idx_to_label[idx] = 4
    # set the static
    for idx, label in enumerate(static_mappings):
        if label != -1:
            map_idx_to_label[idx] = label

    map_label_to_idx = {}
    for idx, label in map_idx_to_label.items():
        if label not in map_label_to_idx:
            map_label_to_idx[label] = []
        map_label_to_idx[label].append(idx)

    # ----- load classifier ----- #
    classifier = classifier_Sprite_all(args)
    args.resume = '../cdsvae/sprite_judge.tar'
    loaded_dict = torch.load(args.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()
    print('loader classifier from: ', args.resume)

    print('--- start multifactor classification --- \n')
    label_to_name_dict = {
        0: 'skin',
        1: 'pant',
        2: 'top',
        3: 'hair',
        4: 'action'
    }
    df = pd.DataFrame(columns=['action', 'skin', 'pant', 'top', 'hair'],
                      index=['action', 'skin', 'pant', 'top', 'hair'])
    for label in map_label_to_idx:
        print(f"Label {label_to_name_dict[label]}: {map_label_to_idx[label]}")
        action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc = check_cls_specific_indexes(model, classifier, test_loader,
                                                                                       None, 'action',
                                                                                       map_label_to_idx[label],
                                                                                       fix=True)

        df.loc[label_to_name_dict[label]] = [action_Acc, skin_Acc, pant_Acc, top_Acc, hair_Acc]
        # print df
    print(df)
