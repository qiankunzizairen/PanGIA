import argparse
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from utils import get_data, data_processing, make_adj
from train import train
import torch as th


def compute_auc_aupr_ri(true, pred):
    labels = true.cpu().numpy()
    scores = pred.cpu().numpy()
    combined = list(zip(labels, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    labels_sorted, _ = zip(*combined)
    indices = np.arange(1, len(labels) + 1)[np.array(labels_sorted) == 1]
    n_test = len(labels)
    n_test_p = sum(labels == 1)
    rank_idx = indices.sum() / n_test / n_test_p if n_test_p > 0 else 0.0
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    precisions, recalls, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recalls, precisions)
    return round(auc, 6), round(aupr, 6), round(rank_idx, 6)


def result(args):
    all_data = get_data(args)
    args.miRNA_number = all_data['miRNA_number']
    args.disease_number = all_data['disease_number']

    md_matrix = make_adj(all_data['md'], (args.miRNA_number, args.disease_number)).numpy()
    one_index, zero_index = [], []

    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])

    one_index = np.array(one_index)
    zero_index = np.array(zero_index)
    np.random.seed(args.random_seed)
    np.random.shuffle(zero_index)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(one_index)):
        print(f"\n====== Fold {fold + 1}/5 ======")

        train_pos = one_index[train_idx]
        test_pos = one_index[test_idx]
        test_neg = zero_index[:len(test_pos)]
        train_neg = zero_index[len(test_pos):len(test_pos) + len(train_pos)]

        train_all = np.concatenate([train_pos, train_neg])
        train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        samples = np.concatenate([train_all, train_labels.reshape(-1, 1)], axis=1)

        data = dict(all_data)
        data['train_samples'] = samples
        data['train_md'] = train_pos

        data_processing(data, args)

        def per_epoch_eval(pred, label, epoch):
            auc, aupr, ri = compute_auc_aupr_ri(label, pred)
            print(f"Epoch {epoch}: AUC={auc}, AUPR={aupr}, RI={ri}")

        _ = train(data, args, per_epoch_eval=per_epoch_eval)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--fm', type=int, default=64, help='length of miRNA feature')
parser.add_argument('--fd', type=int, default=64, help='length of dataset feature')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--in_feats", type=int, default=64, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=64, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--gcn_bias", type=bool, default=True, help='gcn bias')
parser.add_argument("--gcn_batchnorm", type=bool, default=True, help='gcn batchnorm')
parser.add_argument("--gcn_activation", default='relu', help='gcn activation')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 1], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
parser.add_argument('--dataset', default='HMDD v2.0', help='dataset')
parser.add_argument('--save_score', default='True', help='save_score')
parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')

args = parser.parse_args()
args.dd2 = True
args.data_dir = 'data/' + args.dataset + '/'
args.result_dir = 'result/' + args.dataset + '/'
args.save_score = True if str(args.save_score) == 'True' else False

result(args)