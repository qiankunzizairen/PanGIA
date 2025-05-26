import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pandas as pd
from sklearn import metrics
from models import LP
from utils import set_seed

# === argparse 设置 ===
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay.')
parser.add_argument('--hidden', type=int, default=64, help='Hidden representation dimension')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight between miRNA and disease')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)

# === 数据读取 ===
rda_matrix = pd.read_csv("./data/RDA_Matrix.csv", index_col=0)
mirna_sim = pd.read_csv("./data/miR_needleman.csv", index_col=0)
disease_sim = pd.read_csv("./data/disease_do.csv", index_col=0)

# 对齐索引
rda_matrix = rda_matrix.loc[mirna_sim.index, disease_sim.index]

mdi = torch.tensor(rda_matrix.values, dtype=torch.float32)
gm = torch.tensor(mirna_sim.values, dtype=torch.float32)
gd = torch.tensor(disease_sim.values, dtype=torch.float32)

if args.cuda:
    mdi = mdi.cuda()
    gm = gm.cuda()
    gd = gd.cuda()

# === 模型定义 ===
class GNNp(nn.Module):
    def __init__(self):
        super(GNNp, self).__init__()
        self.gnnpl = LP(args.hidden, mdi.shape[1])
        self.gnnpd = LP(args.hidden, mdi.shape[0])

    def forward(self, y0):
        yl, zl = self.gnnpl(gm, y0)
        yd, zd = self.gnnpd(gd, y0.t())
        return yl, zl, yd, zd

# === 指标计算 ===
def compute_auc_aupr_ri(true, pred):
    labels = true.cpu().numpy().flatten()
    scores = pred.cpu().detach().numpy().flatten()
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

# === 训练函数 ===
def train(gnnp, y0, epoch, alpha, target):
    optp = torch.optim.Adam(gnnp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(epoch):
        gnnp.train()
        yl, zl, yd, zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl, y0)
        losspd = F.binary_cross_entropy(yd, y0.t())
        value = alpha * yl + (1 - alpha) * yd.t()
        att = torch.softmax(torch.mm(zl, zd.t()) / math.sqrt(args.hidden), dim=-1) * value
        loss = (alpha * losspl + (1 - alpha) * losspd) + F.mse_loss(att, y0)

        optp.zero_grad()
        loss.backward()
        optp.step()

        gnnp.eval()
        with torch.no_grad():
            yl, zl, yd, zd = gnnp(y0)
            pred = alpha * yl + (1 - alpha) * yd.t()
            auc, aupr, ri = compute_auc_aupr_ri(target, pred)
            print(f"Epoch {e + 1} | Loss: {loss.item():.4f} | AUROC: {auc:.4f} | AUPR: {aupr:.4f} | RI: {ri:.4f}")

    return alpha * yl + (1 - alpha) * yd.t()

# === 训练入口 ===
def trainres(A0, target):
    gnnp = GNNp()
    if args.cuda:
        gnnp = gnnp.cuda()
    result = train(gnnp, A0, args.epochs, args.alpha, target)
    return result

# === Five-Fold 交叉验证 ===
def fivefoldcv(A, alpha=0.5):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    for i in range(5):
        print(f"\nFold {i + 1}")
        A0 = A.clone()
        for j in range(i * N // 5, (i + 1) * N // 5):
            A0[idx[j], :] = 0
        _ = trainres(A0, A)

# === 主流程 ===
print("Training with CSV data | 5-fold Cross Validation")
fivefoldcv(mdi, alpha=args.alpha)