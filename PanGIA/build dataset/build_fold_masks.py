# build_fold_masks_csv.py

import os
import numpy as np
import pandas as pd

# -------------------------
# Paths
# -------------------------
FOLD_CSV_DIR = "./fold_csv"      # 拆分后的 K-fold 正例 CSV
RN_CSV_DIR   = "./rn_csv"        # 每折可靠负样本 CSV（用于训练）
MASK_CSV_DIR = "./mask_csv"      # 输出的 mask CSV 路径
os.makedirs(MASK_CSV_DIR, exist_ok=True)

# -------------------------
# Load adjacency matrix
# -------------------------
adj_df = pd.read_csv("./data/RDA_Matrix.csv", index_col=0)
num_p, num_d = adj_df.shape

# 预先构建所有负例对的索引列表
all_neg_idx = np.argwhere(adj_df.values == 0)  # shape: [N_neg, 2]

# -------------------------
# Build train/test masks for each fold
# -------------------------
for fold in range(5):
    print(f"🌀 Processing fold {fold}")

    # 读取正例训练/测试索引
    pos_tr = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_train.csv").values  # [[i,j],...]
    pos_te = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_test.csv").values

    # 读取本折的“可靠负样本”索引（用于训练）
    rn_tr  = pd.read_csv(f"{RN_CSV_DIR}/fold_{fold}_rn.csv").values

    # ---------- 构建 train_mask ----------
    train_mask = np.zeros((num_p, num_d), dtype=int)
    # 标记正例
    train_mask[tuple(pos_tr.T)] = 1
    # 标记可靠负例
    train_mask[tuple(rn_tr.T)]  = 1

    # ---------- 构建 test_mask ----------
    test_mask = np.zeros((num_p, num_d), dtype=int)
    # 先标记正例
    test_mask[tuple(pos_te.T)] = 1

    # 要排除出采样池的负例：所有训练中的对（正例+可靠负）
    used_train = { (i,j) for i,j in np.vstack([pos_tr, rn_tr]) }

    # 从 all_neg_idx 中剔除训练用的、也不采测试正例对应位置（pos_te 全是正例，all_neg_idx 本身都是负例，此处仅剔除 used_train）
    neg_candidates = np.array([
        idx for idx in all_neg_idx
        if (idx[0], idx[1]) not in used_train
    ])

    # 采样数量：这里默认和正例数相同，也可以按其他比例
    n_pos_te = len(pos_te)
    rng = np.random.default_rng(seed=fold)  # 固定 seed 保证可复现
    sel = rng.choice(len(neg_candidates), size=n_pos_te, replace=False)
    neg_te = neg_candidates[sel]

    # 将采样到的负例标记到 test_mask
    test_mask[tuple(neg_te.T)] = 1

    # ---------- 保存 CSV ----------
    pd.DataFrame(train_mask,
                 index=adj_df.index,
                 columns=adj_df.columns)\
      .to_csv(f"{MASK_CSV_DIR}/fold_{fold}_train_mask.csv")
    pd.DataFrame(test_mask,
                 index=adj_df.index,
                 columns=adj_df.columns)\
      .to_csv(f"{MASK_CSV_DIR}/fold_{fold}_test_mask.csv")

    print(f"✅ Saved: fold_{fold}_train_mask.csv, fold_{fold}_test_mask.csv")

print(f"\n✅ All mask CSVs saved to {MASK_CSV_DIR}")