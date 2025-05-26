# build_fold_masks_csv.py
import os
import numpy as np
import pandas as pd

# -------------------------
# Paths
# -------------------------
FOLD_CSV_DIR = "./fold_csv"      # 拆分后的 K-fold CSVs
RN_CSV_DIR   = "./rn_csv"        # 每折可靠负样本 CSV
MASK_CSV_DIR = "./mask_csv"      # 输出的 mask CSV 路径
os.makedirs(MASK_CSV_DIR, exist_ok=True)

# -------------------------
# Load adjacency matrix
# -------------------------
adj_df = pd.read_csv("./data/RDA_Matrix.csv", index_col=0)
num_p, num_d = adj_df.shape

# -------------------------
# Build train/test masks for each fold
# -------------------------
for fold in range(5):
    print(f"🌀 Processing fold {fold}")

    # 读取索引对
    pos_tr = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_train.csv").values
    pos_te = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_test.csv").values
    rn_ij  = pd.read_csv(f"{RN_CSV_DIR}/fold_{fold}_rn.csv").values

    # ✅ 构建 train_mask：正样本 + 可靠负样本
    train_mask = np.zeros((num_p, num_d), dtype=int)
    train_mask[tuple(pos_tr.T)] = 1
    train_mask[tuple(rn_ij.T)]  = 1

    # ✅ 构建 test_mask：仅正样本（用于排序评估）
    test_mask = np.zeros((num_p, num_d), dtype=int)
    test_mask[tuple(pos_te.T)] = 1

    # ✅ 保存为 CSV，布尔值用 0/1 表示
    pd.DataFrame(train_mask, index=adj_df.index, columns=adj_df.columns) \
      .to_csv(f"{MASK_CSV_DIR}/fold_{fold}_train_mask.csv")
    
    pd.DataFrame(test_mask, index=adj_df.index, columns=adj_df.columns) \
      .to_csv(f"{MASK_CSV_DIR}/fold_{fold}_test_mask.csv")

    print(f"✅ Saved: fold_{fold}_train_mask.csv, fold_{fold}_test_mask.csv")

print(f"\n✅ All mask CSVs saved to {MASK_CSV_DIR}")