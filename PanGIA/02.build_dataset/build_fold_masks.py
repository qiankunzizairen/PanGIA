# build_fold_masks_csv.py

import os
import numpy as np
import pandas as pd

FOLD_CSV_DIR = "./fold_csv"
RN_CSV_DIR   = "./rn_csv"
MASK_CSV_DIR = "./mask_csv"
os.makedirs(MASK_CSV_DIR, exist_ok=True)

rna_map = pd.read_excel("allR_id.xlsx")
dis_map = pd.read_excel("disease_ID.xlsx")
rna_name_to_id = dict(zip(rna_map["RNA"].astype(str), rna_map["ID"].astype(str)))
dis_name_to_id = dict(zip(dis_map["disease"].astype(str), dis_map["ID"].astype(str)))

adj_df = pd.read_csv("./RDA_Matrix.csv", index_col=0)
num_p, num_d = adj_df.shape

row_ids_order = pd.Series(adj_df.index.astype(str)).map(rna_name_to_id).values
col_ids_order = pd.Series(adj_df.columns.astype(str)).map(dis_name_to_id).values

id_to_rowpos = {rid: i for i, rid in enumerate(row_ids_order) if pd.notna(rid)}
id_to_colpos = {did: j for j, did in enumerate(col_ids_order) if pd.notna(did)}

all_neg_idx = np.argwhere(adj_df.values == 0)

def id_pairs_to_pos(pairs_id: np.ndarray):
    if pairs_id.size == 0:
        return np.empty((0, 2), dtype=int)
    r_ids = pairs_id[:, 0].astype(str)
    d_ids = pairs_id[:, 1].astype(str)
    rows, cols = [], []
    for rid, did in zip(r_ids, d_ids):
        i = id_to_rowpos.get(rid, None)
        j = id_to_colpos.get(did, None)
        if i is not None and j is not None:
            rows.append(i); cols.append(j)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=int)
    return np.stack([rows, cols], axis=1)

for fold in range(5):
    print(f"🌀 Processing fold {fold}")

    pos_tr_id = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_train.csv", header=None).values
    pos_te_id = pd.read_csv(f"{FOLD_CSV_DIR}/fold_{fold}_pos_test.csv", header=None).values
    rn_tr_id  = pd.read_csv(f"{RN_CSV_DIR}/fold_{fold}_rn.csv", header=0).values if os.path.exists(f"{RN_CSV_DIR}/fold_{fold}_rn.csv") else np.empty((0,2))

    pos_tr = id_pairs_to_pos(pos_tr_id)
    pos_te = id_pairs_to_pos(pos_te_id)
    rn_tr  = id_pairs_to_pos(rn_tr_id)

    train_mask = np.zeros((num_p, num_d), dtype=int)
    if pos_tr.size:
        train_mask[tuple(pos_tr.T)] = 1
    if rn_tr.size:
        train_mask[tuple(rn_tr.T)] = 1

    test_mask = np.zeros((num_p, num_d), dtype=int)
    if pos_te.size:
        test_mask[tuple(pos_te.T)] = 1

    used_train = set(map(tuple, np.vstack([pos_tr, rn_tr]))) if rn_tr.size else set(map(tuple, pos_tr))

    neg_candidates = np.array([idx for idx in all_neg_idx if (idx[0], idx[1]) not in used_train])
    n_pos_te = len(pos_te)
    if n_pos_te > 0 and len(neg_candidates) >= n_pos_te:
        rng = np.random.default_rng(seed=fold)
        sel = rng.choice(len(neg_candidates), size=n_pos_te, replace=False)
        neg_te = neg_candidates[sel]
        test_mask[tuple(neg_te.T)] = 1

    pd.DataFrame(train_mask, index=adj_df.index, columns=adj_df.columns).to_csv(f"{MASK_CSV_DIR}/fold_{fold}_train_mask.csv")
    pd.DataFrame(test_mask,  index=adj_df.index, columns=adj_df.columns).to_csv(f"{MASK_CSV_DIR}/fold_{fold}_test_mask.csv")

    print(f"✅ Saved: fold_{fold}_train_mask.csv, fold_{fold}_test_mask.csv")

print(f"\n✅ All mask CSVs saved to {MASK_CSV_DIR}")