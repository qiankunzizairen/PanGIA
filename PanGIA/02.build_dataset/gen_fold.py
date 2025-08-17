import os
import numpy as np
import pandas as pd

def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# ---- load maps (RNA name -> ID, Disease name -> ID) ----
rna_map = pd.read_excel("allR_id.xlsx")
dis_map = pd.read_excel("disease_ID.xlsx")
rna_name_to_id = dict(zip(rna_map["RNA"].astype(str), rna_map["ID"].astype(str)))
dis_name_to_id = dict(zip(dis_map["disease"].astype(str), dis_map["ID"].astype(str)))

# ---- load adjacency (rows = RNA name, cols = Disease name) ----
adj_df = pd.read_csv("RDA_Matrix.csv", index_col=0)
adj = adj_df.values
num_p, num_d = adj.shape

# keep row/col orders, mapped to global IDs (may contain NaN if not found)
rna_ids_order = pd.Series(adj_df.index.astype(str)).map(rna_name_to_id).values
dis_ids_order = pd.Series(adj_df.columns.astype(str)).map(dis_name_to_id).values

# ---- build index pairs by matrix position ----
pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)

rng = np.random.default_rng(42)
rng.shuffle(pos_ij)
rng.shuffle(unlabelled_ij)

k_fold = 5
pos_ij_5fold = np.array_split(pos_ij, k_fold)
unlabelled_ij_5fold = np.array_split(unlabelled_ij, k_fold)

def idx_pairs_to_ids(pairs, row_ids_order, col_ids_order):
    if pairs.size == 0:
        return np.empty((0, 2), dtype=object)
    rids = row_ids_order[pairs[:, 0]]
    dids = col_ids_order[pairs[:, 1]]
    mask = (~pd.isna(rids)) & (~pd.isna(dids))
    rids = rids[mask].astype(str)
    dids = dids[mask].astype(str)
    return np.stack([rids, dids], axis=1)

out_dir = "./fold_csv"
os.makedirs(out_dir, exist_ok=True)

for i in range(k_fold):
    extract_idx = list(range(k_fold))
    extract_idx.remove(i)

    pos_train_ij = (
        np.vstack([pos_ij_5fold[idx] for idx in extract_idx])
        if extract_idx else np.empty((0, 2), dtype=int)
    )
    pos_test_ij = pos_ij_5fold[i]

    unlabelled_train_ij = (
        np.vstack([unlabelled_ij_5fold[idx] for idx in extract_idx])
        if extract_idx else np.empty((0, 2), dtype=int)
    )
    unlabelled_test_ij = unlabelled_ij_5fold[i]

    # convert matrix-position pairs -> global ID pairs
    pos_train_ids = idx_pairs_to_ids(pos_train_ij, rna_ids_order, dis_ids_order)
    pos_test_ids = idx_pairs_to_ids(pos_test_ij, rna_ids_order, dis_ids_order)
    unlabelled_train_ids = idx_pairs_to_ids(unlabelled_train_ij, rna_ids_order, dis_ids_order)
    unlabelled_test_ids = idx_pairs_to_ids(unlabelled_test_ij, rna_ids_order, dis_ids_order)

    # save without header, two columns: [RNA_ID, Disease_ID]
    pd.DataFrame(pos_train_ids).to_csv(f"{out_dir}/fold_{i}_pos_train.csv", index=False, header=False)
    pd.DataFrame(pos_test_ids).to_csv(f"{out_dir}/fold_{i}_pos_test.csv", index=False, header=False)
    pd.DataFrame(unlabelled_train_ids).to_csv(f"{out_dir}/fold_{i}_unlabelled_train.csv", index=False, header=False)
    pd.DataFrame(unlabelled_test_ids).to_csv(f"{out_dir}/fold_{i}_unlabelled_test.csv", index=False, header=False)