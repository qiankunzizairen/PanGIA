#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

DATA_DIR = "./"
OUT_PATH = "./data_hetero.pt"
RNA_DIRS = {
    "mi": "./miR",
    "circ": "./circR",
    "lnc": "./lncR",
    "pi": "./piR",
}
RNA_SIM_CANDIDATES = [
    os.path.join(DATA_DIR, "Integrated_RNA.csv"),
]
DIS_SIM_CANDIDATES = [
    os.path.join(DATA_DIR, "Integrated_Disease.csv"),
]

def read_maps():
    rna_map = pd.read_excel(os.path.join(DATA_DIR, "allR_id.xlsx"))
    dis_map = pd.read_excel(os.path.join(DATA_DIR, "disease_ID.xlsx"))
    rna_map = rna_map.sort_values("ID").reset_index(drop=True)
    dis_map = dis_map.sort_values("ID").reset_index(drop=True)
    rna_ids = rna_map["ID"].astype(str).tolist()
    dis_ids = dis_map["ID"].astype(str).tolist()
    rna_name_to_id = dict(zip(rna_map["RNA"], rna_map["ID"].astype(str)))
    dis_name_to_id = dict(zip(dis_map["disease"], dis_map["ID"].astype(str)))
    return rna_ids, dis_ids, rna_name_to_id, dis_name_to_id

def load_adj_as_id(rna_ids, dis_ids, rna_name_to_id, dis_name_to_id):
    df = pd.read_csv(os.path.join(DATA_DIR, "RDA_Matrix.csv"), index_col=0)
    idx = df.index.to_series()
    cols = pd.Series(df.columns)
    if not set(idx).issubset(set(rna_ids)):
        idx = idx.map(rna_name_to_id)
    if not set(cols).issubset(set(dis_ids)):
        cols = cols.map(dis_name_to_id)
    df.index = idx
    df.columns = cols
    df = df.dropna(axis=0, how="any").dropna(axis=1, how="any")
    df = df.reindex(index=rna_ids, columns=dis_ids).fillna(0).astype(np.float32)
    return df

def load_dnabert_dir(d):
    ids_path = os.path.join(d, "ids.txt")
    emb_path = os.path.join(d, "embeddings.npy")
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f]
    emb = np.load(emb_path)
    return ids, emb

def load_all_rna_embeddings_to_id(rna_name_to_id):
    type2idset = {}
    id_to_vec = {}
    dim = None
    for t, d in RNA_DIRS.items():
        ids, emb = load_dnabert_dir(d)
        cur_set = set()
        for k, v in zip(ids, emb):
            rid = rna_name_to_id.get(k, None)
            if rid is None:
                continue
            id_to_vec[rid] = v
            cur_set.add(rid)
            if dim is None:
                dim = len(v)
        type2idset[t] = cur_set
    if dim is None:
        dim = 0
    return type2idset, id_to_vec, dim

def assemble_rna_features_by_id_order(rna_ids, id_to_vec, dim):
    X = np.zeros((len(rna_ids), dim), dtype=np.float32)
    for i, rid in enumerate(rna_ids):
        vec = id_to_vec.get(rid, None)
        if vec is not None:
            X[i] = vec
    return torch.tensor(X, dtype=torch.float32)

def load_similarity_matrix_id_first(candidates, row_ids, col_ids=None, name_to_id=None):
    for p in candidates:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0)
        if col_ids is None:
            if set(df.index) == set(row_ids) and set(df.columns) == set(row_ids):
                df = df.reindex(index=row_ids, columns=row_ids)
                return df.astype(np.float32)
            if name_to_id is not None:
                df.index = df.index.to_series().map(name_to_id)
                df.columns = df.columns.to_series().map(name_to_id)
                df = df.dropna(axis=0, how="any").dropna(axis=1, how="any")
                try:
                    df = df.reindex(index=row_ids, columns=row_ids).astype(np.float32)
                    return df
                except Exception:
                    continue
        else:
            if set(df.index) == set(row_ids) and set(df.columns) == set(col_ids):
                df = df.reindex(index=row_ids, columns=col_ids)
                return df.astype(np.float32)
            if name_to_id is not None:
                row_map, col_map = name_to_id
                df.index = df.index.to_series().map(row_map)
                df.columns = df.columns.to_series().map(col_map)
                df = df.dropna(axis=0, how="any").dropna(axis=1, how="any")
                try:
                    df = df.reindex(index=row_ids, columns=col_ids).astype(np.float32)
                    return df
                except Exception:
                    continue
    return None

def dense_to_edges(sim_df, thresh=0.0, directed=True):
    A = sim_df.values.astype(np.float32)
    rows, cols = np.where(A > thresh)
    w = A[rows, cols].astype(np.float32)
    if rows.size == 0:
        ei = torch.empty((2,0), dtype=torch.long)
        ew = torch.empty((0,), dtype=torch.float32)
        return ei, ew
    ei = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    ew = torch.tensor(w, dtype=torch.float32)
    if not directed:
        rr = torch.tensor(np.vstack([cols, rows]), dtype=torch.long)
        ww = torch.tensor(w, dtype=torch.float32)
        ei = torch.cat([ei, rr], dim=1)
        ew = torch.cat([ew, ww], dim=0)
    return ei, ew

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rna_ids, dis_ids, rna_name_to_id, dis_name_to_id = read_maps()
    adj_id_df = load_adj_as_id(rna_ids, dis_ids, rna_name_to_id, dis_name_to_id)
    type2idset, id_to_vec, dim = load_all_rna_embeddings_to_id(rna_name_to_id)
    X_rna = assemble_rna_features_by_id_order(rna_ids, id_to_vec, dim)
    dis_sim_df = load_similarity_matrix_id_first(DIS_SIM_CANDIDATES, dis_ids, None, name_to_id=dis_name_to_id)
    if dis_sim_df is not None:
        X_dis = torch.tensor(dis_sim_df.values, dtype=torch.float32)
    else:
        X_dis = torch.eye(len(dis_ids), dtype=torch.float32)
    data = HeteroData()
    data["rna"].x = X_rna
    data["disease"].x = X_dis
    rna_id_to_pos = {rid:i for i, rid in enumerate(rna_ids)}
    for t in ["mi", "circ", "lnc", "pi"]:
        idset = type2idset.get(t, set())
        if len(idset) == 0:
            data["rna", f"{t}-d", "disease"].edge_index = torch.empty((2,0), dtype=torch.long)
            data["rna", f"{t}-d", "disease"].edge_weight = torch.empty((0,), dtype=torch.float32)
            data["disease", f"d-{t}", "rna"].edge_index = torch.empty((2,0), dtype=torch.long)
            data["disease", f"d-{t}", "rna"].edge_weight = torch.empty((0,), dtype=torch.float32)
            continue
        row_idx = [rna_id_to_pos[rid] for rid in rna_ids if rid in idset]
        sub = adj_id_df.iloc[row_idx, :].values
        src_l, dst = np.nonzero(sub > 0.0)
        if len(src_l) > 0:
            src_global = np.array([row_idx[s] for s in src_l], dtype=np.int64)
            edge_idx = torch.tensor(np.vstack([src_global, dst]), dtype=torch.long)
            ew = torch.ones(edge_idx.size(1), dtype=torch.float32)
            data["rna", f"{t}-d", "disease"].edge_index = edge_idx
            data["rna", f"{t}-d", "disease"].edge_weight = ew
            data["disease", f"d-{t}", "rna"].edge_index = edge_idx.flip(0)
            data["disease", f"d-{t}", "rna"].edge_weight = ew.clone()
        else:
            data["rna", f"{t}-d", "disease"].edge_index = torch.empty((2,0), dtype=torch.long)
            data["rna", f"{t}-d", "disease"].edge_weight = torch.empty((0,), dtype=torch.float32)
            data["disease", f"d-{t}", "rna"].edge_index = torch.empty((2,0), dtype=torch.long)
            data["disease", f"d-{t}", "rna"].edge_weight = torch.empty((0,), dtype=torch.float32)
    rna_sim_df = load_similarity_matrix_id_first(RNA_SIM_CANDIDATES, rna_ids, None, name_to_id=rna_name_to_id)
    if rna_sim_df is not None:
        ei, ew = dense_to_edges(rna_sim_df, thresh=0.0, directed=True)
        data["rna", "rna", "rna"].edge_index = ei
        data["rna", "rna", "rna"].edge_weight = ew
    else:
        data["rna", "rna", "rna"].edge_index = torch.empty((2,0), dtype=torch.long)
        data["rna", "rna", "rna"].edge_weight = torch.empty((0,), dtype=torch.float32)
        print("No RNA similarity data found, using empty edges.")
    if dis_sim_df is not None:
        ei, ew = dense_to_edges(dis_sim_df, thresh=0.0, directed=True)
        data["disease", "disease", "disease"].edge_index = ei
        data["disease", "disease", "disease"].edge_weight = ew
    else:
        data["disease", "disease", "disease"].edge_index = torch.empty((2,0), dtype=torch.long)
        data["disease", "disease", "disease"].edge_weight = torch.empty((0,), dtype=torch.float32)
        print("No disease similarity data found, using empty edges.")
    torch.save(data, OUT_PATH)
    with open("./rna_ids_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rna_ids))
    with open("./disease_ids_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dis_ids))
    print(f"Saved hetero graph to {OUT_PATH}")

if __name__ == "__main__":
    main()