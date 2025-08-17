#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm

def read_maps():
    rna_map = pd.read_excel("allR_id.xlsx")
    dis_map = pd.read_excel("disease_ID.xlsx")
    rna_map = rna_map.sort_values("ID").reset_index(drop=True)
    dis_map = dis_map.sort_values("ID").reset_index(drop=True)
    rna_name_to_id = dict(zip(rna_map["RNA"], rna_map["ID"].astype(str)))
    dis_name_to_id = dict(zip(dis_map["disease"], dis_map["ID"].astype(str)))
    return rna_name_to_id, dis_name_to_id

def build_id_pos_maps(adj_df, rna_name_to_id, dis_name_to_id):
    rna_names = adj_df.index.tolist()
    dis_names = adj_df.columns.tolist()
    rna_ids_by_pos = [rna_name_to_id[n] for n in rna_names]
    dis_ids_by_pos = [dis_name_to_id[n] for n in dis_names]
    rna_id2pos = {rid: i for i, rid in enumerate(rna_ids_by_pos)}
    dis_id2pos = {did: j for j, did in enumerate(dis_ids_by_pos)}
    return rna_ids_by_pos, dis_ids_by_pos, rna_id2pos, dis_id2pos

def load_and_align_similarity(path, target_names):
    df = pd.read_csv(path, index_col=0)
    df = df.reindex(index=target_names, columns=target_names)
    if df.isna().any().any():
        na_locs = list(zip(*np.where(df.isna().values)))
        raise ValueError(f"NaNs after reindex in {os.path.basename(path)}; example positions: {na_locs[:10]}")
    return df.values

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp_model(x_np, y_np, device, n_epochs=20, batch_size=2048):
    model = MLP(input_dim=x_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.tensor(x_np, dtype=torch.float32),
                            torch.tensor(y_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    model.train()
    for epoch in range(n_epochs):
        batch_bar = tqdm(loader, desc=f"Epoch [{epoch+1}/{n_epochs}]")
        for xb, yb in batch_bar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
    return model

def predict_proba_mlp(model, x_np, device, batch_size=1024):
    model.eval()
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rna_name_to_id, dis_name_to_id = read_maps()

    adj_df = pd.read_csv("RDA_Matrix.csv", index_col=0)
    rna_ids_by_pos, dis_ids_by_pos, rna_id2pos, dis_id2pos = build_id_pos_maps(adj_df, rna_name_to_id, dis_name_to_id)

    p_sim = load_and_align_similarity("Integrated_RNA.csv", adj_df.index.tolist())
    d_sim = load_and_align_similarity("Integrated_Disease.csv", adj_df.columns.tolist())

    adj_np = adj_df.values.astype(np.float32)
    num_p, num_d = adj_np.shape

    n_pca_p_sim = min(85, max(1, min(num_p, p_sim.shape[1])))
    pca_p = PCA(n_components=n_pca_p_sim)
    pca_p_feat = pca_p.fit_transform(p_sim)

    pca_d = PCA()
    pca_d_feat = pca_d.fit_transform(d_sim)

    feat_dim = n_pca_p_sim + pca_d_feat.shape[1]
    feat_mat = np.zeros((num_p, num_d, feat_dim), dtype=np.float32)
    for i in range(num_p):
        for j in range(num_d):
            feat_mat[i, j] = np.concatenate([pca_p_feat[i], pca_d_feat[j]])

    os.makedirs("./classifier", exist_ok=True)
    os.makedirs("./rn_csv", exist_ok=True)

    rna_ids_by_pos_arr = np.array(rna_ids_by_pos, dtype=object)
    dis_ids_by_pos_arr = np.array(dis_ids_by_pos, dtype=object)

    for fold in range(5):
        print(f"\nfold {fold}")

        pos_train_id = pd.read_csv(f"./fold_csv/fold_{fold}_pos_train.csv", header=None).values.astype(str)
        unlab_train_id = pd.read_csv(f"./fold_csv/fold_{fold}_unlabelled_train.csv", header=None).values.astype(str)

        pos_train_ij = np.column_stack([
            np.vectorize(rna_id2pos.get)(pos_train_id[:, 0]),
            np.vectorize(dis_id2pos.get)(pos_train_id[:, 1])
        ]).astype(int)

        unlab_train_ij = np.column_stack([
            np.vectorize(rna_id2pos.get)(unlab_train_id[:, 0]),
            np.vectorize(dis_id2pos.get)(unlab_train_id[:, 1])
        ]).astype(int)

        if (pos_train_ij < 0).any() or (unlab_train_ij < 0).any():
            raise ValueError("Found IDs in fold CSV that are not present in mapping/order from RDA_Matrix.csv.")

        train_ij = np.vstack([pos_train_ij, unlab_train_ij])
        train_feat = feat_mat[train_ij[:, 0], train_ij[:, 1], :]

        n_spy = max(1, int(len(pos_train_ij) * 0.1)) if len(pos_train_ij) > 0 else 0
        if n_spy > 0:
            spy_idx = np.random.choice(len(pos_train_ij), n_spy, replace=False)
            spy_ij = pos_train_ij[spy_idx]
        else:
            spy_ij = np.empty((0, 2), dtype=int)

        adj_hidden = adj_np.copy()
        if len(spy_ij) > 0:
            adj_hidden[spy_ij[:, 0], spy_ij[:, 1]] = 0
        train_label = adj_hidden[train_ij[:, 0], train_ij[:, 1]]

        model = train_mlp_model(train_feat, train_label, device)
        torch.save(model.state_dict(), f"./classifier/mlp_f85_fold{fold}.pt")

        train_prob = predict_proba_mlp(model, train_feat, device)
        prob_mat = np.zeros_like(adj_np, dtype=np.float32)
        prob_mat[train_ij[:, 0], train_ij[:, 1]] = train_prob

        if len(spy_ij) > 0:
            spy_prob = prob_mat[spy_ij[:, 0], spy_ij[:, 1]]
            thresh = np.sort(spy_prob)[int(len(spy_prob) * 0.05)]
        else:
            thresh = 0.5

        if len(unlab_train_ij) > 0:
            unlab_prob = prob_mat[unlab_train_ij[:, 0], unlab_train_ij[:, 1]]
            rn_mask = unlab_prob < thresh
            rn_ij_pos = unlab_train_ij[rn_mask]
        else:
            rn_ij_pos = np.empty((0, 2), dtype=int)

        max_rn = len(pos_train_ij) * 2
        if len(rn_ij_pos) > max_rn and max_rn > 0:
            idx = np.random.choice(len(rn_ij_pos), max_rn, replace=False)
            rn_ij_pos = rn_ij_pos[idx]

        rna_id_out = rna_ids_by_pos_arr[rn_ij_pos[:, 0]]
        dis_id_out = dis_ids_by_pos_arr[rn_ij_pos[:, 1]]

        df_rn = pd.DataFrame({"rna_ID": rna_id_out, "disease_ID": dis_id_out})
        df_rn.to_csv(f"./rn_csv/fold_{fold}_rn.csv", index=False)
        print(f"saved RN: {len(df_rn)} to ./rn_csv/fold_{fold}_rn.csv")

if __name__ == "__main__":
    main()