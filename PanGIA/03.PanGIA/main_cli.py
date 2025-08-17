#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/eval MMoE-HAN with weighted hetero graph and export new association candidates.
Hyperparameters are provided via CLI arguments (no defaults).
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics

from model import MMoEHANNet  

# ------------------------------
# Paths and constants
# ------------------------------
DATA_DIR         = "./"
PROC_DIR         = "./"
HETERO_PATH      = os.path.join(PROC_DIR, "data_hetero.pt")
RNA_ID_ORDER_TXT = os.path.join(PROC_DIR, "rna_ids_order.txt")
DIS_ID_ORDER_TXT = os.path.join(PROC_DIR, "disease_ids_order.txt")
ADJ_PATH         = os.path.join(DATA_DIR, "RDA_Matrix.csv")
RNA_MAP_XLSX     = os.path.join(DATA_DIR, "allR_id.xlsx")        # columns: ID, RNA
DIS_MAP_XLSX     = os.path.join(DATA_DIR, "disease_ID.xlsx")     # columns: ID, disease
MASK_DIR         = "./mask_csv"
RESULTS_DIR      = "./scores"
NEW_ASSOC_DIR    = "./new_associations"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NEW_ASSOC_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/eval MMoE-HAN with CLI hyperparameters."
    )
    parser.add_argument("--hidden-dim", type=int, required=True,
                        help="Hidden dim per HAN head (and encoder output).")
    parser.add_argument("--expert-dim", type=int, required=True,
                        help="Output dim of each expert and task head.")
    parser.add_argument("--num-experts", type=int, required=True,
                        help="Number of experts in MMoE.")
    parser.add_argument("--num-heads", type=int, required=True,
                        help="Number of attention heads in HAN / cross-attn.")
    parser.add_argument("--dropout", type=float, required=True,
                        help="Dropout rate used in encoder / heads.")
    # (optional but handy)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate.")
    return parser.parse_args()


# ------------------------------
# Metrics
# ------------------------------
def compute_auc_aupr_ri(true, pred):
    """
    Compute AUC, AUPR and RI (rank index) on flattened vectors.
    true, pred: 1D torch tensors on GPU
    """
    labels = true.cpu().numpy()
    scores = pred.cpu().numpy()

    # RI
    combined = list(zip(labels, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    labels_sorted, _ = zip(*combined)
    indices = np.arange(1, len(labels) + 1)[np.array(labels_sorted) == 1]
    n_test = len(labels)
    n_test_p = (labels == 1).sum()
    ri = indices.sum() / n_test / n_test_p if n_test_p > 0 else 0.0

    # AUC
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)

    # AUPR
    precisions, recalls, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recalls, precisions)

    return round(auc, 6), round(aupr, 6), round(ri, 6)


# ------------------------------
# Loaders and reindex helpers
# ------------------------------
def load_graph_and_orders():
    """
    Load hetero graph and node ID orders (both orders are lists of string IDs).
    """
    data = torch.load(HETERO_PATH, weights_only=False, map_location=device)
    data = data.to(device)
    with open(RNA_ID_ORDER_TXT, "r", encoding="utf-8") as f:
        rna_ids_order = [line.strip() for line in f if line.strip()]
    with open(DIS_ID_ORDER_TXT, "r", encoding="utf-8") as f:
        dis_ids_order = [line.strip() for line in f if line.strip()]
    return data, rna_ids_order, dis_ids_order


def load_id_maps():
    """
    Return four dicts:
      - RNA name -> ID, Disease name -> ID
      - RNA ID   -> name, Disease ID   -> name
    All IDs are strings.
    """
    rmap = pd.read_excel(RNA_MAP_XLSX)
    dmap = pd.read_excel(DIS_MAP_XLSX)

    rna_name_to_id = dict(zip(rmap["RNA"].astype(str),     rmap["ID"].astype(str)))
    dis_name_to_id = dict(zip(dmap["disease"].astype(str), dmap["ID"].astype(str)))
    rna_id_to_name = dict(zip(rmap["ID"].astype(str),      rmap["RNA"].astype(str)))
    dis_id_to_name = dict(zip(dmap["ID"].astype(str),      dmap["disease"].astype(str)))
    return rna_name_to_id, dis_name_to_id, rna_id_to_name, dis_id_to_name


def reindex_adj_to_id_order(adj_csv_path, rna_ids_order, dis_ids_order,
                            rna_name_to_id, dis_name_to_id):
    """
    Read RDA_Matrix.csv whose index=RNA names, columns=Disease names.
    Convert names -> IDs via mapping, drop NaNs, and reindex to the global ID order.
    Return a float32 dense numpy array aligned with (rna_ids_order x dis_ids_order).
    """
    df = pd.read_csv(adj_csv_path, index_col=0)

    df.index   = pd.Series(df.index.astype(str)).map(rna_name_to_id)
    df.columns = pd.Series(df.columns.astype(str)).map(dis_name_to_id)
    df = df.dropna(axis=0, how="any").dropna(axis=1, how="any")

    df = df.reindex(index=rna_ids_order, columns=dis_ids_order).fillna(0.0)
    return df.values.astype(np.float32)


def load_and_align_masks(fold, rna_ids_order, dis_ids_order,
                         rna_name_to_id, dis_name_to_id):
    """
    Load train/test masks (CSV with names), map to IDs, drop NaNs, and align
    to the same (RNA_ID x Disease_ID) order used by the graph.
    Return torch.bool tensors on the right device.
    """
    train_df = pd.read_csv(os.path.join(MASK_DIR, f"fold_{fold}_train_mask.csv"), index_col=0)
    test_df  = pd.read_csv(os.path.join(MASK_DIR, f"fold_{fold}_test_mask.csv"),  index_col=0)

    for df in [train_df, test_df]:
        df.index   = pd.Series(df.index.astype(str)).map(rna_name_to_id)
        df.columns = pd.Series(df.columns.astype(str)).map(dis_name_to_id)
        df.dropna(axis=0, how="any", inplace=True)
        df.dropna(axis=1, how="any", inplace=True)

    train_df = train_df.reindex(index=rna_ids_order, columns=dis_ids_order).fillna(0).astype(bool)
    test_df  = test_df .reindex(index=rna_ids_order, columns=dis_ids_order).fillna(0).astype(bool)

    return torch.tensor(train_df.values, dtype=torch.bool, device=device), \
           torch.tensor(test_df.values,  dtype=torch.bool, device=device)


# ------------------------------
# Main training/evaluation/prediction
# ------------------------------
def main():
    args = parse_args()

    data, rna_ids_order, dis_ids_order = load_graph_and_orders()
    rna_name_to_id, dis_name_to_id, rna_id_to_name, dis_id_to_name = load_id_maps()

    # Align adjacency to the exact ID order used by the graph node feature tensors
    adj_np = reindex_adj_to_id_order(
        ADJ_PATH, rna_ids_order, dis_ids_order, rna_name_to_id, dis_name_to_id
    )
    adj = torch.tensor(adj_np, dtype=torch.float32, device=device)

    # RNA relation types (4 tasks) and their RNA index sets in the global graph
    rna_types = ['mi-d', 'circ-d', 'lnc-d', 'pi-d']
    rna_indices = {
        rel: torch.unique(data['rna', rel, 'disease'].edge_index[0]).to(device)
        for rel in rna_types
    }

    # Accumulate per-fold metrics
    test_aucs, test_auprs, test_ris = [], [], []

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")

        # Load and align masks to ID order
        train_mask, test_mask = load_and_align_masks(
            fold, rna_ids_order, dis_ids_order, rna_name_to_id, dis_name_to_id
        )

        # -------- Build model (use CLI hyperparams, no defaults here) --------
        model = MMoEHANNet(
            metadata    = data.metadata(),
            rna_in_dim  = data['rna'].x.size(1),
            d_feat_dim  = data['disease'].x.size(1),
            hidden_dim  = args.hidden_dim,
            expert_dim  = args.expert_dim,
            num_experts = args.num_experts,
            num_heads   = args.num_heads,
            num_tasks   = 4,
            dropout     = args.dropout
        ).to(device)

        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
        num_epochs = args.epochs

        # -------- Training --------
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out_dict = model(data, rna_indices)   # {rel: [n_t, N_d]}

            loss = 0.0
            all_labels, all_preds = [], []
            for rel in rna_types:
                idxs   = rna_indices[rel]       # [n_t]
                labels = adj[idxs, :]           # [n_t, N_d]
                mask   = train_mask[idxs, :]    # [n_t, N_d]
                pred   = out_dict[rel]          # [n_t, N_d]

                loss += F.binary_cross_entropy(pred[mask], labels[mask])
                all_labels.append(labels[mask].detach().cpu())
                all_preds.append(pred[mask].detach().cpu())

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == 1:
                all_labels = torch.cat(all_labels)
                all_preds  = torch.cat(all_preds)
                auc, aupr, ri = compute_auc_aupr_ri(all_labels, all_preds)
                print(f"[Train] Fold {fold} Epoch {epoch}/{num_epochs} | "
                      f"Loss: {loss.item():.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f} | RI: {ri:.4f}")

        # Save trained weights
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"mmoe_han_fold{fold}.pt"))
        print(f"Model weights saved: mmoe_han_fold{fold}.pt")

        # -------- Evaluation --------
        model.eval()
        with torch.no_grad():
            out_dict = model(data, rna_indices)

            all_labels, all_preds = [], []
            for rel in rna_types:
                idxs   = rna_indices[rel]
                labels = adj[idxs, :]
                mask   = test_mask[idxs, :]
                pred   = out_dict[rel]

                all_labels.append(labels[mask].detach().cpu())
                all_preds.append(pred[mask].detach().cpu())

            all_labels = torch.cat(all_labels)
            all_preds  = torch.cat(all_preds)
            auc_t, aupr_t, ri_t = compute_auc_aupr_ri(all_labels, all_preds)

        print(f"[Test]  Fold {fold} | AUC: {auc_t:.4f} | AUPR: {aupr_t:.4f} | RI: {ri_t:.4f}")
        test_aucs.append(auc_t); test_auprs.append(aupr_t); test_ris.append(ri_t)

        # -------- New association candidates (unknown links only) --------
        new_rows = []
        with torch.no_grad():
            for rel in rna_types:
                idxs = rna_indices[rel]                 # [n_t]
                pred = out_dict[rel].cpu().numpy()      # [n_t, N_d]
                lbls = adj[idxs, :].cpu().numpy()       # [n_t, N_d]

                # only positions that were 0 in the original adjacency (unknown)
                mask_new = (lbls == 0)
                if not mask_new.any():
                    continue

                r_idx, d_idx = np.where(mask_new)
                scores = pred[r_idx, d_idx]

                for rr, dd, sc in zip(r_idx, d_idx, scores):
                    r_global = int(idxs[rr].item())     # RNA row index in ID order
                    d_global = int(dd)                  # Disease col index in ID order
                    r_id = rna_ids_order[r_global]
                    d_id = dis_ids_order[d_global]
                    r_name = rna_id_to_name.get(r_id, r_id)
                    d_name = dis_id_to_name.get(d_id, d_id)
                    new_rows.append([rel, r_id, d_id, r_name, d_name, float(sc)])

        # Save the "all candidates" file (may be large)
        df_all = pd.DataFrame(new_rows, columns=[
            "task_rel", "RNA_ID", "Disease_ID", "RNA_name", "Disease_name", "score"
        ])
        df_all.sort_values("score", ascending=False, inplace=True)
        all_path = os.path.join(NEW_ASSOC_DIR, f"fold_{fold}_new_candidates_all.csv")
        df_all.to_csv(all_path, index=False)

        # Also save per RNA type separately
        rel_to_fname = {
            'mi-d':   f"fold_{fold}_new_candidates_mi-d.csv",
            'circ-d': f"fold_{fold}_new_candidates_circ-d.csv",
            'lnc-d':  f"fold_{fold}_new_candidates_lnc-d.csv",
            'pi-d':   f"fold_{fold}_new_candidates_pi-d.csv",
        }
        for rel, fname in rel_to_fname.items():
            sub = df_all[df_all["task_rel"] == rel].copy()
            if len(sub) == 0:
                sub = sub.reindex(columns=df_all.columns)
            sub.sort_values("score", ascending=False, inplace=True)
            sub.to_csv(os.path.join(NEW_ASSOC_DIR, fname), index=False)

    # -------- Summary across folds --------
    print("\n=== Overall Validation Performance ===")
    print(f"Mean AUC:  {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Mean AUPR: {np.mean(test_auprs):.4f} ± {np.std(test_auprs):.4f}")
    print(f"Mean RI:   {np.mean(test_ris):.4f} ± {np.std(test_ris):.4f}")


if __name__ == "__main__":
    main()