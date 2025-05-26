#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-only script for MMoE-HAN on ncRNA–disease data.
Evaluates AUC, AUPR, RI on training set during training.

Hyperparameters `hidden_dim`, `expert_dim`, `num_experts`, `num_heads`, and
`dropout` must be provided at runtime via command-line arguments.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn import metrics

from model import MMoEHANNet


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without default values."""
    parser = argparse.ArgumentParser(
        description="Train MMoE-HAN on ncRNA–disease heterogeneous graph"
    )
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension size of node embeddings")
    parser.add_argument("--expert_dim", type=int, required=True, help="Dimension of each expert in MMoE")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts in MMoE")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads in HAN")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (default: 0.005)")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    return parser.parse_args()


def compute_auc_aupr_ri(true: torch.Tensor, pred: torch.Tensor):
    """Compute AUC, AUPR and Ranking Index (RI)."""
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


# --------------------
HETERO_PATH = "./processed/data_hetero_.pt"
MASK_DIR = "./mask_csv"
ADJ_PATH = "./data/RDA_Matrix.csv"
RNA_NAMES_PATH = "./processed_csv/rna_names.csv"
DIS_NAMES_PATH = "./processed_csv/dis_names.csv"
RESULTS_DIR = "./scores"
NEW_ASSOC_DIR = "./new_associations"


def main() -> None:
    args = parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(NEW_ASSOC_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Load data
    raw_data = torch.load(HETERO_PATH)
    data = raw_data.to(device)
    adj_np = pd.read_csv(ADJ_PATH, index_col=0).values
    adj = torch.FloatTensor(adj_np).to(device)

    rna_types = ["mi-d", "circ-d", "lnc-d", "pi-d"]
    rna_sizes = [
        data["rna", rel, "disease"].edge_index[0].unique().numel()
        for rel in rna_types
    ]
    offsets = np.cumsum([0] + rna_sizes[:-1])

    rna_indices = {
        rel: torch.unique(data["rna", rel, "disease"].edge_index[0]).to(device)
        for rel in rna_types
    }

    rna_names = pd.read_csv(RNA_NAMES_PATH)["rna_name"].values
    dis_names = pd.read_csv(DIS_NAMES_PATH)["disease_name"].values

    # --------------------
    # Train
    for fold in range(args.folds):
        print(f"\n=== Fold {fold} ===")

        train_mask_path = f"{MASK_DIR}/fold_{fold}_train_mask.csv"
        train_mask = (
            pd.read_csv(train_mask_path, index_col=0).values.astype(bool)
        )
        train_mask = torch.tensor(train_mask, dtype=torch.bool)

        model = MMoEHANNet(
            metadata=data.metadata(),
            rna_in_dim=data["rna"].x.size(1),
            d_feat_dim=data["disease"].x.size(1),
            hidden_dim=args.hidden_dim,
            expert_dim=args.expert_dim,
            num_experts=args.num_experts,
            num_heads=args.num_heads,
            num_tasks=4,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            out_dict = model(data, rna_indices)

            loss = 0.0
            all_labels, all_preds = [], []
            for idx, rel in enumerate(rna_types):
                start = offsets[idx]
                size = rna_sizes[idx]
                labels = torch.FloatTensor(adj_np[start : start + size]).to(device)
                mask = train_mask[start : start + size]
                pred = out_dict[rel]

                loss += F.binary_cross_entropy(pred[mask], labels[mask])

                all_labels.append(labels[mask].detach().cpu())
                all_preds.append(pred[mask].detach().cpu())

            loss.backward()
            optimizer.step()

            all_labels_cat = torch.cat(all_labels)
            all_preds_cat = torch.cat(all_preds)
            auc, aupr, ri = compute_auc_aupr_ri(all_labels_cat, all_preds_cat)

            print(
                f"[Train] Fold {fold} Epoch {epoch}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | AUC: {auc:.4f} | "
                f"AUPR: {aupr:.4f} | RI: {ri:.4f}"
            )

        # Save model checkpoint
        checkpoint_path = os.path.join(RESULTS_DIR, f"mmoe_han_fold{fold}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # --------------------
        # Save new RNA–disease associations
        model.eval()
        with torch.no_grad():
            out_dict = model(data, rna_indices)
            for idx, rel in enumerate(rna_types):
                start = offsets[idx]
                size = rna_sizes[idx]
                pred = out_dict[rel].cpu().numpy()
                label = adj_np[start : start + size, :]
                unknown_idx = np.argwhere(label == 0)

                records = []
                for i, j in unknown_idx:
                    rna_name = rna_names[start + i]
                    dis_name = dis_names[j]
                    score = pred[i, j]
                    records.append((rna_name, dis_name, score))

                df = pd.DataFrame(records, columns=["RNA", "Disease", "Score"])
                df.sort_values("Score", ascending=False, inplace=True)
                csv_path = f"{NEW_ASSOC_DIR}/fold_{fold}_{rel}_new_assoc.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
