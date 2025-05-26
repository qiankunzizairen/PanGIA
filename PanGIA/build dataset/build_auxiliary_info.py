# build_auxiliary_info.py
# output:
#   - rna_indices.pkl
#   - rna_names.npy
#   - dis_names.npy

import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

# ------------ Paths ------------
data_dir = "./data"
save_dir = "./processed"
os.makedirs(save_dir, exist_ok=True)

# ------------ Load data_hetero.pt ------------
hetero_path = os.path.join(save_dir, "data_hetero_.pt")
data = torch.load(hetero_path)

# ------------ Build rna_indices ------------
rna_indices = {}
for rel in ['mi-d', 'circ-d', 'lnc-d', 'pi-d']:
    src = data['rna', rel, 'disease'].edge_index[0]
    rna_indices[rel] = torch.unique(src)

# Save rna_indices
with open(os.path.join(save_dir, "rna_indices.pkl"), "wb") as f:
    pickle.dump(rna_indices, f)
print(f"rna_indices.pkl saved to {save_dir}")

# ------------ Extract RNA and disease names ------------
csv_path = os.path.join(data_dir, "RDA_Matrix.csv")
adj_df = pd.read_csv(csv_path, index_col=0)

rna_names = np.array(adj_df.index.tolist())   # row index
dis_names = np.array(adj_df.columns.tolist()) # column headers

# Save names
np.save(os.path.join(save_dir, "rna_names.npy"), rna_names)
np.save(os.path.join(save_dir, "dis_names.npy"), dis_names)
print(f"rna_names.npy and dis_names.npy saved to {save_dir}")