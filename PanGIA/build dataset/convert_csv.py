import os
import pickle
import numpy as np
import pandas as pd

# 设定输入和输出路径
save_dir = "./processed"
output_dir = "./processed_csv"
os.makedirs(output_dir, exist_ok=True)

# -------- rna_names.npy → rna_names.csv --------
rna_names = np.load(os.path.join(save_dir, "rna_names.npy"), allow_pickle=True)
rna_names_df = pd.DataFrame({'rna_index': np.arange(len(rna_names)), 'rna_name': rna_names})
rna_names_df.to_csv(os.path.join(output_dir, "rna_names.csv"), index=False)
print("✅ rna_names.csv saved.")

# -------- dis_names.npy → dis_names.csv --------
dis_names = np.load(os.path.join(save_dir, "dis_names.npy"), allow_pickle=True)
dis_names_df = pd.DataFrame({'disease_index': np.arange(len(dis_names)), 'disease_name': dis_names})
dis_names_df.to_csv(os.path.join(output_dir, "dis_names.csv"), index=False)
print("✅ dis_names.csv saved.")

# -------- rna_indices.pkl → rna_indices_{type}.csv --------
with open(os.path.join(save_dir, "rna_indices.pkl"), "rb") as f:
    rna_indices = pickle.load(f)

for rel, indices in rna_indices.items():
    indices = indices.cpu().numpy() if hasattr(indices, "cpu") else np.array(indices)
    df = pd.DataFrame({'rna_index': indices})
    df.to_csv(os.path.join(output_dir, f"rna_indices_{rel}.csv"), index=False)
    print(f"✅ rna_indices_{rel}.csv saved.")

print(f"All files saved to {output_dir}")