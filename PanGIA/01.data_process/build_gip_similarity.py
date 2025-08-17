import numpy as np
import pandas as pd
from pathlib import Path

def gip_similarity(X: np.ndarray) -> np.ndarray:
    row_norm2 = np.sum(X**2, axis=1)
    mean_norm2 = float(np.mean(row_norm2)) if row_norm2.size else 0.0
    gamma = 1.0 / mean_norm2 if mean_norm2 > 0 else 1.0
    G = X @ X.T
    D2 = row_norm2[:, None] + row_norm2[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    S = np.exp(-gamma * D2)
    np.fill_diagonal(S, 1.0)
    return S

def main(input_csv="RDA_Matrix.csv", out_rna_csv="GIP_RNA.csv", out_dis_csv="GIP_Disease.csv"):
    adj = pd.read_csv(input_csv, index_col=0)
    adj = adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rna_names = adj.index.tolist()
    rna_S = gip_similarity(adj.values)
    pd.DataFrame(rna_S, index=rna_names, columns=rna_names).to_csv(out_rna_csv, float_format="%.10f")
    dis_names = adj.columns.tolist()
    dis_S = gip_similarity(adj.values.T)
    pd.DataFrame(dis_S, index=dis_names, columns=dis_names).to_csv(out_dis_csv, float_format="%.10f")
    print("Done.")
    print(f"- RNA GIP -> {Path(out_rna_csv).resolve()}")
    print(f"- Disease GIP -> {Path(out_dis_csv).resolve()}")

if __name__ == "__main__":
    main()