import os
import sys
import numpy as np
import pandas as pd

def read_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def check_and_align(A, B, nameA, nameB):
    idx_equal = list(A.index) == list(B.index)
    col_equal = list(A.columns) == list(B.columns)
    if not set(A.index) == set(B.index):
        only_A = sorted(set(A.index) - set(B.index))
        only_B = sorted(set(B.index) - set(A.index))
        print(f"[ERROR] {nameA} vs {nameB}: row label sets differ.")
        if only_A:
            print(f"  In {nameA} only (rows): {only_A[:20]}{' ...' if len(only_A)>20 else ''}")
        if only_B:
            print(f"  In {nameB} only (rows): {only_B[:20]}{' ...' if len(only_B)>20 else ''}")
        sys.exit(1)
    if not set(A.columns) == set(B.columns):
        only_A = sorted(set(A.columns) - set(B.columns))
        only_B = sorted(set(B.columns) - set(A.columns))
        print(f"[ERROR] {nameA} vs {nameB}: column label sets differ.")
        if only_A:
            print(f"  In {nameA} only (cols): {only_A[:20]}{' ...' if len(only_A)>20 else ''}")
        if only_B:
            print(f"  In {nameB} only (cols): {only_B[:20]}{' ...' if len(only_B)>20 else ''}")
        sys.exit(1)

    if not idx_equal:
        diffs = [(i, A.index[i], B.index[i]) for i in range(len(A.index)) if A.index[i] != B.index[i]]
        print(f"[CHECK] Row order differs in {len(diffs)} positions between {nameA} and {nameB}. Showing up to 20:")
        for i,(pos,a,b) in enumerate(diffs[:20]):
            print(f"  pos {pos}: {a} vs {b}")
    else:
        print(f"[CHECK] Row order identical for {nameA} and {nameB}.")

    if not col_equal:
        diffs = [(i, A.columns[i], B.columns[i]) for i in range(len(A.columns)) if A.columns[i] != B.columns[i]]
        print(f"[CHECK] Column order differs in {len(diffs)} positions between {nameA} and {nameB}. Showing up to 20:")
        for i,(pos,a,b) in enumerate(diffs[:20]):
            print(f"  pos {pos}: {a} vs {b}")
    else:
        print(f"[CHECK] Column order identical for {nameA} and {nameB}.")

    B_aligned = B.reindex(index=A.index, columns=A.columns)
    return A, B_aligned

def integrate_two(A, B):
    stacked = np.stack([A.values, B.values], axis=0)
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(stacked, axis=0)
    out = pd.DataFrame(avg, index=A.index, columns=A.columns)
    return out.round(6)

def main():
    dis_sem_path = "d2d_do.csv"
    dis_gip_path = "GIP_Disease.csv"
    rna_emb_path = "allR_cosine_similarity.csv"
    rna_gip_path = "GIP_RNA.csv"
    out_dis_path = "Integrated_Disease.csv"
    out_rna_path = "Integrated_RNA.csv"

    if not os.path.exists(dis_sem_path) or not os.path.exists(dis_gip_path):
        print(f"[ERROR] Missing disease files: {dis_sem_path} or {dis_gip_path}")
        sys.exit(1)
    if not os.path.exists(rna_emb_path) or not os.path.exists(rna_gip_path):
        print(f"[ERROR] Missing RNA files: {rna_emb_path} or {rna_gip_path}")
        sys.exit(1)

    D1 = read_matrix(dis_sem_path)
    D2 = read_matrix(dis_gip_path)
    D1, D2 = check_and_align(D1, D2, "d2d_do.csv", "GIP_Disease.csv")
    D_out = integrate_two(D1, D2)
    D_out.to_csv(out_dis_path, float_format="%.6f")
    print(f"[OK] Wrote {out_dis_path}")

    R1 = read_matrix(rna_emb_path)
    R2 = read_matrix(rna_gip_path)
    R1, R2 = check_and_align(R1, R2, "allR_cosine_similarity.csv", "GIP_RNA.csv")
    R_out = integrate_two(R1, R2)
    R_out.to_csv(out_rna_path, float_format="%.6f")
    print(f"[OK] Wrote {out_rna_path}")

if __name__ == "__main__":
    main()