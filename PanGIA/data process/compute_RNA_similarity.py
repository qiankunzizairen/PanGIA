# The sequence comparison algorithm in this script is implemented using 
# dynamic programming and can consume substantial CPU resources. 
# When running it in practice, 
# please ensure you have sufficient memory and CPU capacity, 
# or consider incorporating parallelization logic into your program to improve performance.

import pandas as pd
import numpy as np
from minineedle import needle
import time
import os

def compute_half_similarity_matrix(seq_dict):
    """
    Compute the upper triangle Needleman-Wunsch similarity matrix for the given sequence dictionary.
    """
    keys = list(seq_dict.keys())
    matrix = pd.DataFrame(index=keys, columns=keys)
    total = len(keys)

    start_time = time.time()
    for i, key_i in enumerate(keys):
        print(f"[{i + 1}/{total}] Processing {key_i}")
        seq_i = seq_dict[key_i]
        for j in range(i + 1, total):
            key_j = keys[j]
            seq_j = seq_dict[key_j]
            alignment = needle.NeedlemanWunsch(seq_j, seq_i)
            score = alignment.get_score()
            matrix.at[key_i, key_j] = score
    elapsed = time.time() - start_time
    print(f"Half matrix computed in {elapsed:.2f} seconds")
    return matrix

def normalize_and_complete_matrix(half_matrix):
    """
    Make the matrix symmetric, normalize it to [0, 1], and set the diagonal to 1.
    """
    values = half_matrix.values.astype(float)
    values[np.isnan(values)] = 0
    values = values + values.T

    max_val = values.max()
    min_val = values.min()
    if max_val != min_val:
        values = (values - min_val) / (max_val - min_val)
    else:
        values = np.zeros_like(values)

    np.fill_diagonal(values, 1)
    return pd.DataFrame(values, index=half_matrix.index, columns=half_matrix.columns)

def process_rna_type(rna_type):
    """
    Process one RNA type: compute and save both half and full similarity matrices.
    """
    input_file = f"{rna_type}_seq.csv"
    half_output_file = f"{rna_type}_half_p2p_needleman.csv"
    full_output_file = f"{rna_type}_p2p_needleman.csv"

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    # Load sequences
    seq_df = pd.read_csv(input_file)
    seq_dict = dict(seq_df.values)

    # Compute half matrix
    half_matrix = compute_half_similarity_matrix(seq_dict)
    half_matrix.to_csv(half_output_file)
    print(f"Saved half matrix to {half_output_file}")

    # Normalize and complete full matrix
    full_matrix = normalize_and_complete_matrix(half_matrix)
    full_matrix.to_csv(full_output_file)
    print(f"Saved full normalized matrix to {full_output_file}\n")

# Run for all RNA types
rna_types = ["miRNA", "piRNA", "lncRNA", "circRNA"]
for rna in rna_types:
    print(f"=== Processing {rna} ===")
    process_rna_type(rna)