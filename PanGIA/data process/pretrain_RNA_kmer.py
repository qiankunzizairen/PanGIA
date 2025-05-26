#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified preprocessing script for k-mer embeddings of four types of ncRNAs
(miRNA, circRNA, lncRNA, piRNA):

1. Reads sequences using the column name 'seq'; checks base length distribution.
2. Computes k-mer length distribution (95th percentile), plots histograms.
3. Uses a unified pad_len = 95th percentile of total k-mer counts across types.
4. For sequences longer than pad_len, performs sliding-window segmentation.
5. Applies padding, trains Word2Vec embeddings, and saves features as:
   gensim_feat_{type}_{VECTOR_SIZE}.npy

Output includes:
- k-mer embeddings
- padded ID sequences
- segment-to-original-sequence mapping
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

# --- Configuration ---
DATA_DIR    = "./data"
RNA_FILES   = {
    "mi":   "miR_seq.xlsx",
    "circ": "circR_seq.xlsx",
    "lnc":  "lncR_seq.xlsx",
    "pi":   "piR_seq.xlsx",
}
SEQ_COLUMN  = "seq"   # Column header name for RNA sequences
KMER_SIZE   = 3
VECTOR_SIZE = 128
WINDOW      = 10
EPOCHS      = 500
WORKERS     = 4
SLIDE_STEP  = None    # If None, defaults to pad_len // 2

# --- Helper Functions ---
def read_sequences(path):
    """Read sequences from Excel or CSV based on the SEQ_COLUMN header."""
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    if SEQ_COLUMN not in df.columns:
        raise ValueError(f"Missing '{SEQ_COLUMN}' column in {path}")
    return df[SEQ_COLUMN].dropna().astype(str).str.strip().tolist()

def kmer_tokenize(seqs, k):
    """Split sequences into overlapping k-mers."""
    return [[seq[i:i+k] for i in range(len(seq)-k+1)] for seq in seqs]

def sliding_window(seq, win, step):
    """Slide a window over the sequence to produce segments."""
    segs = [seq[i:i+win] for i in range(0, len(seq)-win+1, step)]
    if len(seq) > win and (len(seq)-win) % step != 0:
        segs.append(seq[-win:])
    return segs

def build_vocab(token_seqs):
    """Build a vocabulary from k-mers with PAD and EOS tokens."""
    km2id, id2km = {"<PAD>":0, "<EOS>":1}, ["<PAD>", "<EOS>"]
    for seq in token_seqs:
        for km in seq:
            if km not in km2id:
                km2id[km] = len(id2km)
                id2km.append(km)
    return km2id, id2km

def pad_or_sample_with_mapping(token_seqs, pad_len, km2id, slide_step):
    """Pad or segment sequences to fixed length and return index mapping."""
    all_segs = []
    seg2seq = []
    for seq_idx, seq in enumerate(token_seqs):
        if len(seq) <= pad_len:
            seg_lists = [seq]
        else:
            step = slide_step or (pad_len // 2)
            seg_lists = sliding_window(seq, pad_len, step)
        for s in seg_lists:
            ids = [km2id[k] for k in s] + [km2id["<EOS>"]]
            if len(ids) < pad_len:
                ids += [km2id["<PAD>"]] * (pad_len - len(ids))
            else:
                ids = ids[:pad_len]
            all_segs.append(ids)
            seg2seq.append(seq_idx)
    return np.array(all_segs, dtype=np.int64), np.array(seg2seq, dtype=np.int64)

# --- Main Pipeline ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Read & tokenize sequences, gather k-mer lengths
    kmer_seqs = {}
    all_counts = []
    for rna_type, filename in RNA_FILES.items():
        seqs = read_sequences(os.path.join(DATA_DIR, filename))
        nt_lens = [len(s) for s in seqs]
        print(f"[INFO] {rna_type}: nucleotide lengths → min={min(nt_lens)}, max={max(nt_lens)}")
        kseqs = kmer_tokenize(seqs, KMER_SIZE)
        kmer_seqs[rna_type] = kseqs
        all_counts += [len(kmer_list) for kmer_list in kseqs]

    # Step 2: Determine global pad length (95th percentile)
    pad_len = int(np.percentile(all_counts, 95))
    print(f"=> pad_len (95th percentile of k-mer counts): {pad_len}")

    # Step 3: Plot and save histograms for each RNA type
    for rna_type, kseqs in kmer_seqs.items():
        counts = [len(seq) for seq in kseqs]
        plt.figure(figsize=(6, 4))
        plt.hist(counts, bins=50)
        plt.title(f"{rna_type} k-mer length distribution")
        plt.xlabel("k-mer count")
        plt.ylabel("Number of sequences")
        out_png = os.path.join(DATA_DIR, f"{rna_type}_kmer_hist.png")
        plt.savefig(out_png)
        plt.close()
        print(f"[{rna_type}] #sequences: {len(counts)}, min={min(counts)}, max={max(counts)}, 95%={np.percentile(counts,95):.0f}")
        print(f"    Histogram saved to → {out_png}")

    # Step 4: Preprocess each type — vocab, pad/sample, train W2V, save
    for rna_type, kseqs in kmer_seqs.items():
        print(f"\n== Processing {rna_type} ==")
        km2id, id2km = build_vocab(kseqs)
        segs, mapping = pad_or_sample_with_mapping(kseqs, pad_len, km2id, SLIDE_STEP)
        print(f"  Input sequences: {len(kseqs)} → segments: {len(segs)}")

        # Train Word2Vec embeddings
        docs = [[id2km[i] for i in seg if i != km2id["<PAD>"]] for seg in segs]
        w2v = Word2Vec(docs, vector_size=VECTOR_SIZE, window=WINDOW,
                       min_count=1, sg=1, epochs=EPOCHS, workers=WORKERS)

        # Export embedding matrix
        emb = np.zeros((len(id2km), VECTOR_SIZE), dtype=np.float32)
        for idx, km in enumerate(id2km):
            if km in w2v.wv.key_to_index:
                emb[idx] = w2v.wv[km]

        # Save output
        output = {
            f"{rna_type}_kmers_emb": emb,
            f"pad_{rna_type}_id_seq": segs,
            f"{rna_type}_seg2seq":   mapping
        }
        out_path = os.path.join(DATA_DIR, f"gensim_feat_{rna_type}_{VECTOR_SIZE}.npy")
        np.save(out_path, output)
        print(f"  Saved preprocessed features → {out_path}")