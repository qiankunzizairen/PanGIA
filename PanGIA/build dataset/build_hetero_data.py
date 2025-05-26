# build_hetero_data.py
# output : data_hetero.pt
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch.nn as nn

# ------------------------------
# DeepLncLoc feature extractor
# ------------------------------
class DeepLncLoc(nn.Module):
    def __init__(self, w2v_emb, dropout, merge_win_size, context_size_list, out_size):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(w2v_emb, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.merge_win = nn.AdaptiveAvgPool1d(merge_win_size)
        assert out_size % len(context_size_list) == 0
        filter_out_size = int(out_size / len(context_size_list))
        self.con_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=w2v_emb.shape[1],
                        out_channels=filter_out_size,
                        kernel_size=context_size_list[i],
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for i in range(len(context_size_list))
            ]
        )

    def forward(self, p_kmers_id):
        x = self.dropout(self.embedding(p_kmers_id))
        x = x.transpose(1, 2)
        x = self.merge_win(x)
        x = [conv(x).squeeze(dim=2) for conv in self.con_list]
        x = torch.cat(x, dim=1)
        return x

# ------------------------------
# Load and aggregate sequence embeddings for one RNA type
# ------------------------------
def load_and_aggregate(rtype,
                       data_dir="./data",
                       device="cpu",  
                       dropout=0.4,
                       merge_win_size=32,
                       context_size_list=[1, 3, 5],
                       dll_out_size=128 * 3):
    path = f"{data_dir}/gensim_feat_{rtype}_128.npy"
    feat = np.load(path, allow_pickle=True).item()

    kmers_emb = torch.FloatTensor(feat[f"{rtype}_kmers_emb"]).to(device)
    segs_id   = torch.LongTensor(feat[f"pad_{rtype}_id_seq"]).to(device)
    seg2seq   = torch.LongTensor(feat[f"{rtype}_seg2seq"]).to(device)

    model = DeepLncLoc(
        w2v_emb=kmers_emb,
        dropout=dropout,
        merge_win_size=merge_win_size,
        context_size_list=context_size_list,
        out_size=dll_out_size
    ).to(device)

    with torch.no_grad():
        seg_feats = model(segs_id)

    num_seq   = seg2seq.max().item() + 1
    feat_dim  = seg_feats.size(1)
    seq_feats = torch.zeros((num_seq, feat_dim), device=device)
    counts    = torch.zeros(num_seq, device=device)
    for idx, sid in enumerate(seg2seq):
        seq_feats[sid] += seg_feats[idx]
        counts[sid] += 1
    seq_feats /= counts.unsqueeze(1)
    return seq_feats.cpu()  # save-friendly

# ------------------------------
# Main build logic
# ------------------------------
def main():
    device = "cpu"
    data_dir = "./data"
    save_path = "./processed/data_hetero.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load RNA-disease associations and similarities
    adj_np   = pd.read_csv(os.path.join(data_dir, "RDA_Matrix.csv"), index_col=0).values
    p_sim_np = pd.read_csv(os.path.join(data_dir, "allR_needleman.csv"), index_col=0).values
    d_sim_np = pd.read_csv(os.path.join(data_dir, "d2d_do.csv"), index_col=0).values

    # Load 4 types of RNA features
    mi_feats   = load_and_aggregate("mi", data_dir)
    circ_feats = load_and_aggregate("circ", data_dir)
    lnc_feats  = load_and_aggregate("lnc", data_dir)
    pi_feats   = load_and_aggregate("pi", data_dir)
    p_feats = torch.cat([mi_feats, circ_feats, lnc_feats, pi_feats], dim=0)
    d_feats = torch.FloatTensor(d_sim_np)

    # Initialize HeteroData
    data = HeteroData()
    data['rna'].x = p_feats
    data['disease'].x = d_feats

    # Build edge index for each RNA type
    rna_sizes = [mi_feats.size(0), circ_feats.size(0), lnc_feats.size(0), pi_feats.size(0)]
    offsets = np.cumsum([0] + rna_sizes[:-1])
    for rtype, offset, size in zip(['mi', 'circ', 'lnc', 'pi'], offsets, rna_sizes):
        sub_adj = adj_np[offset:offset + size, :]
        src, dst = np.nonzero(sub_adj)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data['rna', f'{rtype}-d', 'disease'].edge_index = edge_index

    # Optional similarity edges
    src_rr, dst_rr = np.nonzero(p_sim_np)
    data['rna', 'rna', 'rna'].edge_index = torch.tensor([src_rr, dst_rr], dtype=torch.long)

    src_dd, dst_dd = np.nonzero(d_sim_np)
    data['disease', 'disease', 'disease'].edge_index = torch.tensor([src_dd, dst_dd], dtype=torch.long)

    # Save to disk
    torch.save(data, save_path)
    print(f"HeteroData saved to {save_path}")

if __name__ == "__main__":
    main()