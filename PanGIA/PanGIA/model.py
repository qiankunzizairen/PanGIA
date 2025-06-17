import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv


class HANEncoder(nn.Module):
    def __init__(
        self,
        *,
        rna_in_dim: int,
        d_feat_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        metadata: tuple,
    ):
        super().__init__()
        self.lin_d = nn.Linear(d_feat_dim, rna_in_dim)
        self.han = HANConv(
            in_channels=rna_in_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.han_out_dim = hidden_dim
        self.lin = nn.Linear(self.han_out_dim, self.hidden_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        x_dict = x_dict.copy()
        x_dict['disease'] = self.lin_d(x_dict['disease'])
        h_dict = self.han(x_dict, edge_index_dict)
        return {
            'rna':     self.lin(h_dict['rna']),
            'disease': self.lin(h_dict['disease']),
        }


class MMoEHANNet(nn.Module):
    def __init__(
        self,
        *,
        metadata,
        rna_in_dim: int,
        d_feat_dim: int,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int,
        num_heads: int,
        num_tasks: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = HANEncoder(
            rna_in_dim = rna_in_dim,
            d_feat_dim = d_feat_dim,
            hidden_dim = hidden_dim,
            num_heads  = num_heads,
            dropout    = dropout,
            metadata   = metadata,
        )
        fusion_dim = hidden_dim * 2
        self.expert_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.gates = nn.ModuleDict({
            rel: nn.Linear(hidden_dim, num_experts, bias=False)
            for rel in ['mi-d','circ-d','lnc-d','pi-d']
        })
        self.heads = nn.ModuleDict({
            rel: nn.Sequential(
                nn.Linear(expert_dim * 2, expert_dim),
                nn.ReLU()
            ) for rel in ['mi-d','circ-d','lnc-d','pi-d']
        })
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = expert_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.dis_proj = nn.Linear(hidden_dim, expert_dim)
        self.num_tasks   = num_tasks
        self.hidden_dim  = hidden_dim
        self.expert_dim  = expert_dim

    def forward(self, data, rna_indices):
        x_dict = data.x_dict
        e_dict = data.edge_index_dict
        h_dict = self.encoder(x_dict, e_dict)
        h_rna = h_dict['rna']        # [N_p, hidden_dim]
        h_dis = h_dict['disease']    # [N_d, hidden_dim]
        d_global = h_dis.mean(dim=0, keepdim=True).expand_as(h_rna)
        fusion = torch.cat([h_rna, d_global], dim=1)  # [N_p, 2*hidden_dim]
        E = torch.stack([ex(fusion) for ex in self.expert_pool], dim=1)  # [N_p, K, expert_dim]

        U_list, task_repr = [], {}
        for rel in ['mi-d','circ-d','lnc-d','pi-d']:
            idxs   = rna_indices[rel]                          # <- changed here
            h_t    = h_rna[idxs]                               # [n_t, hidden_dim]
            scores = self.gates[rel](h_t)                      # [n_t, K]
            u_t    = (E[idxs] * scores.unsqueeze(-1)).sum(dim=1)  # [n_t, expert_dim]
            task_repr[rel] = u_t
            U_list.append(u_t.mean(dim=0))                     # [expert_dim]

        U_stack = torch.stack(U_list, dim=0)                   # [T, expert_dim]
        U_prime, _ = self.cross_attn(
            U_stack.unsqueeze(0), U_stack.unsqueeze(0), U_stack.unsqueeze(0)
        )
        U_prime = U_prime.squeeze(0)                           # [T, expert_dim]

        out = {}
        d_proj = self.dis_proj(h_dis)                          # [N_d, expert_dim]
        for i, rel in enumerate(['mi-d','circ-d','lnc-d','pi-d']):
            u_t   = task_repr[rel]                             # [n_t, expert_dim]
            cross = U_prime[i].unsqueeze(0).expand_as(u_t)     # [n_t, expert_dim]
            cat   = torch.cat([u_t, cross], dim=1)             # [n_t, 2*expert_dim]
            p_rep = self.heads[rel](cat)                       # [n_t, expert_dim]
            out[rel] = torch.sigmoid(p_rep @ d_proj.t())       # [n_t, N_d]
        return out