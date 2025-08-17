import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv


class WeightedHeteroEncoder(nn.Module):
    """
    Heterogeneous graph encoder using HeteroConv + GATv2Conv(edge_dim=1).
    - 将 disease 特征线性映射到与 RNA 相同的维度；
    - 对每个异质关系用 GATv2Conv（可接收 edge_weight 一维权重）；
    - 关系级别用 HeteroConv 聚合；
    - 多头输出再做一次线性映射到 hidden_dim。
    """

    def __init__(
        self,
        *,
        rna_in_dim: int,
        d_feat_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        metadata: tuple,
        aggr: str = "sum",
    ):
        super().__init__()
        # 先把 disease 节点特征投到 rna_in_dim
        self.lin_d = nn.Linear(d_feat_dim, rna_in_dim)

        # 为每个异质关系构建一个 GATv2Conv
        convs = {}
        for src, rel, dst in metadata[1]:
            # 两端节点都视作 rna_in_dim（disease 已经线性投影）
            convs[(src, rel, dst)] = GATv2Conv(
                in_channels=(rna_in_dim, rna_in_dim),
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=1,          # 关键：接收一维 edge_weight
                add_self_loops=False # 异构图里一般不再自动加自环
            )

        self.conv = HeteroConv(convs, aggr=aggr)

        # 合并多头到 hidden_dim
        self.lin_out = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_weight_dict: dict) -> dict:
        # 将 disease 特征映射到与 RNA 相同的维度
        x_dict = x_dict.copy()
        x_dict["disease"] = self.lin_d(x_dict["disease"])

        # 组装每条关系的 edge_attr（形状 [E, 1]）
        edge_attr_dict = {}
        for etype, eidx in edge_index_dict.items():
            if etype in edge_weight_dict:
                w = edge_weight_dict[etype]
                if w is not None:
                    edge_attr_dict[etype] = w.view(-1, 1)

        # 关系级别消息传递
        h_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)

        # 多头拼接后的线性投影
        for ntype in h_dict:
            h_dict[ntype] = self.lin_out(h_dict[ntype])

        return h_dict


class MMoEHANNet(nn.Module):
    """
    Multi-task MMoE-style model on top of a weighted heterogeneous encoder.
    - 编码器：WeightedHeteroEncoder (HeteroConv + GATv2Conv(edge_dim=1))
    - 顶层：MMoE 专家池 + task gate + cross-task attention + 任务头
    """

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

        # 1) 编码器（能用 edge_weight）
        self.encoder = WeightedHeteroEncoder(
            rna_in_dim=rna_in_dim,
            d_feat_dim=d_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            metadata=metadata,
            aggr="sum",
        )

        # 2) 融合维度：RNA 表示 + 全局 disease 上下文
        fusion_dim = hidden_dim * 2

        # 3) 专家池
        self.expert_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ) for _ in range(num_experts)
        ])
        self.num_experts = num_experts

        # 4) 任务 gate（对每个任务，基于其 RNA 表示给出 K 个专家权重）
        self.task_rels = ['mi-d', 'circ-d', 'lnc-d', 'pi-d']
        self.gates = nn.ModuleDict({
            rel: nn.Linear(hidden_dim, num_experts, bias=False)
            for rel in self.task_rels
        })

        # 5) 每个任务 head：融合专家输出与 cross-task 表示
        self.heads = nn.ModuleDict({
            rel: nn.Sequential(
                nn.Linear(expert_dim * 2, expert_dim),
                nn.ReLU()
            ) for rel in self.task_rels
        })

        # 6) 任务间 cross attention（把每个任务的“均值专家表示”互相做注意力）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=expert_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 7) disease 表示投到 expert_dim，用于与任务表示做打分
        self.dis_proj = nn.Linear(hidden_dim, expert_dim)

        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

    def forward(self, data, rna_indices: dict):
        """
        Args:
            data: PyG HeteroData（要求每条边关系有 edge_index，且如果有权重则存在 edge_weight）
            rna_indices: dict，形如 {'mi-d': LongTensor(idx), ...}，给出每个任务对应的 RNA 节点下标
        Returns:
            out: dict，{rel: [n_t, N_d] 的打分矩阵（Sigmoid 概率）}
        """
        # 取图结构
        x_dict = data.x_dict
        e_dict = data.edge_index_dict
        # 收集 edge_weight（若存在）
        ew_dict = {}
        for etype in data.edge_types:
            ew_dict[etype] = getattr(data[etype], "edge_weight", None)

        # 1) 编码
        h_dict = self.encoder(x_dict, e_dict, ew_dict)
        h_rna = h_dict['rna']       # [N_p, hidden_dim]
        h_dis = h_dict['disease']   # [N_d, hidden_dim]

        # 2) 全局 disease 上下文
        d_global = h_dis.mean(dim=0, keepdim=True).expand_as(h_rna)
        fusion = torch.cat([h_rna, d_global], dim=1)  # [N_p, 2*hidden_dim]

        # 3) 专家池前向：得到每个 RNA 的 K 个专家输出
        E = torch.stack([ex(fusion) for ex in self.expert_pool], dim=1)  # [N_p, K, expert_dim]

        # 4) 各任务 gate 聚合专家
        U_list, task_repr = [], {}
        for rel in self.task_rels:
            idxs = rna_indices[rel]                       # [n_t]
            h_t = h_rna[idxs]                             # [n_t, hidden_dim]
            scores = self.gates[rel](h_t)                 # [n_t, K]
            u_t = (E[idxs] * scores.unsqueeze(-1)).sum(dim=1)  # [n_t, expert_dim]
            task_repr[rel] = u_t
            U_list.append(u_t.mean(dim=0))                # [expert_dim]

        # 5) 任务间 cross-attention
        U_stack = torch.stack(U_list, dim=0)              # [T, expert_dim]
        U_prime, _ = self.cross_attn(
            U_stack.unsqueeze(0), U_stack.unsqueeze(0), U_stack.unsqueeze(0)
        )
        U_prime = U_prime.squeeze(0)                      # [T, expert_dim]

        # 6) 计算每个任务对所有疾病的打分
        out = {}
        d_proj = self.dis_proj(h_dis)                     # [N_d, expert_dim]
        for i, rel in enumerate(self.task_rels):
            u_t = task_repr[rel]                          # [n_t, expert_dim]
            cross = U_prime[i].unsqueeze(0).expand_as(u_t)     # [n_t, expert_dim]
            cat = torch.cat([u_t, cross], dim=1)               # [n_t, 2*expert_dim]
            p_rep = self.heads[rel](cat)                       # [n_t, expert_dim]
            out[rel] = torch.sigmoid(p_rep @ d_proj.t())       # [n_t, N_d]

        return out