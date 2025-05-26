import torch
import numpy as np
import os

from torch_geometric.data import HeteroData

# ------------------------------
# 设置路径
# ------------------------------
load_path = './processed/data_hetero.pt'
save_path = './processed/data_hetero_.pt'

# 确保输出目录存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ------------------------------
# 加载原始图数据
# ------------------------------
data = torch.load(load_path)
print("✅ 原始图已加载")

# ------------------------------
# 自动推断每类 RNA 的节点数量
# ------------------------------
# 你也可以用 rna_names.npy 来直接判断每类数量，以下为自动推断版本
rna_edge_types = ['mi-d', 'circ-d', 'lnc-d', 'pi-d']
rna_sizes = []

for rtype in rna_edge_types:
    src_nodes = data['rna', rtype, 'disease'].edge_index[0]
    rna_sizes.append(src_nodes.max().item() + 1)  # 局部最大索引 + 1

# 计算 offset
offsets = np.cumsum([0] + rna_sizes[:-1])

print("RNA type sizes:", rna_sizes)
print("Offsets:", offsets)

# ------------------------------
# 修复每类 RNA→Disease 边的编号
# ------------------------------
for rtype, offset in zip(rna_edge_types, offsets):
    edge_index = data['rna', rtype, 'disease'].edge_index
    src = edge_index[0] + offset  # ✅ 关键修复：加 offset
    dst = edge_index[1]
    new_edge_index = torch.stack([src, dst], dim=0)
    data['rna', rtype, 'disease'].edge_index = new_edge_index
    print(f"✅ 修复了边类型: {rtype}, 加 offset: {offset}")

# ------------------------------
# 保存修复后的图数据
# ------------------------------
torch.save(data, save_path)
print(f"✅ 修复后的图已保存至: {save_path}")