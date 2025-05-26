import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm

# ------------------------------
# 环境设定
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 读取数据
# ------------------------------
adj_df = pd.read_csv("./data/RDA_Matrix.csv", index_col=0)
adj_np = adj_df.values
p_sim_np = pd.read_csv("./data/allR_needleman.csv", index_col=0).values
d_sim_np = pd.read_csv("./data/d2d_do.csv", index_col=0).values

rna_names = adj_df.index.tolist()
dis_names = adj_df.columns.tolist()

num_p, num_d = adj_np.shape
n_pca_p_sim = 85

# PCA 降维
pca_p_sim = PCA(n_components=n_pca_p_sim)
pca_p_sim_feat = pca_p_sim.fit_transform(p_sim_np)

pca_d_sim = PCA()
pca_d_sim_feat = pca_d_sim.fit_transform(d_sim_np)

# 构建特征矩阵 [num_p, num_d, feat_dim]
feat_dim = n_pca_p_sim + num_d
feat_mat = np.zeros((num_p, num_d, feat_dim), dtype=np.float32)
for i in range(num_p):
    for j in range(num_d):
        feat_mat[i, j] = np.concatenate([pca_p_sim_feat[i], pca_d_sim_feat[j]])

# ------------------------------
# 模型定义
# ------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp_model(x_np, y_np, device, n_epochs=20, batch_size=2048):
    model = MLP(input_dim=x_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(x_np, dtype=torch.float32),
                            torch.tensor(y_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model.train()
    for epoch in range(n_epochs):
        batch_bar = tqdm(loader, desc=f"Epoch [{epoch+1}/{n_epochs}]")
        for xb, yb in batch_bar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
    return model

def predict_proba_mlp(model, x_np, device, batch_size=1024):
    model.eval()
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list)

# ------------------------------
# 主流程
# ------------------------------
os.makedirs("./classifier", exist_ok=True)
os.makedirs("./rn_csv", exist_ok=True)

for fold in range(5):
    print(f"\n🌀 fold {fold}")

    # 读取该折数据
    pos_train_ij = pd.read_csv(f"./fold_csv/fold_{fold}_pos_train.csv").values
    unlab_train_ij = pd.read_csv(f"./fold_csv/fold_{fold}_unlabelled_train.csv").values
    train_ij = np.vstack([pos_train_ij, unlab_train_ij])

    # 构造训练特征
    train_feat = feat_mat[train_ij[:, 0], train_ij[:, 1], :]

    # 10% 正样本作为 spy
    n_spy = int(len(pos_train_ij) * 0.1)
    spy_idx = np.random.choice(len(pos_train_ij), n_spy, replace=False)
    spy_ij = pos_train_ij[spy_idx]

    # 训练标签：把 spy 从邻接矩阵中去掉
    adj_hidden = adj_np.copy()
    adj_hidden[spy_ij[:, 0], spy_ij[:, 1]] = 0
    train_label = adj_hidden[train_ij[:, 0], train_ij[:, 1]]

    # 训练模型
    model = train_mlp_model(train_feat, train_label, device)
    torch.save(model.state_dict(), f"./classifier/mlp_f85_fold{fold}.pt")

    # 预测训练集分数
    train_prob = predict_proba_mlp(model, train_feat, device)
    prob_mat = np.zeros_like(adj_np, dtype=np.float32)
    prob_mat[train_ij[:, 0], train_ij[:, 1]] = train_prob

    # 设置阈值：spy 得分最低 5%
    spy_prob = prob_mat[spy_ij[:, 0], spy_ij[:, 1]]
    thresh = np.sort(spy_prob)[int(len(spy_prob) * 0.05)]

    # ✅ 只在 unlabelled_train_ij 中筛选
    unlab_prob = prob_mat[unlab_train_ij[:, 0], unlab_train_ij[:, 1]]
    rn_mask = unlab_prob < thresh
    rn_ij = unlab_train_ij[rn_mask]

    # ✅ 限制 RN 数量不超过正样本数量 * 2
    max_rn = len(pos_train_ij) * 2
    if len(rn_ij) > max_rn:
        idx = np.random.choice(len(rn_ij), max_rn, replace=False)
        rn_ij = rn_ij[idx]

    print(f"✅ fold {fold}: selected {len(rn_ij)} reliable negatives")

    # ✅ 保存为 CSV
    df_rn = pd.DataFrame(rn_ij, columns=["rna_idx", "disease_idx"])
    df_rn.to_csv(f"./rn_csv/fold_{fold}_rn.csv", index=False)

print("\n✅ 所有折的可靠负样本已保存到 ./rn_csv 目录")

