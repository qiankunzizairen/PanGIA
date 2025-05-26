from minimda import MINIMDA
from torch import optim, nn
from tqdm import trange
from utils import k_matrix
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th


def train(data, args, per_epoch_eval=None):
    model = MINIMDA(args)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    loss_fn = nn.BCELoss()

    epochs = trange(args.epochs, desc='train')
    miRNA = data['ms']
    disease = data['ds']

    for epoch in epochs:
        model.train()
        optimizer.zero_grad()

        # 构建相似图（K近邻图）
        mm_matrix = k_matrix(miRNA, args.neighbor)
        dd_matrix = k_matrix(disease, args.neighbor)

        mm_nx = nx.from_numpy_array(mm_matrix)
        dd_nx = nx.from_numpy_array(dd_matrix)

        mm_graph = dgl.from_networkx(mm_nx)
        dd_graph = dgl.from_networkx(dd_nx)

        # 构建 miRNA-disease 图（异构图）
        md_copy = copy.deepcopy(data['train_md'])
        md_copy[:, 1] += args.miRNA_number  # 偏移 disease 索引

        md_graph = dgl.graph(
            (np.concatenate((md_copy[:, 0], md_copy[:, 1])),
             np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
            num_nodes=args.miRNA_number + args.disease_number
        )

        # 转为 Tensor
        miRNA_th = th.FloatTensor(miRNA)
        disease_th = th.FloatTensor(disease)
        train_samples_th = th.FloatTensor(data['train_samples'])

        # 模型前向
        train_score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['train_samples'])

        # 损失计算 + 反向传播
        train_loss = loss_fn(train_score.flatten(), train_samples_th[:, 2])
        train_loss.backward()
        optimizer.step()

        # 每个 epoch 的评估回调（训练集）
        if per_epoch_eval is not None:
            with th.no_grad():
                per_epoch_eval(train_score.flatten(), train_samples_th[:, 2], epoch)

    # 测试阶段
    model.eval()
    with th.no_grad():
        score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['unsamples'])
        score = score.cpu().numpy()

    # 打印预测结果（miRNA编号, 疾病编号, 预测得分）
    result = np.concatenate((
        data['unsamples'],  # [N, 2] -> miRNA, disease
        score.reshape(-1, 1)  # [N, 1]
    ), axis=1)
    print(result)

    return score