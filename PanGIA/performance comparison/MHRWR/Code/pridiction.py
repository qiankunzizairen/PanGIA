import sys
import os
import collections
import numpy as np
from Walker import Walker
from sklearn.metrics import auc
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import heapq


    
#load disease/lncRNA name"
def read_name(filename):
    f=open(filename,'r')   
    name=[]                         
    for linename in f.readlines():
        name.append( linename.replace('\n', ''))     
    return name    


def prediction(disId,D,L,P,dlMat,DP,LP,dis_name,lnc_name,pro_name):
    
    restart_prob=0.7
    lenda_dl=0.1
    lenda_dp=0.1
    lenda_lp=0.1
    lenda_d=0.2
    lenda_l=0.8
    lenda_p=0.1
  
    dlMat[disId] = [0]*dlMat.shape[1]
    
    if np.sum(dlMat[disId]) ==0 :
        #dislist=Sim_dis(dlMat,D,disId)
        #for m in range (dlMat.shape[1]):
        dlMat[disId]=Sim_dis(dlMat,D,disId)
        
       
    for lncId in range (dlMat.shape[1]) :                   
        if np.sum(dlMat[:,lncId]) ==0 :
            lnclist=Sim_lnc(dlMat,L,lncId)
            for n in range (dlMat.shape[0]):
                dlMat[n][lncId]=lnclist[n]
    
    wk = Walker(lenda_dl,lenda_dp,lenda_lp,D,L,P,dlMat,DP,LP)
    ans=wk.run_exp(lenda_d,lenda_l,lenda_p,disId,D,dlMat,DP,restart_prob,dis_name,lnc_name,pro_name)    
    return ans

def Sim_lnc(A,L,k):
    A=A.T
    lnc_curr_sim = L[k]
    lnc_curr_sim=lnc_curr_sim.reshape(1,-1)
    inter_prof_vec = np.dot(lnc_curr_sim,A) 
    inter_prof_vec=inter_prof_vec.T
    return inter_prof_vec
    
def Sim_dis(A,D,disId):
    dis_curr_sim = D[disId]
    dis_curr_sim=dis_curr_sim.reshape(1,-1)
    inter_prof_vec = np.dot(dis_curr_sim , A)
    return inter_prof_vec

if __name__ == '__main__': 
    
    D=np.loadtxt("DisSemGauSim.txt")
    L=np.loadtxt("LncFunGauSim.txt")
    P=np.loadtxt("GeneLlsGauSim.txt")

    dlMat=np.loadtxt("DisLncMat.txt")
    PD=np.loadtxt("GeneDisMat.txt")
    DP=PD.T
    PL=np.loadtxt("GeneLncMat.txt") 
    LP=PL.T
    dis_name=read_name("disease_name.txt")
    lnc_name=read_name("lncRNA_name.txt")    
    pro_name=read_name("gene_name.txt")   
    for s in range (len(D)):
        seed=s
        pre=prediction(seed,D,L,P,dlMat,DP,LP,dis_name,lnc_name,pro_name) 
        
        
    from sklearn.metrics import roc_auc_score, average_precision_score
import heapq

# 保存预测结果和真实标签
y_true_all = []
y_score_all = []

# 遍历每个疾病节点，逐一预测
for s in range(len(D)):
    seed = s
    true_labels = dlMat[seed].copy()  # 真实标签（1表示关联）
    
    pre_scores = prediction(seed, D, L, P, dlMat.copy(), DP, LP, dis_name, lnc_name, pro_name)  # 得分列表
    
    y_true_all.extend(true_labels)
    y_score_all.extend(pre_scores)

    # 转为numpy数组
    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)

    # ✅ AUC
    auc_score = roc_auc_score(y_true_all, y_score_all)
    print("AUC:", auc_score)

    # ✅ AUPR
    aupr_score = average_precision_score(y_true_all, y_score_all)
    print("AUPR:", aupr_score)

    # ✅ Top-K Precision 示例（K=100）
    K = 100
    top_k_idx = np.argsort(y_score_all)[-K:]  # 得分最高的K个索引
    precision_at_k = np.sum(y_true_all[top_k_idx]) / K
    print(f"Top-{K} Precision:", precision_at_k)
