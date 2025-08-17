import sys, numpy as np, torch, torch.nn.functional as F

def main(emb_path, ids_path, out_csv):
    ids=[l.strip() for l in open(ids_path)]
    emb=np.load(emb_path)
    N,D=emb.shape
    assert N==len(ids)
    device=torch.device("cuda")
    E=torch.from_numpy(emb).to(device)
    E=F.normalize(E,p=2,dim=1)
    S=(E@E.T).detach().cpu().numpy()
    with open(out_csv,"w") as f:
        f.write("," + ",".join(ids) + "\n")
        for i in range(N):
            f.write(ids[i] + "," + ",".join(f"{x:.6f}" for x in S[i]) + "\n")

if __name__=="__main__":
    if len(sys.argv)<4:
        print(f"usage: python {sys.argv[0]} embeddings.npy ids.txt similarity.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])