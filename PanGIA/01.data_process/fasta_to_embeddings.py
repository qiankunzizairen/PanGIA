# python fasta_to_embeddings.py your.fasta out
import sys, torch, numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def make_windows(L,k=6,max_tokens=510,stride=250):
    T=L-k+1
    if T<=max_tokens:return [(0,T)]
    r=[];s=0
    while s<T:
        e=min(s+max_tokens,T)
        r.append((s,e))
        if e==T:break
        s+=stride
    return r

def kmers_range(seq,k,st,ed):
    return ' '.join(seq[i:i+k] for i in range(st,ed) if 'N' not in seq[i:i+k])

def load_fasta(path):
    ids=[];seqs=[]
    for r in SeqIO.parse(path,"fasta"):
        ids.append(r.id)
        s=str(r.seq).upper().replace('U','T')
        seqs.append(''.join(ch for ch in s if ch in 'ACGTN'))
    return ids,seqs

def main(fa,outdir,model_name = "DNA_bert_6",k=6,max_tokens=510,stride=250,batch=128,fp16=True):
    import os
    os.makedirs(outdir,exist_ok=True)
    ids,seqs=load_fasta(fa)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(model_name,do_lower_case=False)
    model=AutoModel.from_pretrained(model_name).to(device)
    if device.type=="cuda" and fp16:model=model.half()
    model.eval()
    with torch.no_grad():
        hdim=model(**tok("AAAAAA",return_tensors="pt").to(device)).last_hidden_state.shape[-1]
    N=len(seqs)
    sums=np.zeros((N,hdim),dtype=np.float64)
    cnts=np.zeros((N,),dtype=np.int64)
    bt=[];own=[]
    def flush():
        nonlocal bt,own
        if not bt:return
        with torch.no_grad():
            enc=tok(bt,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
            if device.type=="cuda" and fp16:
                with torch.cuda.amp.autocast():
                    out=model(**enc).last_hidden_state[:,0,:]
            else:
                out=model(**enc).last_hidden_state[:,0,:]
            out=out.float().cpu().numpy()
        for v,o in zip(out,own):
            sums[o]+=v;cnts[o]+=1
        bt=[];own=[]
    for si,s in enumerate(tqdm(seqs,total=N)):
        L=len(s)
        if L<k:continue
        for st,ed in make_windows(L,k,max_tokens,stride):
            txt=kmers_range(s,k,st,ed)
            if not txt:continue
            bt.append(txt);own.append(si)
            if len(bt)>=batch:flush()
    flush()
    valid=cnts>0
    emb=np.zeros((N,hdim),dtype=np.float32)
    emb[valid]=(sums[valid]/cnts[valid,None]).astype(np.float32)
    np.save(f"{outdir}/embeddings.npy",emb)
    with open(f"{outdir}/ids.txt","w") as f:
        for x in ids:f.write(x+"\n")

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])