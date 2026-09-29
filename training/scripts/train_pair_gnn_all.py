#!/usr/bin/env python
"""Train the ML3 two-ligand graph oracle on all 1689 labelled complexes.

No train/validation/test split is used for the deployment fit.  The model is
used for screening candidate L1/L2 combinations after JT-VAE generation.
"""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'app_support'))
from gnn_architecture import PairGNN, smiles_graph, BatchGraph

TARGETS=['Ucal','Ueff','tio']
SEED=42

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def canonical(s):
 m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m,canonical=True) if m else None

class PairDS(Dataset):
 def __init__(self,df,cache,mean,std): self.df=df.reset_index(drop=True); self.cache=cache; self.mean=mean; self.std=std
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]
  y=(r[TARGETS].to_numpy(np.float32)-self.mean)/self.std
  return self.cache[r.L1],self.cache[r.L2],np.array([r.CN1,r.CN2],np.float32),y,i

def collate(batch):
 g1=BatchGraph([x[0] for x in batch]); g2=BatchGraph([x[1] for x in batch]); cn=torch.tensor(np.stack([x[2] for x in batch])); y=torch.tensor(np.stack([x[3] for x in batch])); idx=[x[4] for x in batch]; return g1,g2,cn,y,idx

def move(g,device):
 g.x=g.x.to(device); g.edge_index=g.edge_index.to(device); g.edge_attr=g.edge_attr.to(device); g.batch=g.batch.to(device); return g

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--batch_size',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=SEED); args=ap.parse_args(); seed_all(args.seed)
 out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 df=pd.read_csv(args.data)
 for c in ['CN1','CN2',*TARGETS]: df[c]=pd.to_numeric(df[c],errors='coerce')
 df['L1']=df['L1'].map(canonical); df['L2']=df['L2'].map(canonical); df=df.dropna(subset=['L1','L2','CN1','CN2',*TARGETS]).drop_duplicates(subset=['L1','L2','CN1','CN2','Ucal','Ueff','tio']).reset_index(drop=True)
 mean=df[TARGETS].mean().to_numpy(np.float32); std=df[TARGETS].std(ddof=1).to_numpy(np.float32); std[std<1e-8]=1.0
 cache={}; all_smiles=sorted(set(df.L1)|set(df.L2));
 print('='*72); print('ML3 FINAL PAIR-GNN — ALL DATA FIT'); print('='*72); print(f'Rows: {len(df)} | unique ligands: {len(all_smiles)} | device: {device}')
 for i,s in enumerate(all_smiles,1): cache[s]=smiles_graph(s); assert cache[s] is not None; 
 ds=PairDS(df,cache,mean,std); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
 model=PairGNN(64,64,3).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4); crit=nn.SmoothL1Loss(); hist=[]; best=1e9
 for ep in range(1,args.epochs+1):
  model.train(); losses=[]
  for g1,g2,cn,y,_ in loader:
   g1,g2=move(g1,device),move(g2,device); cn,y=cn.to(device),y.to(device); opt.zero_grad(set_to_none=True); p=model(g1,g2,cn); loss=crit(p,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.item()))
  l=float(np.mean(losses)); hist.append({'epoch':ep,'train_loss':l});
  if ep==1 or ep%10==0: print(f'Epoch {ep:03d} | train={l:.6f}')
  if l<best: best=l; torch.save({'model_state_dict':model.state_dict(),'target_mean':mean,'target_std':std,'targets':TARGETS,'config':{'hidden':64,'embed':64,'gnn_layers':3,'fit_mode':'all_data','n_rows':len(df),'n_unique_ligands':len(all_smiles),'seed':args.seed}},out/'model.pt')
 pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); json.dump({'fit_mode':'all_data','n_rows':len(df),'n_unique_ligands':len(all_smiles),'best_train_loss':best,'epochs':args.epochs,'targets':TARGETS},open(out/'config.json','w'),indent=2)
 # full-data descriptive predictions
 ck=torch.load(out/'model.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model_state_dict']); model.eval(); preds=np.zeros((len(ds),3),np.float32)
 eval_loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
 with torch.no_grad():
  for g1,g2,cn,y,idx in eval_loader:
   p=model(move(g1,device),move(g2,device),cn.to(device)).cpu().numpy()*std+mean
   preds[np.array(idx)]=p
 pred=df.copy()
 for i,t in enumerate(TARGETS): pred[t+'_pred']=preds[:,i]; pred[t+'_error']=preds[:,i]-df[t].to_numpy()
 pred.to_csv(out/'train_predictions.csv',index=False)
 print('Saved',out/'model.pt')

if __name__=='__main__': main()
