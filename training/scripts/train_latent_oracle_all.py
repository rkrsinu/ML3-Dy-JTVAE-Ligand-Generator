#!/usr/bin/env python
"""Fit the ML3 latent-property oracle on 100% of the 474-ligand latent library."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
SEED=42
class PropNet(nn.Module):
    def __init__(self,d=56,h=256,out=3):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,out))
    def forward(self,x): return self.net(x)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--latent',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=500); ap.add_argument('--batch_size',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.latent); zcols=[c for c in df.columns if str(c).lower().startswith('z_')]; zcols=sorted(zcols,key=lambda c:int(str(c).split('_')[1]))[:56]
 targets=['Ucal','Ueff','tio']; df=df.dropna(subset=zcols+targets).reset_index(drop=True); x=df[zcols].to_numpy(np.float32); y=df[targets].to_numpy(np.float32)
 xm=x.mean(0); xs=x.std(0); xs[xs<1e-8]=1.; ym=y.mean(0); ys=y.std(0); ys[ys<1e-8]=1.; X=(x-xm)/xs; Y=(y-ym)/ys
 ds=TensorDataset(torch.tensor(X),torch.tensor(Y)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True); model=PropNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5); lossfn=nn.SmoothL1Loss(); hist=[]; best=1e9
 print('='*72); print('ML3 LATENT PROPERTY ORACLE — ALL DATA FIT'); print('='*72); print(f'Rows: {len(df)} | latent dim: 56 | device: {device}')
 for ep in range(1,args.epochs+1):
  model.train(); ls=[]
  for xb,yb in loader:
   xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); p=model(xb); loss=lossfn(p,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); ls.append(float(loss.item()))
  l=float(np.mean(ls)); hist.append({'epoch':ep,'train_loss':l})
  if ep==1 or ep%25==0: print(f'Epoch {ep:03d} | train={l:.6f}')
  if l<best: best=l; torch.save({'model_state_dict':model.state_dict(),'latent_dim':56,'targets':targets,'latent_mean':xm,'latent_std':xs,'target_mean':ym,'target_std':ys,'fit_mode':'all_data'},out/'property_oracle.pt')
 pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); pd.DataFrame({'property':targets,'mean':ym,'std':ys}).to_csv(out/'property_scaler.csv',index=False)
 json.dump({'architecture':'PropNet 56-256-256-3 SiLU','fit_mode':'all_data','n_rows':len(df),'best_train_loss':best,'epochs':args.epochs,'seed':SEED,'targets':targets},open(out/'config.json','w'),indent=2)
 ck=torch.load(out/'property_oracle.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model_state_dict']); model.eval();
 with torch.no_grad(): pred=model(torch.tensor(X).to(device)).cpu().numpy()*ys+ym
 outdf=df[['smiles',*targets]].copy();
 for i,t in enumerate(targets): outdf[t+'_pred']=pred[:,i]; outdf[t+'_error']=pred[:,i]-y[:,i]
 outdf.to_csv(out/'train_predictions.csv',index=False)
 print('Saved',out/'property_oracle.pt')
if __name__=='__main__': main()
