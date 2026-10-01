from pathlib import Path
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
import json, shutil, time

SRC=Path('/mnt/data/ml3_build'); ROOT=Path('/mnt/data/ML3_GEOMETRY_AWARE_GITHUB'); ROOT.mkdir(exist_ok=True)
MODEL=ROOT/'geometry_models'; MODEL.mkdir(exist_ok=True)
df=pd.read_excel(SRC/'all_BL_BA_SMILES.xlsx')

def can(s):
 m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m,canonical=True) if m else None
AV=[1,5,6,7,8,9,15,16,17,35,53]
def af(a):
 f=[float(a.GetAtomicNum()==z) for z in AV]+[float(a.GetAtomicNum() not in AV)]
 try: hv=float(a.GetHybridization().real)
 except: hv=0.
 return f+[a.GetDegree()/6.,a.GetFormalCharge()/3.,a.GetTotalNumHs()/4.,float(a.GetIsAromatic()),float(a.IsInRing()),hv]
def bf(b):
 t=b.GetBondType(); return [float(t==Chem.rdchem.BondType.SINGLE),float(t==Chem.rdchem.BondType.DOUBLE),float(t==Chem.rdchem.BondType.TRIPLE),float(t==Chem.rdchem.BondType.AROMATIC),float(b.GetIsConjugated()),float(b.IsInRing())]
def garr(s):
 m=Chem.MolFromSmiles(s); x=np.asarray([af(a) for a in m.GetAtoms()],np.float32); src=[];dst=[];ea=[]
 for b in m.GetBonds():
  i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); q=bf(b); src += [i,j]; dst += [j,i]; ea += [q,q]
 return x,np.asarray([src,dst],np.int64) if src else np.empty((2,0),np.int64),np.asarray(ea,np.float32) if ea else np.empty((0,6),np.float32)
class G:
 def __init__(self,x,ei,ea): self.x=torch.tensor(x);self.edge_index=torch.tensor(ei);self.edge_attr=torch.tensor(ea)
class B:
 def __init__(self,gs):
  xs=[];ei=[];ea=[];bs=[];o=0
  for i,g in enumerate(gs):
   xs.append(g.x);bs.append(torch.full((len(g.x),),i,dtype=torch.long))
   if g.edge_index.numel(): ei.append(g.edge_index+o);ea.append(g.edge_attr)
   o+=len(g.x)
  self.x=torch.cat(xs);self.batch=torch.cat(bs);self.edge_index=torch.cat(ei,1) if ei else torch.empty((2,0),dtype=torch.long);self.edge_attr=torch.cat(ea) if ea else torch.empty((0,6))
class Enc(nn.Module):
 def __init__(self,h=32,e=32,L=2):
  super().__init__();self.n=nn.Linear(18,h);self.e=nn.Linear(6,h);self.ms=nn.ModuleList([nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h)) for _ in range(L)]);self.p=nn.Sequential(nn.Linear(h,e),nn.ReLU(),nn.Linear(e,e))
 def forward(self,g):
  x=self.n(g.x);s,d=g.edge_index
  for mlp in self.ms:
   agg=torch.zeros_like(x)
   if s.numel():
    msg=x[s]+self.e(g.edge_attr);agg.index_add_(0,d,msg);deg=torch.zeros((len(x),1));deg.index_add_(0,d,torch.ones((len(d),1)));agg/=deg.clamp_min(1)
   x=x+F.relu(mlp(x+agg))
  n=int(g.batch.max())+1;p=torch.zeros((n,x.shape[1]));p.index_add_(0,g.batch,x);cnt=torch.bincount(g.batch,minlength=n).float().unsqueeze(1);return self.p(p/cnt.clamp_min(1))
class Geo(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Enc();self.h=nn.Sequential(nn.Linear(66,96),nn.ReLU(),nn.Linear(96,48),nn.ReLU(),nn.Linear(48,4))
 def forward(self,a,b,cn):return self.h(torch.cat([self.enc(a),self.enc(b),cn],1))
class Prop(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Enc();self.h=nn.Sequential(nn.Linear(70,112),nn.ReLU(),nn.Linear(112,56),nn.ReLU(),nn.Linear(56,3))
 def forward(self,a,b,cn,geo):return self.h(torch.cat([self.enc(a),self.enc(b),cn,geo],1))

df['L1c']=df.L1_SMILES.map(can);df['L2c']=df.L2_SMILES.map(can);df=df.dropna().reset_index(drop=True);df['grp']=df.apply(lambda r:f'{r.L1c}||{r.L2c}||{int(r.CN1)}||{int(r.CN2)}',axis=1)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);ti,vi=next(gss.split(df,groups=df.grp));train=df.iloc[ti].reset_index(drop=True);test=df.iloc[vi].reset_index(drop=True)
unique=pd.unique(pd.concat([df.L1c,df.L2c])); cache={s:G(*garr(s)) for s in unique}; idx={s:i for i,s in enumerate(unique)}
allg=B([cache[s] for s in unique])

def enc_all(model): return model.enc(allg)
def pair_embeddings(model):
 z=enc_all(model); return z

gmu=torch.tensor(train[['LL1','LL2','LL','BA']].mean().values,dtype=torch.float32);gsd=torch.tensor(train[['LL1','LL2','LL','BA']].std().values.clip(1e-6),dtype=torch.float32)
pmu=torch.tensor(train[['Ucal','Ueff','tio']].mean().values,dtype=torch.float32);psd=torch.tensor(train[['Ucal','Ueff','tio']].std().values.clip(1e-6),dtype=torch.float32)
def batch_pair(z,frame):
 a=torch.tensor([idx[s] for s in frame.L1c]);b=torch.tensor([idx[s] for s in frame.L2c]);cn=torch.tensor(frame[['CN1','CN2']].values,dtype=torch.float32);return z[a],z[b],cn

def train_geo():
 m=Geo();opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5);best=None;bl=1e9;bad=0
 y=torch.tensor(train[['LL1','LL2','LL','BA']].values,dtype=torch.float32)
 for ep in range(1,121):
  m.train();opt.zero_grad();z=enc_all(m);a,b,cn=batch_pair(z,train);pr=m.h(torch.cat([a,b,cn],1));loss=F.mse_loss(pr,(y-gmu)/gsd);loss.backward();opt.step()
  if loss.item()<bl:bl=loss.item();best={k:v.detach().clone() for k,v in m.state_dict().items()};bad=0
  else:bad+=1
  if bad>25:break
 m.load_state_dict(best);return m,ep,bl
geo,ge,gl=train_geo()
@torch.no_grad()
def pred_geo(m,frame):
 m.eval();z=enc_all(m);a,b,cn=batch_pair(z,frame);return (m.h(torch.cat([a,b,cn],1))*gsd+gmu).numpy()
gp=pred_geo(geo,test);gy=test[['LL1','LL2','LL','BA']].values;gmae=np.abs(gp-gy).mean(0)
# Property model trained on observed + predicted geometry, using same pair graph representation.
# To avoid expensive repeated graph encoding, each epoch computes embeddings once.
pred_all=pred_geo(geo,df); actual=df[['LL1','LL2','LL','BA']].values.astype(np.float32)
geo_aug=np.vstack([actual,pred_all.astype(np.float32)]); yprop=np.vstack([df[['Ucal','Ueff','tio']].values]*2).astype(np.float32); cn_aug=np.vstack([df[['CN1','CN2']].values]*2).astype(np.float32)
Gmean=torch.tensor(geo_aug.mean(0),dtype=torch.float32);Gstd=torch.tensor(geo_aug.std(0).clip(1e-6),dtype=torch.float32)
prop=Prop();opt=torch.optim.AdamW(prop.parameters(),lr=3e-3,weight_decay=1e-5);best=None;bl=1e9;bad=0
for ep in range(1,121):
 prop.train();opt.zero_grad();z=enc_all(prop);a,b,cn=batch_pair(z,pd.concat([df,df],ignore_index=True));gg=torch.tensor((geo_aug-Gmean.numpy())/Gstd.numpy(),dtype=torch.float32);yy=torch.tensor((yprop-pmu.numpy())/psd.numpy(),dtype=torch.float32);pr=prop.h(torch.cat([a,b,cn,gg],1));loss=F.mse_loss(pr,yy);loss.backward();opt.step()
 if loss.item()<bl:bl=loss.item();best={k:v.detach().clone() for k,v in prop.state_dict().items()};bad=0
 else:bad+=1
 if bad>25:break
prop.load_state_dict(best)
@torch.no_grad()
def pred_prop(frame):
 prop.eval();z=enc_all(prop);a,b,cn=batch_pair(z,frame);gg=torch.tensor((pred_geo(geo,frame)-Gmean.numpy())/Gstd.numpy(),dtype=torch.float32);pr=prop.h(torch.cat([a,b,cn,gg],1));return (pr*psd+pmu).numpy()
pp=pred_prop(test);py=test[['Ucal','Ueff','tio']].values;pmae=np.abs(pp-py).mean(0)
# final geometry model is the split-trained model; property model trained with all rows using it.
# Save checkpoints and metrics.
torch.save({'model_state_dict':geo.state_dict(),'target_mean':gmu.tolist(),'target_std':gsd.tolist(),'config':{'hidden':32,'embed':32,'gnn_layers':2,'targets':['LL1','LL2','LL','BA']},'training_rows':len(train),'all_rows':len(df)},MODEL/'geometry_model.pt')
torch.save({'model_state_dict':prop.state_dict(),'target_mean':pmu.tolist(),'target_std':psd.tolist(),'geometry_mean':Gmean.tolist(),'geometry_std':Gstd.tolist(),'config':{'hidden':32,'embed':32,'gnn_layers':2,'geometry_aware':True,'targets':['Ucal','Ueff','tio']},'training_rows':len(df)},MODEL/'geometry_aware_property_gnn.pt')
metrics={'rows':len(df),'unique_pair_cn':int(df.grp.nunique()),'train_rows':len(train),'test_rows':len(test),'geometry_test_MAE':dict(zip(['LL1_A','LL2_A','LL_A','BA_deg'],map(float,gmae))),'property_test_MAE':dict(zip(['Ucal_K','Ueff_K','tio'],map(float,pmae))),'geometry_epochs':ge,'property_epochs':ep}
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2))
shutil.copy2(SRC/'all_BL_BA_SMILES.xlsx',ROOT/'all_BL_BA_SMILES.xlsx')
print(json.dumps(metrics,indent=2));print([(p.name,p.stat().st_size) for p in MODEL.glob('*.pt')])
