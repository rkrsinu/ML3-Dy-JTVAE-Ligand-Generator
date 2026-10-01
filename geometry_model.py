from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
ATOM_VOCAB=[1,5,6,7,8,9,15,16,17,35,53]
class Graph:
    def __init__(self,x,edge_index,edge_attr): self.x=x; self.edge_index=edge_index; self.edge_attr=edge_attr
class BatchGraph:
    def __init__(self,graphs):
        xs=[];eis=[];eas=[];bs=[];off=0
        for i,g in enumerate(graphs):
            xs.append(g.x);bs.append(torch.full((len(g.x),),i,dtype=torch.long))
            if g.edge_index.numel(): eis.append(g.edge_index+off);eas.append(g.edge_attr)
            off+=len(g.x)
        self.x=torch.cat(xs,0);self.batch=torch.cat(bs,0)
        self.edge_index=torch.cat(eis,1) if eis else torch.empty((2,0),dtype=torch.long)
        self.edge_attr=torch.cat(eas,0) if eas else torch.empty((0,6),dtype=torch.float32)
class GNNEncoder(nn.Module):
    def __init__(self,hidden=32,embed=32,layers=2):
        super().__init__();self.n=nn.Linear(18,hidden);self.e=nn.Linear(6,hidden)
        self.ms=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.p=nn.Sequential(nn.Linear(hidden,embed),nn.ReLU(),nn.Linear(embed,embed))
    def forward(self,g):
        x=self.n(g.x);s,d=g.edge_index
        for mlp in self.ms:
            agg=torch.zeros_like(x)
            if s.numel():
                msg=x[s]+self.e(g.edge_attr);agg.index_add_(0,d,msg)
                deg=torch.zeros((len(x),1));deg.index_add_(0,d,torch.ones((len(d),1)));agg/=deg.clamp_min(1.0)
            x=x+F.relu(mlp(x+agg))
        n=int(g.batch.max().item())+1;pooled=torch.zeros((n,x.shape[1]));pooled.index_add_(0,g.batch,x)
        cnt=torch.bincount(g.batch,minlength=n).float().unsqueeze(1)
        return self.p(pooled/cnt.clamp_min(1.0))
class GeometryGNN(nn.Module):
    def __init__(self,hidden=32,embed=32,layers=2):
        super().__init__();self.enc=GNNEncoder(hidden,embed,layers)
        self.h=nn.Sequential(nn.Linear(embed*2+2,96),nn.ReLU(),nn.Linear(96,48),nn.ReLU(),nn.Linear(48,4))
    def forward(self,g1,g2,cn): return self.h(torch.cat([self.enc(g1),self.enc(g2),cn],1))
class GeometryAwarePairGNN(nn.Module):
    def __init__(self,hidden=32,embed=32,layers=2):
        super().__init__();self.enc=GNNEncoder(hidden,embed,layers)
        self.h=nn.Sequential(nn.Linear(embed*2+2+4,112),nn.ReLU(),nn.Linear(112,56),nn.ReLU(),nn.Linear(56,3))
    def forward(self,g1,g2,cn,geom): return self.h(torch.cat([self.enc(g1),self.enc(g2),cn,geom],1))
def atom_features(a):
    f=[float(a.GetAtomicNum()==z) for z in ATOM_VOCAB]+[float(a.GetAtomicNum() not in ATOM_VOCAB)]
    try: hv=float(a.GetHybridization().real)
    except Exception: hv=0.0
    return f+[a.GetDegree()/6.0,a.GetFormalCharge()/3.0,a.GetTotalNumHs()/4.0,float(a.GetIsAromatic()),float(a.IsInRing()),hv]
def bond_features(b):
    bt=b.GetBondType();return [float(bt==Chem.rdchem.BondType.SINGLE),float(bt==Chem.rdchem.BondType.DOUBLE),float(bt==Chem.rdchem.BondType.TRIPLE),float(bt==Chem.rdchem.BondType.AROMATIC),float(b.GetIsConjugated()),float(b.IsInRing())]
def smiles_graph(smiles):
    m=Chem.MolFromSmiles(str(smiles))
    if m is None:return None
    x=torch.tensor([atom_features(a) for a in m.GetAtoms()],dtype=torch.float32);src=[];dst=[];ea=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();q=bond_features(b);src += [i,j];dst += [j,i];ea += [q,q]
    ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.empty((2,0),dtype=torch.long)
    e=torch.tensor(ea,dtype=torch.float32) if ea else torch.empty((0,6),dtype=torch.float32)
    return Graph(x,ei,e)
def batch_graph(smiles): return BatchGraph([smiles_graph(s) for s in smiles])
def load_geometry_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);c=ck.get('config',{});m=GeometryGNN(c.get('hidden',32),c.get('embed',32),c.get('gnn_layers',2));m.load_state_dict(ck['model_state_dict']);m.eval();return m,torch.tensor(ck['target_mean'],dtype=torch.float32),torch.tensor(ck['target_std'],dtype=torch.float32)
def load_property_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);c=ck.get('config',{});m=GeometryAwarePairGNN(c.get('hidden',32),c.get('embed',32),c.get('gnn_layers',2));m.load_state_dict(ck['model_state_dict']);m.eval();return m,{k:torch.tensor(ck[k],dtype=torch.float32) for k in ['target_mean','target_std','geometry_mean','geometry_std']}
