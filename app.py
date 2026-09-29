import json, math, random, sys
from pathlib import Path
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# -----------------------------------------------------------------------------
# ML3 JT-VAE TARGET-DIRECTED TWO-LIGAND GENERATOR
# Workflow:
# target -> JT-VAE latent search -> novel ligands -> two-ligand combinations
# -> pair GNN screening -> target-ranked ligand combinations
# -----------------------------------------------------------------------------

st.set_page_config(page_title="ML3 JT-VAE Target-Directed Ligand Generator", page_icon="🧪", layout="wide")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

DATA = BASE / "ML3_Ucal_Ueff_tio_2.csv"
JT_MODEL = BASE / "true_jtvae_model"
JT_VOCAB = BASE / "true_jtvae_vocab.txt"
LATENT = BASE / "latent_oracle"
PREGEN = BASE / "generated_candidates" / "generated_candidates.csv"
GNN_DIR = BASE / "gnn_oracle"
TAU_REF = 100.0

TARGETS = {"Ucal": "Ucal (K)", "Ueff": "Ueff (K)", "Tor": "T_or (K)"}

# ------------------------- latent property oracle ----------------------------
class PropNet(nn.Module):
    def __init__(self, d=56, h=256, out=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h), nn.SiLU(), nn.Linear(h,h), nn.SiLU(), nn.Linear(h,out))
    def forward(self,x): return self.net(x)

# ------------------------- deployment GNN -----------------------------------
# Pure-PyTorch implementation of the supplied two-ligand GNN idea.  It avoids
# torch-geometric so the Streamlit deployment has fewer dependencies.
class Graph:
    def __init__(self, x, edge_index, edge_attr):
        self.x=x; self.edge_index=edge_index; self.edge_attr=edge_attr

class BatchGraph:
    def __init__(self, graphs):
        xs=[]; eis=[]; eas=[]; bs=[]; off=0
        for i,g in enumerate(graphs):
            xs.append(g.x); bs.append(torch.full((len(g.x),), i, dtype=torch.long))
            if g.edge_index.numel():
                eis.append(g.edge_index + off); eas.append(g.edge_attr)
            off += len(g.x)
        self.x=torch.cat(xs,0); self.batch=torch.cat(bs,0)
        self.edge_index=torch.cat(eis,1) if eis else torch.empty((2,0),dtype=torch.long)
        self.edge_attr=torch.cat(eas,0) if eas else torch.empty((0,6),dtype=torch.float32)

class GNNEncoder(nn.Module):
    def __init__(self, hidden=64, embed=64, layers=3):
        super().__init__()
        self.n=nn.Linear(18,hidden); self.e=nn.Linear(6,hidden)
        self.ms=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.pr=nn.Sequential(nn.Linear(hidden,embed),nn.ReLU(),nn.Linear(embed,embed))
    def forward(self,g):
        x=self.n(g.x); s,d=g.edge_index
        for mlp in self.ms:
            agg=torch.zeros_like(x)
            if s.numel():
                msg=x[s]+self.e(g.edge_attr)
                agg.index_add_(0,d,msg)
                deg=torch.zeros((len(x),1),device=x.device)
                deg.index_add_(0,d,torch.ones((len(d),1),device=x.device))
                agg/=deg.clamp_min(1.0)
            x=x+F.relu(mlp(x+agg))
        n=int(g.batch.max().item())+1
        pooled=torch.zeros((n,x.shape[1]),device=x.device); pooled.index_add_(0,g.batch,x)
        cnt=torch.bincount(g.batch,minlength=n).float().unsqueeze(1).to(x.device); pooled/=cnt.clamp_min(1.0)
        return self.pr(pooled)

class PairGNN(nn.Module):
    def __init__(self, hidden=64, embed=64, layers=3):
        super().__init__(); self.enc=GNNEncoder(hidden,embed,layers)
        self.f=nn.Sequential(nn.Linear(embed*2+2,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,g1,g2,cn):
        return self.f(torch.cat([self.enc(g1),self.enc(g2),cn],1))


def atom_features(a):
    vocab=[1,5,6,7,8,9,15,16,17,35,53]
    f=[float(a.GetAtomicNum()==z) for z in vocab]+[float(a.GetAtomicNum() not in vocab)]
    try: hv=float(a.GetHybridization().real)
    except Exception: hv=0.0
    f += [a.GetDegree()/6.0,a.GetFormalCharge()/3.0,a.GetTotalNumHs()/4.0,float(a.GetIsAromatic()),float(a.IsInRing()),hv]
    return f

def bond_features(b):
    bt=b.GetBondType()
    return [float(bt==Chem.rdchem.BondType.SINGLE),float(bt==Chem.rdchem.BondType.DOUBLE),float(bt==Chem.rdchem.BondType.TRIPLE),float(bt==Chem.rdchem.BondType.AROMATIC),float(b.GetIsConjugated()),float(b.IsInRing())]

def smiles_graph(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return None
    x=torch.tensor([atom_features(a) for a in m.GetAtoms()],dtype=torch.float32)
    src=[]; dst=[]; ea=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); bf=bond_features(b)
        src += [i,j]; dst += [j,i]; ea += [bf,bf]
    ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.empty((2,0),dtype=torch.long)
    e=torch.tensor(ea,dtype=torch.float32) if ea else torch.empty((0,6),dtype=torch.float32)
    return Graph(x,ei,e)

# ------------------------------ loading -------------------------------------
@st.cache_resource
def load_all():
    from fast_jtnn import JTNNVAE, Vocab
    cfg=json.load(open(JT_MODEL/"config.json"))
    vocab=Vocab([x.strip() for x in open(JT_VOCAB) if x.strip()])
    jt=JTNNVAE(vocab,cfg["hidden_size"],cfg["latent_size"],cfg["depthT"],cfg["depthG"])
    jt.load_state_dict(torch.load(JT_MODEL/"best_model.pt",map_location="cpu",weights_only=False)); jt.eval()
    prop=PropNet(); prop.load_state_dict(torch.load(LATENT/"property_oracle.pt",map_location="cpu",weights_only=False)); prop.eval()
    sc=pd.read_csv(LATENT/"property_scaler.csv")
    mu=torch.tensor(sc["mean"].values,dtype=torch.float32); sd=torch.tensor(sc["std"].values,dtype=torch.float32)
    ck=torch.load(GNN_DIR/"model.pt",map_location="cpu",weights_only=False)
    gc=ck.get("config",{}); gnn=PairGNN(gc.get("hidden",64),gc.get("embed",64),gc.get("gnn_layers",3))
    gnn.load_state_dict(ck["model_state_dict"]); gnn.eval()
    gmu=torch.tensor(ck["target_mean"],dtype=torch.float32); gsd=torch.tensor(ck["target_std"],dtype=torch.float32)
    df=pd.read_csv(DATA)
    known=set()
    for s in pd.concat([df.L1,df.L2]).dropna():
        m=Chem.MolFromSmiles(str(s));
        if m: known.add(Chem.MolToSmiles(m,canonical=True))
    return jt,prop,mu,sd,gnn,gmu,gsd,df,known

# ------------------------------ helpers -------------------------------------
def tor_from(ueff,tio):
    den=(math.log10(TAU_REF)-tio)*math.log(10.0)
    return float(ueff/den) if den>0 else float("nan")

def canonical(s):
    m=Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m,canonical=True) if m else None

def target_value(ucal,ueff,tio,kind):
    return float(ucal if kind=="Ucal" else ueff if kind=="Ueff" else tor_from(ueff,tio))

def generate_novel(target_kind,target_value,n_starts,steps,seed,jt,prop,mu,sd,known):
    random.seed(seed); torch.manual_seed(seed)
    # Unselected properties are kept near the training mean while the selected
    # property is driven toward the user's requested value.
    if target_kind=="Ucal":
        idx=0
        tz=(torch.tensor(target_value)-mu[0])/sd[0]
    elif target_kind=="Ueff":
        idx=1
        tz=(torch.tensor(target_value)-mu[1])/sd[1]
    else: idx=None
    out={}
    for _ in range(n_starts):
        z=(torch.randn(1,56)*1.1).requires_grad_(True)
        opt=torch.optim.Adam([z],lr=0.08)
        for _ in range(steps):
            opt.zero_grad(); p=prop(z)
            if idx is not None:
                loss=(p[0,idx]-tz)**2 + 0.03*(p[0,[j for j in range(3) if j!=idx]]**2).mean()
            else:
                raw=p*sd+mu; u=raw[0,1]; tio=raw[0,2]
                den=torch.clamp((math.log10(TAU_REF)-tio)*math.log(10.0), min=0.25); tor=u/den
                scale=max(abs(target_value),100.0)
                loss=((tor-target_value)/scale)**2 + 0.02*(p[0,0]**2).mean()
            loss=loss+1e-4*(z*z).mean(); loss.backward(); opt.step()
        try:
            with torch.no_grad(): zt,zm=torch.chunk(z,2,1); s=jt.decode(zt,zm,False)
        except Exception: s=None
        s=canonical(s) if s else None
        if not s or s in known or s in out: continue
        with torch.no_grad(): raw=(prop(z)[0]*sd+mu).numpy()
        val=target_value if target_kind else None
        out[s]={"smiles":s,"source":"JT-VAE target search","Ucal_latent":float(raw[0]),"Ueff_latent":float(raw[1]),"tio_latent":float(raw[2]),"target_distance":abs(target_value-target_value_from_raw(raw,target_kind))}
    return pd.DataFrame(out.values())

def target_value_from_raw(raw,kind): return target_value(raw[0],raw[1],raw[2],kind)

def pair_predictions(candidates,cn1,cn2,gnn,gmu,gsd,max_pairs=30000):
    smiles=list(candidates)
    pairs=list(combinations_with_replacement(smiles,2)) if cn1==cn2 else list(product(smiles,smiles))
    if len(pairs)>max_pairs:
        random.Random(42).shuffle(pairs); pairs=pairs[:max_pairs]
    rows=[]
    for st in range(0,len(pairs),256):
        pp=pairs[st:st+256]
        g1=BatchGraph([smiles_graph(a) for a,b in pp]); g2=BatchGraph([smiles_graph(b) for a,b in pp]); cn=torch.tensor([[cn1,cn2]]*len(pp),dtype=torch.float32)
        with torch.no_grad(): pred=gnn(g1,g2,cn)*gsd+gmu
        for (a,b),p in zip(pp,pred.numpy()):
            rows.append({"Ligand 1":a,"Ligand 2":b,"CN1":cn1,"CN2":cn2,"Predicted Ucal (K)":p[0],"Predicted Ueff (K)":p[1],"Predicted log10(tau0)":p[2],"Predicted Tor (K)":tor_from(p[1],p[2])})
    return pd.DataFrame(rows)

# -------------------------------- UI -----------------------------------------
st.title("🧪 ML3 JT-VAE Target-Directed Ligand Combination Generator")
st.caption("JT-VAE latent generation → novel ligands → two-ligand GNN screening → target-ranked combinations")

try:
    jt,prop,mu,sd,gnn,gmu,gsd,df,known=load_all()
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}: {e}")
    st.stop()

with st.sidebar:
    st.header("Generation target")
    kind=st.selectbox("Target property",["Ueff","Ucal","Tor"],index=0)
    target=st.number_input(TARGETS[kind],min_value=0.1,value=3000.0 if kind=="Ueff" else 2000.0 if kind=="Ucal" else 100.0,step=10.0)
    st.header("Complex")
    cn1=st.number_input("CN1",min_value=1,max_value=6,value=2,step=1)
    cn2=st.number_input("CN2",min_value=1,max_value=6,value=2,step=1)
    st.header("Search")
    n_starts=st.slider("JT-VAE latent starts",5,40,12)
    steps=st.slider("Latent optimization steps",5,30,12)
    extra=st.slider("Additional novel ligands to keep",5,50,20)
    topn=st.slider("Final ligand combinations",5,100,20)
    run=st.button("🚀 Generate ligand combinations",type="primary")

# Dataset range / extrapolation indicator
obs_kind=None
if kind=="Ucal": obs_kind=df.Ucal
elif kind=="Ueff": obs_kind=df.Ueff
else: obs_kind=df.apply(lambda r:tor_from(r.Ueff,r.tio),axis=1)
lo=float(np.nanmin(obs_kind)); hi=float(np.nanmax(obs_kind))
if target<lo or target>hi:
    st.warning(f"Extrapolation request: dataset range for {kind} is {lo:.1f}–{hi:.1f} K; requested target is {target:.1f} K. Results are model extrapolations and require experimental/quantum-chemical validation.")
else:
    st.info(f"Requested target is within the observed dataset range for {kind}: {lo:.1f}–{hi:.1f} K.")

if run:
    with st.spinner("Generating novel ligands in JT-VAE latent space..."):
        newdf=generate_novel(kind,float(target),n_starts,steps,42,jt,prop,mu,sd,known)
    # Existing pre-generated JT-VAE novel library is retained as an additional seed pool.
    seed_smiles=[]
    if PREGEN.exists():
        try:
            pg=pd.read_csv(PREGEN)
            if "smiles" in pg: seed_smiles=[canonical(s) for s in pg.smiles.dropna()]
        except Exception: pass
    seed_smiles=[s for s in seed_smiles if s]
    new_smiles=(newdf.sort_values("target_distance").head(extra).smiles.tolist() if len(newdf) else [])
    # Add target-relevant original ligands, but always retain generated ligands.
    tmp=df.copy(); tmp["target_value"]=tmp.apply(lambda r:target_value(r.Ucal,r.Ueff,r.tio,kind),axis=1); tmp["dist"]=(tmp.target_value-target).abs()
    originals=[]
    for _,r in tmp.sort_values("dist").head(25).iterrows():
        originals += [canonical(r.L1),canonical(r.L2)]
    pool=[]
    for s in originals+seed_smiles+new_smiles:
        if s and s not in pool: pool.append(s)
    st.session_state["pool"]=pool
    st.session_state["newdf"]=newdf
    st.session_state["target"]=float(target); st.session_state["kind"]=kind
    st.session_state["cn1"]=cn1; st.session_state["cn2"]=cn2

if "pool" in st.session_state:
    pool=st.session_state["pool"]
    st.subheader("1. Generated / selected ligand library")
    c1,c2,c3=st.columns(3); c1.metric("Unique ligands",len(pool)); c2.metric("New JT-VAE ligands",len(st.session_state.get("newdf",[]))); c3.metric("Pair combinations screened",f"up to {min(30000, len(pool)*(len(pool)+1)//2):,}" if st.session_state["cn1"]==st.session_state["cn2"] else f"up to {min(30000,len(pool)**2):,}")
    with st.spinner("Screening ligand combinations with the pair GNN..."):
        res=pair_predictions(pool,st.session_state["cn1"],st.session_state["cn2"],gnn,gmu,gsd)
    k=st.session_state["kind"]; t=st.session_state["target"]
    col={"Ueff":"Predicted Ueff (K)","Ucal":"Predicted Ucal (K)","Tor":"Predicted Tor (K)"}[k]
    res["Target error"]=(res[col]-t).abs()
    res=res.sort_values("Target error").head(topn).reset_index(drop=True); res.insert(0,"Rank",np.arange(1,len(res)+1))
    st.subheader("2. Target-ranked ligand combinations")
    st.dataframe(res,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download ligand combinations CSV",res.to_csv(index=False),file_name="ML3_target_ligand_combinations.csv",mime="text/csv")
    st.success("The outcome is the ligand combination itself: L1 + L2, together with the GNN-predicted magnetic properties. No ExtraTrees model is used in this app.")

with st.expander("How extrapolation works"):
    st.markdown("""
**Target → latent search → ligand generation → pair screening.**

1. The trained JT-VAE provides a 56-dimensional latent representation.
2. Random latent points are optimized against the requested target using the learned latent property oracle.
3. The optimized latent vectors are decoded by the actual JT-VAE decoder into chemically valid ligand SMILES.
4. Novel ligands are combined with other generated and target-relevant ligands.
5. The two-ligand GNN predicts Ucal, Ueff and log10(tau0) for each combination.
6. For a T_or target, T_or is calculated from the predicted Ueff and log10(tau0) using the same tau_ref = 100 s convention used in the project.
7. Combinations are ranked by absolute error from the user's requested target.

If the target lies outside the training/data range, the app still searches the learned latent space; that is **model extrapolation**, not evidence that the requested property has been experimentally achieved.
""")
