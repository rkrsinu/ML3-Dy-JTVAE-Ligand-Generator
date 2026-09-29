#!/usr/bin/env python
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.optim as optim
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--data_dir',required=True); ap.add_argument('--jtvae_root',required=True); ap.add_argument('--init_model',required=True); ap.add_argument('--out_dir',required=True); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--batch_size',type=int,default=64); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--beta',type=float,default=0.02); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); root=Path(a.jtvae_root).resolve(); sys.path.insert(0,str(root)); from fast_jtnn import JTNNVAE,Vocab; from fast_jtnn.mol_tree import MolTree; from fast_jtnn.datautils_prop import get_tensors,set_batch_nodeID
 data=Path(a.data_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 vocab=Vocab([x.strip() for x in (data/'vocab.txt').read_text().splitlines() if x.strip()]); df=pd.read_csv(a.csv); lig=sorted(set([str(x) for x in df.L1.dropna()]+[str(x) for x in df.L2.dropna()])); lig=[Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True) for s in lig if Chem.MolFromSmiles(s)]; lig=sorted(set(lig));
 trees=[]
 for s in lig: t=MolTree(s); t.recover(); t.assemble(); trees.append(t)
 model=JTNNVAE(vocab,256,56,10,3).to(device); ck=torch.load(a.init_model,map_location=device,weights_only=False); model.load_state_dict(ck.get('model_state_dict',ck),strict=True); opt=optim.Adam(model.parameters(),lr=a.lr)
 best=1e99; hist=[]
 for ep in range(1,a.epochs+1):
  model.train(); order=list(range(len(trees))); random.shuffle(order); total=0; n=0
  for st in range(0,len(order),a.batch_size):
   batch=[trees[i] for i in order[st:st+a.batch_size]]; set_batch_nodeID(batch,vocab); jt,mpn,(jtmpn,bidx)=get_tensors(batch); x=(batch,jt,mpn,(jtmpn,bidx)); opt.zero_grad(set_to_none=True); loss,log=model(x,a.beta); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),50); opt.step(); total+=float(loss.item())*len(batch); n+=len(batch)
  l=total/n; hist.append({'epoch':ep,'train_loss':l,'word_acc':log.get('word_acc',0),'topo_acc':log.get('topo_acc',0),'assm_acc':log.get('assm_acc',0)}); print(f'Epoch {ep}: {l:.5f}');
  if l<best: best=l; torch.save(model.state_dict(),out/'best_model.pt')
 pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); torch.save(model.state_dict(),out/'final_model.pt'); json.dump({'fit_mode':'all_data_finetune','n_unique_ligands':len(lig),'epochs':a.epochs,'initial_checkpoint':a.init_model,'best_train_loss':best,'hidden_size':256,'latent_size':56,'depthT':10,'depthG':3},open(out/'config.json','w'),indent=2)
 model.load_state_dict(torch.load(out/'best_model.pt',map_location=device,weights_only=False)); model.eval(); rows=[]
 with torch.no_grad():
  for st in range(0,len(trees),a.batch_size):
   batch=trees[st:st+a.batch_size]; set_batch_nodeID(batch,vocab); jt,mpn,_=get_tensors(batch); z,_=model.encode_latent(jt,mpn); arr=z.cpu().numpy();
   for s,v in zip(lig[st:st+len(batch)],arr): rows.append({'smiles':s,**{f'z_{i+1:02d}':float(x) for i,x in enumerate(v)}})
 latent=pd.DataFrame(rows)
 # attach mean complex properties for each ligand
 rec={}
 for _,r in df.iterrows():
  for c in ['L1','L2']:
   m=Chem.MolFromSmiles(str(r[c]));
   if m: rec.setdefault(Chem.MolToSmiles(m,canonical=True),[]).append([float(r.Ucal),float(r.Ueff),float(r.tio)])
 vals=[]
 for s in latent.smiles:
  a0=np.asarray(rec.get(s,[]),float); vals.append(a0.mean(0) if len(a0) else [np.nan]*3)
 vals=np.asarray(vals); latent['Ucal']=vals[:,0]; latent['Ueff']=vals[:,1]; latent['tio']=vals[:,2]; latent.to_csv(out/'latent_all.csv',index=False); (out/'split_ligands.csv').write_text('smiles,split\n'+'\n'.join(f'{s},all_data' for s in lig)+'\n')
 print('DONE',out)
if __name__=='__main__': main()
