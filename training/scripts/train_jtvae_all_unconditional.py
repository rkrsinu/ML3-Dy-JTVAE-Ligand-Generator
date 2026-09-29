#!/usr/bin/env python
"""Final ML3 JT-VAE fit on ALL 474 unique ligands.

This is the generative model used by the deployment app.  It is the original
JTNNVAE architecture from JT-VAE-tmcinvdes-main, trained on every valid ligand
in the ML3 library.  No ligand holdout is used for the final deployment fit.

Properties are not fed into the JTNNVAE itself.  The 56-D latent representation
is paired with Ucal/Ueff/tio afterwards and a separate latent property oracle
performs target-directed latent optimization.  This keeps the generator and
property oracle modular and is the architecture used by the memory search app.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def canonical(s):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m,canonical=True) if m else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--jtvae_root', required=True)
    ap.add_argument('--out_dir', default='true_jtvae_model_all')
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--latent_size', type=int, default=56)
    ap.add_argument('--depthT', type=int, default=10)
    ap.add_argument('--depthG', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beta', type=float, default=0.006)
    ap.add_argument('--beta_max', type=float, default=0.10)
    ap.add_argument('--beta_step', type=float, default=0.002)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_dir).resolve(); root=Path(args.jtvae_root).resolve()
    sys.path.insert(0,str(root))
    from fast_jtnn import JTNNVAE, Vocab
    from fast_jtnn.mol_tree import MolTree
    from fast_jtnn.datautils_prop import get_tensors, set_batch_nodeID

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path=data/'vocab.txt'
    vocab=Vocab([x.strip() for x in vocab_path.read_text(encoding='utf-8').splitlines() if x.strip()])

    df=pd.read_csv(args.csv)
    required=['L1','L2','Ucal','Ueff','tio']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    lig=[]
    for c in ['L1','L2']:
        lig.extend(df[c].dropna().map(canonical).dropna().tolist())
    lig=sorted(set(lig))
    # Keep only ligands represented in the prepared all-data library.
    lib=pd.read_csv(data/'ligands'/'unique_ligands.csv')
    lib_col='canonical_smiles' if 'canonical_smiles' in lib.columns else 'smiles'
    libset=set(lib[lib_col].astype(str))
    lig=[s for s in lig if s in libset]
    if not lig: raise RuntimeError('No valid ligands found.')
    (out/'all_smiles.txt').write_text('\n'.join(lig)+'\n',encoding='utf-8')
    pd.DataFrame({'smiles':lig,'split':['all_data']*len(lig)}).to_csv(out/'split_ligands.csv',index=False)
    print('='*72); print('ML3 FINAL UNCONDITIONAL JT-VAE — ALL DATA FIT'); print('='*72)
    print(f'Ligands: {len(lig)} | vocab: {vocab.size()} | device: {device} | batch: {args.batch_size} | epochs: {args.epochs}')

    trees=[]
    for i,s in enumerate(lig,1):
        t=MolTree(s); t.recover(); t.assemble(); trees.append(t)
        if i%50==0 or i==len(lig): print(f'Prepared trees: {i}/{len(lig)}')

    model=JTNNVAE(vocab,args.hidden_size,args.latent_size,args.depthT,args.depthG).to(device)
    opt=optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf'); best_epoch=0; history=[]; beta=args.beta

    def run_epoch(train):
        model.train(train); order=list(range(len(trees)))
        if train: random.shuffle(order)
        total=0.; n=0; mets={}
        for st in range(0,len(order),args.batch_size):
            batch=[trees[i] for i in order[st:st+args.batch_size]]
            set_batch_nodeID(batch,vocab)
            jt,mpn,(jtmpn,bidx)=get_tensors(batch)
            x=(batch,jt,mpn,(jtmpn,bidx))
            if train: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                loss,log=model(x,beta)
                if train:
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),50.0); opt.step()
            bs=len(batch); total+=float(loss.item())*bs; n+=bs
            for k,v in log.items(): mets[k]=mets.get(k,0.)+float(v)*bs
        return total/max(n,1), {k:v/max(n,1) for k,v in mets.items()}

    t0=time.time()
    for ep in range(1,args.epochs+1):
        if ep>args.warmup_epochs: beta=min(args.beta_max,beta+args.beta_step)
        loss,met=run_epoch(True)
        rec={'epoch':ep,'train_loss':loss,'beta':beta,**{f'train_{k}':v for k,v in met.items()}}
        history.append(rec)
        print(f"Epoch {ep:03d} | loss={loss:.5f} | beta={beta:.4f} | word={met.get('word_acc',0):.1f}% | topo={met.get('topo_acc',0):.1f}% | assm={met.get('assm_acc',0):.1f}%")
        # Deployment fit: select the lowest full-data training loss observed.
        if loss < best:
            best=loss; best_epoch=ep; torch.save(model.state_dict(),out/'best_model.pt')
        if ep%10==0:
            torch.save(model.state_dict(),out/f'model_epoch_{ep}.pt')
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    torch.save(model.state_dict(),out/'final_model.pt')
    cfg=vars(args).copy(); cfg.update({'fit_mode':'all_data','n_unique_ligands':len(lig),'vocab_size':vocab.size(),'device':str(device)})
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'fit_mode':'all_data','n_unique_ligands':len(lig),'vocab_size':vocab.size(),'latent_dimension':args.latent_size,'best_training_loss':best,'best_epoch':best_epoch,'epochs_completed':len(history),'device':str(device),'elapsed_seconds':time.time()-t0},indent=2),encoding='utf-8')

    # Export deterministic 56-D mean latents for all ligands.
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=device,weights_only=False)); model.eval()
    rows=[]
    with torch.no_grad():
        for st in range(0,len(trees),args.batch_size):
            batch=trees[st:st+args.batch_size]; set_batch_nodeID(batch,vocab)
            jt,mpn,_=get_tensors(batch)
            z,_=model.encode_latent(jt,mpn)
            arr=z.detach().cpu().numpy()
            for s,v in zip(lig[st:st+len(batch)],arr): rows.append({'smiles':s,**{f'z_{i+1:02d}':float(x) for i,x in enumerate(v)}})
    latent=pd.DataFrame(rows)
    props=df[['L1','L2','Ucal','Ueff','tio']].copy()
    prop_map={}
    for _,r in props.iterrows():
        for c in ['L1','L2']:
            s=canonical(r[c]);
            if s: prop_map.setdefault(s,[]).append([float(r['Ucal']),float(r['Ueff']),float(r['tio'])])
    vals=[]
    for s in latent.smiles:
        a=np.asarray(prop_map.get(s,[]),float)
        if len(a): vals.append(a.mean(0))
        else: vals.append([np.nan,np.nan,np.nan])
    vals=np.asarray(vals)
    latent['Ucal']=vals[:,0]; latent['Ueff']=vals[:,1]; latent['tio']=vals[:,2]
    latent.to_csv(out/'latent_all.csv',index=False)
    print('Saved:',out/'best_model.pt'); print('Saved:',out/'latent_all.csv')

if __name__=='__main__': main()
