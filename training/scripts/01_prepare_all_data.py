#!/usr/bin/env python
"""Prepare the complete ML3 dataset for final all-data JT-VAE training.

This is the only preparation step required before final training.
It does NOT create a train/validation/test split.

Inputs
------
--csv        ML3_Ucal_Ueff_tio_2.csv
--jtvae_root JT-VAE-tmcinvdes-main
--out_dir    jtvae_data

Outputs
-------
jtvae_data/
  complexes/all_complexes.csv
  ligands/unique_ligands.csv
  ligands/valid_ligands.csv
  ligands/invalid_ligands.csv
  vocab.txt
  preparation_summary.json
  all_ligand_properties.csv

The vocabulary is extracted from the actual upstream JT-VAE MolTree
implementation, so the training model and vocabulary are consistent.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PROPS=['Ucal','Ueff','tio']

def canonical(s):
    if pd.isna(s): return None
    s=str(s).strip()
    if not s or s.lower()=='nan': return None
    try:
        m=Chem.MolFromSmiles(s)
        if m is None: return None
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True)
    ap.add_argument('--jtvae_root',required=True)
    ap.add_argument('--out_dir',default='jtvae_data')
    args=ap.parse_args()
    csv=Path(args.csv).resolve(); root=Path(args.jtvae_root).resolve(); out=Path(args.out_dir).resolve()
    if not csv.exists(): raise FileNotFoundError(f'Dataset not found: {csv}')
    if not (root/'fast_jtnn').exists(): raise FileNotFoundError(f'JT-VAE source is missing fast_jtnn: {root}')
    sys.path.insert(0,str(root))
    try:
        from fast_jtnn.mol_tree import MolTree
    except Exception as e:
        raise RuntimeError(f'Could not import upstream JT-VAE MolTree from {root}: {e}') from e

    df=pd.read_csv(csv)
    required=['L1','L2','Ucal','Ueff','tio']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}. Available: {list(df.columns)}')
    clean=pd.DataFrame({c:df[c] for c in df.columns}).copy()
    clean['L1']=clean['L1'].map(canonical); clean['L2']=clean['L2'].map(canonical)
    for c in PROPS: clean[c]=pd.to_numeric(clean[c],errors='coerce')
    clean=clean.dropna(subset=['L1','L2']+PROPS).reset_index(drop=True)
    if clean.empty: raise RuntimeError('No valid complexes remain after cleaning.')

    complex_out=out/'complexes'; lig_out=out/'ligands'; complex_out.mkdir(parents=True,exist_ok=True); lig_out.mkdir(parents=True,exist_ok=True)
    clean.to_csv(complex_out/'all_complexes.csv',index=False)

    ligands=sorted(set(clean['L1']).union(clean['L2']))
    valid=[]; invalid=[]; vocab=set()
    for i,s in enumerate(ligands,1):
        try:
            tree=MolTree(s)
            if tree is None or tree.size()<=0: raise ValueError('empty MolTree')
            frags=[]
            for node in tree.nodes:
                ns=getattr(node,'smiles',None)
                if ns: vocab.add(str(ns)); frags.append(str(ns))
            valid.append({'smiles':s,'n_nodes':int(tree.size())})
        except Exception as e:
            invalid.append({'smiles':s,'reason':f'{type(e).__name__}: {e}'})
        if i%50==0 or i==len(ligands): print(f'Validated {i}/{len(ligands)} | valid={len(valid)} | invalid={len(invalid)}')

    vdf=pd.DataFrame(valid); idf=pd.DataFrame(invalid,columns=['smiles','reason'])
    vdf.to_csv(lig_out/'valid_ligands.csv',index=False); idf.to_csv(lig_out/'invalid_ligands.csv',index=False)
    # Keep only complexes whose two ligands are actually representable by upstream JT-VAE.
    valid_set=set(vdf['smiles'])
    filtered=clean[clean['L1'].isin(valid_set)&clean['L2'].isin(valid_set)].reset_index(drop=True)
    filtered.to_csv(complex_out/'all_complexes.csv',index=False)

    pd.DataFrame({'smiles':sorted(valid_set)}).to_csv(lig_out/'unique_ligands.csv',index=False)
    # Ligand-level mean properties, useful for diagnostics and future latent fitting.
    rows=[]
    for s in sorted(valid_set):
        vals=pd.concat([clean.loc[clean.L1==s,PROPS],clean.loc[clean.L2==s,PROPS]],ignore_index=True)
        vals=vals.mean(numeric_only=True)
        rows.append({'smiles':s,**{p:float(vals[p]) for p in PROPS}})
    pd.DataFrame(rows).to_csv(out/'all_ligand_properties.csv',index=False)
    with open(out/'vocab.txt','w',encoding='utf-8') as f:
        for token in sorted(vocab): f.write(token+'\n')
    summary={'input_csv':str(csv),'jtvae_root':str(root),'input_complexes':int(len(df)),'clean_complexes':int(len(clean)),'final_jtvae_compatible_complexes':int(len(filtered)),'unique_ligands':int(len(ligands)),'valid_ligands':int(len(valid_set)),'invalid_ligands':int(len(invalid)),'vocab_size':int(len(vocab)),'fit_mode':'all_data'}
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+'='*72); print('ALL-DATA JT-VAE PREPARATION COMPLETE'); print('='*72)
    for k,v in summary.items(): print(f'{k}: {v}')
    print(f'\nComplexes : {complex_out/"all_complexes.csv"}')
    print(f'Ligands   : {lig_out/"unique_ligands.csv"}')
    print(f'Vocabulary: {out/"vocab.txt"}')

if __name__=='__main__': main()
