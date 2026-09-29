"""Command-line version of the memory-augmented ML3 search.

Example:
python run_iterative_search.py --target_kind Ueff --target 3000 --cn1 2 --cn2 2 --iterations 4
"""
from pathlib import Path
import argparse, json
import pandas as pd
import torch

from fast_jtnn import JTNNVAE, Vocab
from memory_engine import PropNet, PairGNN, SearchConfig, run_iterative_search

BASE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_kind', choices=['Ucal','Ueff','Tor'], required=True)
    ap.add_argument('--target', type=float, required=True)
    ap.add_argument('--cn1', type=float, required=True)
    ap.add_argument('--cn2', type=float, required=True)
    ap.add_argument('--iterations', type=int, default=4)
    ap.add_argument('--starts', type=int, default=12)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--decode_per_seed', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='runtime_memory')
    args = ap.parse_args()

    cfg = json.loads((BASE/'true_jtvae_model/config.json').read_text())
    vocab = Vocab([x.strip() for x in (BASE/'true_jtvae_vocab.txt').read_text().splitlines() if x.strip()])
    jt = JTNNVAE(vocab, cfg['hidden_size'], cfg['latent_size'], cfg['depthT'], cfg['depthG'])
    jt.load_state_dict(torch.load(BASE/'true_jtvae_model/best_model.pt', map_location='cpu', weights_only=False)); jt.eval()
    prop = PropNet(); 
    _prop_ck = torch.load(BASE/'latent_oracle/property_oracle.pt', map_location='cpu', weights_only=False)
    prop.load_state_dict(_prop_ck.get('model_state_dict', _prop_ck)); prop.eval()
    sc = pd.read_csv(BASE/'latent_oracle/property_scaler.csv')
    mu = torch.tensor(sc['mean'].values, dtype=torch.float32); sd = torch.tensor(sc['std'].values, dtype=torch.float32)
    ck = torch.load(BASE/'gnn_oracle/model.pt', map_location='cpu', weights_only=False)
    gc = ck.get('config', {})
    gnn = PairGNN(gc.get('hidden',64), gc.get('embed',64), gc.get('gnn_layers',3)); gnn.load_state_dict(ck['model_state_dict']); gnn.eval()
    gmu = torch.tensor(ck['target_mean'], dtype=torch.float32); gsd = torch.tensor(ck['target_std'], dtype=torch.float32)
    df = pd.read_csv(BASE/'ML3_Ucal_Ueff_tio_2.csv')
    known = set(df['L1'].astype(str).tolist()+df['L2'].astype(str).tolist())
    generated = pd.read_csv(BASE/'generated_candidates/generated_candidates.csv')
    seed_col = 'smiles' if 'smiles' in generated.columns else generated.columns[0]
    seeds = [str(x) for x in generated[seed_col].dropna().tolist()[:24]]
    models = (jt, vocab, prop, mu, sd, gnn, gmu, gsd, known, df)
    cfgs = SearchConfig(iterations=args.iterations, starts_per_iteration=args.starts, latent_steps=args.steps, decode_per_seed=args.decode_per_seed, seed=args.seed)
    archive, final = run_iterative_search(models, args.target_kind, args.target, args.cn1, args.cn2, cfgs, seeds)
    out = BASE/args.out; archive.save(out); final.to_csv(out/'final_pairs.csv', index=False)
    print(f'Saved: {out.resolve()}')
    print(final.head(20).to_string(index=False) if len(final) else 'No final pairs.')

if __name__ == '__main__': main()
