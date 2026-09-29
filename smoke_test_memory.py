"""Fast deployment smoke test: imports, checkpoint loading and one latent encode/decode."""
from pathlib import Path
import json
import torch
import pandas as pd
from fast_jtnn import JTNNVAE, Vocab
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.datautils_prop import set_batch_nodeID
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from memory_engine import PropNet, PairGNN

BASE=Path(__file__).resolve().parent
cfg=json.loads((BASE/'true_jtvae_model/config.json').read_text())
vocab=Vocab([x.strip() for x in (BASE/'true_jtvae_vocab.txt').read_text().splitlines() if x.strip()])
jt=JTNNVAE(vocab,cfg['hidden_size'],cfg['latent_size'],cfg['depthT'],cfg['depthG'])
jt.load_state_dict(torch.load(BASE/'true_jtvae_model/best_model.pt',map_location='cpu',weights_only=False)); jt.eval()
prop=PropNet(); _prop_ck = torch.load(BASE/'latent_oracle/property_oracle.pt',map_location='cpu',weights_only=False); prop.load_state_dict(_prop_ck.get('model_state_dict', _prop_ck)); prop.eval()
ck=torch.load(BASE/'gnn_oracle/model.pt',map_location='cpu',weights_only=False); gc=ck.get('config',{})
gnn=PairGNN(gc.get('hidden',64),gc.get('embed',64),gc.get('gnn_layers',3)); gnn.load_state_dict(ck['model_state_dict']); gnn.eval()
smiles='CC1=CC=CC1'
t=[MolTree(smiles)]; set_batch_nodeID(t,vocab); j,_=JTNNEncoder.tensorize(t); mp=MPN.tensorize([smiles])
with torch.no_grad(): z,_=jt.encode_latent(j,mp); p=prop(z); a,b=torch.chunk(z,2,1); decoded=jt.decode(a,b,False)
print('JT-VAE latent:', tuple(z.shape)); print('Latent oracle:', tuple(p.shape)); print('Decoded:', decoded); print('GNN loaded:', type(gnn).__name__); print('SMOKE TEST PASSED')
