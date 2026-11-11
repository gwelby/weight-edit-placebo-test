import os, numpy as np, sys, json
sys.path.insert(0, os.path.join(os.environ.get('LM_UNDERSTAND_LOGS', 'lm_understand-logs'), 'exp8_math/code'))
import st_numpy as S
BASE=os.environ.get('BASE_CKPT', 'ckpts/base') + '/model.safetensors'
z=np.load('control_edit.npz')
names=['model.language_model.layers.15.mlp.down_proj.weight','model.language_model.layers.19.mlp.down_proj.weight']
tensors=S.load(BASE)
for i,n in enumerate(names):
    dt,b=tensors[n]
    e=b.copy().reshape(-1); e[z[f't{i}_idx']]=z[f't{i}_val']
    tensors[n]=(dt,e.reshape(b.shape))
    print(n,'control applied',len(z[f't{i}_idx']))
S.save('control.safetensors',tensors,{'format':'pt'})
