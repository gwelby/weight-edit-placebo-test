import numpy as np, json, sys, os
LOGS=os.environ.get('LM_UNDERSTAND_LOGS', 'lm_understand-logs')
sys.path.insert(0, os.path.join(LOGS, 'exp8_math/code'))
import st_numpy as S

BASE=os.environ.get('BASE_CKPT', 'ckpts/base') + '/model.safetensors'
ED=os.path.join(LOGS, 'exp8_math/runs/p50c/edit')
man=json.load(open(ED+'/manifest.json')); z=np.load(ED+'/edit.npz')
rng=np.random.default_rng(20261103)
out={}
report=[]
for i,t in enumerate(man['tensors']):
    dt,b=S.load(BASE,[t['name']])[t['name']]
    idx=z[f't{i}_idx']; val=z[f't{i}_val']
    delta=S.bf16_to_f32(val.reshape(-1,1)).reshape(-1)-S.bf16_to_f32(b).reshape(-1)[idx]
    sd=np.float32(delta.std())
    n_cols=64
    W=S.bf16_to_f32(b)
    cols=rng.choice(W.shape[1],size=n_cols,replace=False)
    ridx=np.concatenate([np.arange(W.shape[0])*0+r for c in cols for r in [c]]) if False else np.tile(cols,W.shape[0])
    rows=np.repeat(np.arange(W.shape[0]),n_cols)
    flat_idx=(rows*W.shape[1]+ridx).astype(np.int64)
    noise=rng.normal(0,sd,size=flat_idx.shape).astype(np.float32)
    newvals=S.bf16_bits(W.reshape(-1)[flat_idx]+noise)
    out[f't{i}_idx']=flat_idx; out[f't{i}_val']=newvals
    report.append((t['name'],len(flat_idx),float(sd),int(man['tensors'][i]['changed_entries'])))
np.savez('control_edit.npz',**out)
for r in report: print(r)
