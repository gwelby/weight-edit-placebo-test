import json, numpy as np
from datasets import load_dataset
ds=load_dataset('HuggingFaceH4/MATH-500',split='test')
rng=np.random.default_rng(42)
by={}
for i,r in enumerate(ds): by.setdefault(r['level'],[]).append(i)
sel=[]
for lv in sorted(by): sel+=list(rng.choice(by[lv],size=10,replace=False))
sel=sorted(sel)
with open('subset_math500.jsonl','w') as f:
    for i in sel:
        r=ds[int(i)]
        f.write(json.dumps({'idx':int(i),'level':r['level'],'problem':r['problem'],'gold':r['answer']})+'\n')
print('subset:',len(sel),'levels:',{r['level'] for r in ds.select(sel)})
