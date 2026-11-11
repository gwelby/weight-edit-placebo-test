import os, sys, shutil
from safetensors.torch import save_file, load_file
import torch

SRC=os.environ['QWEN_SNAPSHOT']  # local snapshot of Qwen/Qwen3.5-0.8B, revision 2fc06364715b967f1860aea9cf38778875588b17
REPO=os.path.join(os.environ.get('LM_UNDERSTAND_LOGS', 'lm_understand-logs'), 'exp8_math/base_model/Qwen3.5-0.8B-text')
OUT=sys.argv[1]
sd=load_file(os.path.join(SRC,'model.safetensors-00001-of-00001.safetensors'))
keep={k:v for k,v in sd.items() if k.startswith('model.language_model.') or k=='lm_head.weight'}
print('kept',len(keep),'of',len(sd))
os.makedirs(OUT,exist_ok=True)
save_file(keep,os.path.join(OUT,'model.safetensors'),metadata={'format':'pt'})
for f in os.listdir(REPO):
    shutil.copyfile(os.path.join(REPO,f),os.path.join(OUT,f))
print('ckpt ->',OUT)
