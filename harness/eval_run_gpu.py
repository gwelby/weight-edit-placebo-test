import torch, json, time, sys, os, argparse
sys.path.insert(0, os.path.join(os.environ.get('LM_UNDERSTAND_LOGS', 'lm_understand-logs'), 'exp8_math/code'))  # authors' grader (mathqa.py)
import mathqa as Q
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import torch as T

T.set_num_threads(int(os.environ.get('NT','28')))

class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty on generated (output) tokens: scores[t] -= p for any t already generated."""
    def __init__(self, p, plen): self.p=p; self.plen=plen
    def __call__(self, ids, scores):
        gen=ids[:,self.plen:]
        if gen.shape[1]:
            for b in range(gen.shape[0]):
                scores[b,gen[b].unique()]-=self.p
        return scores

ap=argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('subset'); ap.add_argument('out')
ap.add_argument('--samples',type=int,default=4); ap.add_argument('--batch',type=int,default=8)
ap.add_argument('--max_tokens',type=int,default=32768); ap.add_argument('--dtype',default='bfloat16'); a=ap.parse_args()

tok=AutoTokenizer.from_pretrained(a.ckpt)
tok.padding_side='left'
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(a.ckpt,dtype=getattr(T,a.dtype),attn_implementation='sdpa')
model.cuda().eval()

probs=[json.loads(l) for l in open(a.subset)]
out=open(a.out,'a')
B=a.batch; PP=1.5
SAMP=dict(do_sample=True,temperature=0.6,top_p=0.95,top_k=20,max_new_tokens=a.max_tokens)

t0=time.time(); ntok=0
for s in range(a.samples):
    for i in range(0,len(probs),B):
        chunk=probs[i:i+B]
        msgs=[Q.build_prompt(p['problem']) for p in chunk]
        texts=[tok.apply_chat_template([{'role':'user','content':m}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for m in msgs]
        enc=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        plen=enc['input_ids'].shape[1]
        T.manual_seed(1000*s)
        with T.no_grad():
            g=model.generate(**enc,logits_processor=[PresencePenalty(PP,plen)],**SAMP)
        for j,p in enumerate(chunk):
            seq=g[j][plen:]
            n_new=len(seq)
            txt=tok.decode(seq,skip_special_tokens=True)
            cut=n_new>=a.max_tokens
            rec={'idx':p['idx'],'sample':s,'n_tokens':n_new,'cut':cut,'correct':Q.is_correct(Q.extract(txt)[0],p['gold']),'gold':p['gold'],'text_tail':txt[-400:]}
            ntok+=n_new
            out.write(json.dumps(rec)+'\n'); out.flush()
            T.cuda.empty_cache()
        el=time.time()-t0
        print(f'sample{s} batch{i//B}: {ntok} tok, {ntok/el:.1f} tok/s, {el/60:.1f} min',flush=True)
print('DONE',ntok,'tokens',round((time.time()-t0)/60,1),'min')
