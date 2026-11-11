import json, time, sys, os, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mathqa as Q
from transformers import AutoTokenizer

# Gemma lane: identical protocol to eval_run_gpu.py, transport = koboldcpp /v1/completions.
# Prompt templating stays CLIENT-side via HF tokenizer so enable_thinking=True is preserved.
# Deviations vs Qwen arm: engine=llama.cpp(koboldcpp), weights=bf16 GGUF (same precision class),
# presence_penalty applied server-side (OpenAI semantics == our PresencePenalty class).

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')            # tokenizer path (for templating); model lives server-side
ap.add_argument('subset'); ap.add_argument('out')
ap.add_argument('--samples', type=int, default=1)
ap.add_argument('--batch', type=int, default=1)
ap.add_argument('--max_tokens', type=int, default=32768)
ap.add_argument('--dtype', default='bfloat16')  # recorded, not applied
ap.add_argument('--server', default='http://127.0.0.1:18080')
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.ckpt)
probs = [json.loads(l) for l in open(a.subset)]
out = open(a.out, 'a')
B = a.batch
URL = a.server + '/v1/completions'

def gen(text):
    body = json.dumps({
        'prompt': text,
        'max_tokens': a.max_tokens,
        'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
        'presence_penalty': 1.5,
        'stream': False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        j = json.loads(r.read())
    ch = j['choices'][0]
    return ch['text'], ch.get('finish_reason', ''), j.get('usage', {})

t0 = time.time(); ntok = 0
for s in range(a.samples):
    for i in range(0, len(probs), B):
        chunk = probs[i:i+B]
        for j, p in enumerate(chunk):
            msg = Q.build_prompt(p['problem'])
            text = tok.apply_chat_template([{'role': 'user', 'content': msg}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
            # deterministic per-sample seed parity is engine-side; kobold seed via env not per-request
            try:
                txt, fin, usage = gen(text)
                n_new = usage.get('completion_tokens') or len(tok.encode(txt))
            except Exception as e:
                rec = {'idx': p['idx'], 'sample': s, 'error': str(e)}
                out.write(json.dumps(rec) + '\n'); out.flush(); continue
            cut = fin == 'length' or n_new >= a.max_tokens
            rec = {'idx': p['idx'], 'sample': s, 'n_tokens': n_new, 'cut': cut,
                   'correct': Q.is_correct(Q.extract(txt)[0], p['gold']),
                   'gold': p['gold'], 'text_tail': txt[-400:]}
            ntok += n_new
            out.write(json.dumps(rec) + '\n'); out.flush()
        el = time.time() - t0
        print(f'sample{s} batch{i//B}: {ntok} tok, {ntok/el:.1f} tok/s, {el/60:.1f} min', flush=True)
print('DONE', ntok, 'tokens', round((time.time()-t0)/60, 1), 'min')
