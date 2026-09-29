import os, time, traceback
os.environ.setdefault('HF_ENDPOINT', 'https://aifasthub.com')
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification

targets = [
    ('nghuyong/ernie-3.0-base-zh',                    AutoModel),
    ('uer/t5-base-chinese-cluecorpussmall',           AutoModelForSeq2SeqLM),
    ('uer/gpt2-chinese-cluecorpussmall',              AutoModelForCausalLM),
    ('uer/roberta-base-finetuned-jd-binary-chinese',  AutoModelForSequenceClassification),
]
for name, cls in targets:
    t0 = time.time()
    try:
        AutoTokenizer.from_pretrained(name)
        cls.from_pretrained(name)
        print(f'[OK]   {name}  ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'[FAIL] {name}: {e}', flush=True)
print('[DONE] all downloads finished', flush=True)
