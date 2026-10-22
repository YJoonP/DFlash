# Inputs of run_spec6.py: six Spec-Bench prompts (question_id 81 writing, 101 reasoning, 121 coding, 141 stem,
# 161 translation, 401 math_reasoning; first turn only) -> RESULTS_DIR/inputs/{prompt_<id>.txt, manifest.json, ids.txt}.
# Same format as results/s26_snapshot_vs_recompute_20261012/inputs/manifest.json; this tokenizer + chat template reproduces
# all 18 entries there exactly (token_ids, expected_n_past thinking / no-thinking, raw_sha256 = sha256 of int64 token ids).
# usage: SPECBENCH_JSONL=<Spec-Bench data/spec_bench/question.jsonl> [RESULTS_DIR=<folder>] python3 make_spec6_inputs.py
import json, hashlib, os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HERE = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[3] / 'results/s26_spec6_20261016'))
SRC = Path(os.environ['SPECBENCH_JSONL'])
IDS = [81, 101, 121, 141, 161, 401]
MT = {'writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities'}
t = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')
rows = {r['question_id']: r for r in map(json.loads, SRC.open())}
old = {r['id']: r for r in json.loads((HERE.parent / 's26_snapshot_vs_recompute_20261012/inputs/manifest.json').read_text())}


def tok(text, think):
    ids = t.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, enable_thinking=think)
    return list(ids['input_ids'] if hasattr(ids, 'keys') else ids)


man = []
for qid in IDS:
    r = rows[qid]
    text = r['turns'][0]
    raw = text.encode()
    (HERE / f'inputs/prompt_{qid}.txt').write_bytes(raw)
    ids = tok(text, True)
    cat = 'mt_bench' if r['category'] in MT else r['category']
    e = {'id': qid, 'category': cat, 'subcategory': r['category'], 'file': f'prompt_{qid}.txt', 'bytes': len(raw),
         'sha256': hashlib.sha256(raw).hexdigest(), 'trailing_newline_added': False, 'has_inner_newline': '\n' in text,
         'expected_n_past': len(ids), 'expected_n_past_nothink': len(tok(text, False)), 'prompt_head': text[:80],
         'token_ids': ids, 'raw_sha256': hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest()}
    if qid in old:
        assert e == old[qid], f'{qid} differs from the 18-prompt manifest'
    man.append(e)
    print(qid, cat, r['category'], len(raw), 'bytes', len(ids), 'tokens', 'same as 18-prompt manifest' if qid in old else 'new')
(HERE / 'inputs/manifest.json').write_text(json.dumps(man, indent=2, ensure_ascii=False) + '\n')
(HERE / 'inputs/ids.txt').write_text(' '.join(map(str, IDS)) + '\n')
