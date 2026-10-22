"""DFlash 추측 디코딩 참조 루프 (서버 CPU, fp32 정적 모델) — 프로토콜 검증 + 수락 길이 측정.

타깃: 정적 Qwen3.5-4B (QNN_DEFER_STATE=1, ar_len=T 검증 그래프 인스턴스) + 즉시 커밋 ar_len=1 인스턴스(AR 참조).
드래프트: 정적 DFlash 8-hidden 모델(변환 ckpt).
검증 규칙(온도 0): 드래프트 토큰 d_1..d_15 vs 타깃 argmax, 앞에서부터 일치한 개수 acc, bonus=posterior[acc].
게이트: SD 생성 토큰열 == 타깃 AR greedy 토큰열 (접두사 일치). 이것이 안 맞으면 수락 길이는 의미가 없다.

사용: python dflash_sd_reference.py --prompt-idx 0 --max-new 96 [--verify-ar 16]
"""
import argparse, json, os, sys, time
os.environ["QNN_DEFER_STATE"] = "1"
os.environ.setdefault("QNN_DELTA_KERNEL", "0")
import torch, torch.nn.functional as F
ET_DRAFT = os.environ["ET_DRAFT"]   # ExecuTorch checkout + executorch/patches/draft.patch (scripts/env.sh)
CKPT_DIR = os.environ["CKPT_DIR"]   # qwen3_5_4b_meta.pth, qwen3_5_4b_dflash_meta.pth
sys.path.insert(0, os.path.dirname(ET_DRAFT))  # the executorch namespace resolves to the patched checkout
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import LlamaModel
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import make_dflash_model_class
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask, CausalAttentionMask

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../scripts/export/calib_specbench39.json"))
ap.add_argument("--prompt-idx", type=int, default=0)
ap.add_argument("--prompt-text", default=None)
ap.add_argument("--max-new", type=int, default=96)
ap.add_argument("--verify-ar", type=int, default=16, help="target verify graph width T (>= block)")
ap.add_argument("--block", type=int, default=16)
ap.add_argument("--ctx", type=int, default=1152)
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--skip-ar", action="store_true", help="skip the AR reference (no identity gate)")
ap.add_argument("--out", default=None)
ap.add_argument("--no-think", action="store_true", help="chat template with enable_thinking=False (empty <think></think> block, as the device driver does)")
ap.add_argument("--prompt-ids-file", default=None, help="use these token ids as the prompt (e.g. the device's own tokenization)")
ap.add_argument("--fq-w", default="none", choices=["none", "w4g16", "w8"],
                help="fake-quantize draft weights like the NPU recipe: w4g16 = int4 symmetric per 16-input block "
                     "(Linear/conv) with lm_head int8 per-channel; w8 = int8 per-channel everywhere")
ap.add_argument("--fq-kv8", action="store_true", help="fake-quantize the draft KV cache to uint8 with the exported per-layer scales")
ap.add_argument("--fq-hid16", type=float, default=0.0, help="fake-quantize the target hiddens fed to the draft to uint16 per-tensor with this margin (0 = off)")
args = ap.parse_args()
torch.set_num_threads(args.threads)
T, B, CTX, NI = args.verify_ar, args.block, args.ctx, 16
assert T >= B
TARGET_LAYERS = [1, 5, 9, 13, 17, 21, 25, 29]
MASK_ID = 248077

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
if args.prompt_text:
    msgs = [{"role": "user", "content": args.prompt_text}]
else:
    msgs = json.load(open(args.calib))[args.prompt_idx]["messages"]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=not args.no_think)
ids = tok(prompt, add_special_tokens=False)["input_ids"]
if args.prompt_ids_file:
    ids = [int(x) for x in open(args.prompt_ids_file).read().split()]
    print(f"prompt ids taken from {args.prompt_ids_file}: {len(ids)} tokens", flush=True)
n_prompt = len(ids)
print(f"prompt {args.prompt_idx}: {n_prompt} tokens | {msgs[0]['content'][:60]!r}", flush=True)

t0 = time.time()
tcfg = json.load(open(ET_DRAFT + "/examples/models/qwen3_5/config/4b_config.json"))
def target_cfg(ar):
    c = ModelArgs(**tcfg); c.max_batch_size = 1; c.max_seq_len = CTX; c.max_context_len = CTX
    c.use_kv_cache = True; c.enable_masked_softmax = True; c.ar_len = ar; return c
sd = torch.load(CKPT_DIR + "/qwen3_5_4b_meta.pth", weights_only=True, map_location="cpu", mmap=True)
V = LlamaModel(target_cfg(T), ar_len=T, output_new_cache_only=True, output_cache=True)
V.load_state_dict(sd, strict=False); V = V.to(torch.float32).eval()
assert V.deferred_state
layer_types = V.layer_types
if not args.skip_ar:
    A = LlamaModel(target_cfg(1), ar_len=1, output_new_cache_only=True, output_cache=True)
    # 체크포인트에서 직접 로드한다. V.state_dict() 를 넣으면 _fold_unit_offset 훅이 두 번 돌아
    # RMSNorm 이 2+w 가 된다 (2026-09-08 의 사고와 같은 경로).
    A.load_state_dict(sd, strict=False); A = A.to(torch.float32).eval()
dcfg = ModelArgs(**json.load(open(ET_DRAFT + "/examples/models/qwen3_5_dflash/config/4b_dflash_config.json")))
dcfg.max_batch_size = 1; dcfg.max_seq_len = CTX; dcfg.max_context_len = CTX; dcfg.use_kv_cache = True; dcfg.enable_masked_softmax = True
D = make_dflash_model_class(8)(dcfg, ar_len=B, output_new_cache_only=True, output_cache=True, n_inject=NI, n_target_layers=8)
dsd = torch.load(CKPT_DIR + "/qwen3_5_4b_dflash_meta.pth", weights_only=True, map_location="cpu", mmap=True)
D.load_state_dict(dsd, strict=False); D = D.to(torch.float32).eval()
emb_w = D.tok_embeddings.weight.detach()

def _fq_sym(w2d, bits, group):
    """symmetric integer fake-quant of a [out, in] weight, one scale per (out, group of `group` inputs)."""
    out, inn = w2d.shape
    qmax = 2 ** (bits - 1) - 1
    if group is None or inn % group != 0:
        s_ = w2d.abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
        return (torch.round(w2d / s_).clamp(-qmax, qmax) * s_)
    wg = w2d.reshape(out, inn // group, group)
    s_ = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(wg / s_).clamp(-qmax, qmax) * s_).reshape(out, inn)

if args.fq_w != "none":
    n_q = 0
    with torch.no_grad():
        for name, mod in D.named_modules():
            if not isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)) or "tok_embeddings" in name:
                continue
            w = mod.weight
            w2 = w.reshape(w.shape[0], -1)
            is_head = name.split(".")[-1] == "output" or name == "output"
            if args.fq_w == "w8" or is_head:
                q = _fq_sym(w2, 8, None)
            else:
                q = _fq_sym(w2, 4, 16)
            w.copy_(q.reshape(w.shape)); n_q += 1
    print(f"fake-quant weights: {args.fq_w} on {n_q} modules", flush=True)
# exported draft KV cache encodings (ctx_draft/forward_0_json.json, prefill_forward): uint8, offset -128
KV8_K = [0.06832, 0.05846, 0.07118, 0.07868, 0.06862, 0.0709]
KV8_V = [0.1598, 0.2289, 0.3006, 0.4473, 0.4362, 0.6098]
def _fq_u8(x, scale):
    return (torch.round(x / scale).clamp(-128, 127)) * scale
def _fq_u16_tensor(x, margin):
    lo, hi = x.min() * margin, x.max() * margin
    s_ = ((hi - lo) / 65535).clamp(min=1e-12)
    return torch.round((x - lo) / s_).clamp(0, 65535) * s_ + lo
print(f"models loaded {time.time()-t0:.0f}s", flush=True)

captured = {}
for i in TARGET_LAYERS:
    V.layers[i].register_forward_hook(lambda m, inp, out, i=i: captured.__setitem__(i, out[0].detach()))

def hidden_rows(n):  # rows 0..n-1 of the last V call -> [n, 8*dim]
    return torch.cat([captured[i][0, :n] for i in TARGET_LAYERS], dim=-1)

# ---------------- target verify-graph state (deferred protocol) ----------------
exV = V.get_example_inputs()
maskV = AttentionMask(CausalAttentionMask(1, T, CTX)); kV = [t.clone() for t in exV[3]]; vV = [t.clone() for t in exV[4]]
valid_prev = torch.zeros_like(exV[5]); ingV = [t.clone() for t in exV[6]]
kv_pos = 0  # committed length in attention KV caches (== number of accepted tokens so far)

@torch.no_grad()
def target_step(tokens, start, n_commit_now):
    """Run V on a T-chunk at positions start.. ; commit the previous chunk (valid_prev).
    Returns logits [T, V]. Caller decides how many of this chunk's positions are accepted (n_acc) and
    calls target_commit(start, n_acc) to write attention KV + set valid_prev/ingredients."""
    global ingV, valid_prev
    chunk = list(tokens) + [0] * (T - len(tokens))
    out = V(torch.tensor([chunk], dtype=torch.int32), list(maskV)[0],
            torch.arange(start, start + T, dtype=torch.int32).unsqueeze(0), *kV, *vV, valid_prev, *ingV)
    logits, newK, newV, new_ing = out
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention": kV[i], vV[i] = newK[i], newV[i]
    target_step.pending = (newK, newV, new_ing)
    return logits[0]

def target_commit(start, n_acc):
    """accept the first n_acc positions of the chunk just verified"""
    global valid_prev, ingV, kv_pos
    newK, newV, new_ing = target_step.pending
    for i, lt in enumerate(layer_types):
        if lt != "linear_attention":
            kV[i][:, :, :, start:start + n_acc] = newK[i][:, :, :, :n_acc]
            vV[i][:, :, start:start + n_acc, :] = newV[i][:, :, :n_acc, :]
    ingV = [t.clone() for t in new_ing]
    valid_prev = torch.zeros_like(valid_prev); valid_prev[0, :n_acc] = 1.0
    maskV.smart_mask_update(start, n_acc); kv_pos = start + n_acc

# ---------------- draft state ----------------
exD = D.get_example_inputs()
kD = [t.clone() for t in exD[5]]; vD = [t.clone() for t in exD[6]]; d_cache_pos = 0

@torch.no_grad()
def draft_step(rows, start, block_ids):
    """inject `rows` ([r, 8*dim], positions start-r..start-1) and propose the block at start."""
    global d_cache_pos
    r = rows.shape[0]
    hs = []
    for i in range(8):
        h = torch.zeros(1, NI, D.dim); h[0, :r] = rows[:, i*D.dim:(i+1)*D.dim]
        if args.fq_hid16 > 0:
            h = _fq_u16_tensor(h, args.fq_hid16)
        hs.append(h)
    m, mc = D.make_masks(n_ctx_valid=r, n_cache_valid=d_cache_pos)
    pos_ctx = torch.arange(start - r, start - r + NI, dtype=torch.int32).clamp(max=CTX-1).unsqueeze(0)
    pos_blk = torch.arange(start, start + B, dtype=torch.int32).unsqueeze(0)
    out = D(*hs, F.embedding(torch.tensor([block_ids]), emb_w), m, mc, pos_blk, pos_ctx, *kD, *vD)
    logits, ks, vs = out
    for i in range(len(kD)):
        kn, vn = ks[i][:, :, :, :r], vs[i][:, :, :r, :]
        if args.fq_kv8:
            kn, vn = _fq_u8(kn, KV8_K[i]), _fq_u8(vn, KV8_V[i])
        kD[i][:, :, :, d_cache_pos:d_cache_pos + r] = kn
        vD[i][:, :, d_cache_pos:d_cache_pos + r, :] = vn
    d_cache_pos += r
    return logits[0]

# ---------------- 1) prompt prefill through the verify graph ----------------
t1 = time.time()
prompt_rows = []
first_logits = None
for s in range(0, n_prompt, T):
    chunk = ids[s:s + T]; n = len(chunk)
    lg = target_step(chunk, s, None)
    prompt_rows.append(hidden_rows(n))
    if s + n == n_prompt:
        first_logits = lg[n - 1]
    target_commit(s, n)
H_prompt = torch.cat(prompt_rows, 0)
t_last = int(first_logits.argmax())
generated = [t_last]
print(f"prefill {n_prompt} tokens in {time.time()-t1:.1f}s; first token {t_last} {tok.decode([t_last])!r}", flush=True)

# 2) inject prompt hiddens into the draft (16 rows per call; the last call carries the real block)
start = n_prompt
pending_rows = H_prompt
acc_lens = []
verify_times, draft_times = [], []
while len(generated) < args.max_new:
    # draft: inject pending rows (<=16 each call), the last injection proposes the block at `start`
    td = time.time()
    while pending_rows.shape[0] > NI:
        rows, pending_rows = pending_rows[:NI], pending_rows[NI:]
        draft_step(rows, start - pending_rows.shape[0], [t_last] + [MASK_ID] * (B - 1))
    block = [t_last] + [MASK_ID] * (B - 1)
    dlog = draft_step(pending_rows, start, block)
    drafts = dlog[1:B].argmax(-1).tolist()          # d_1..d_{B-1}
    draft_times.append(time.time() - td)
    # verify
    tv = time.time()
    chunk = [t_last] + drafts
    lg = target_step(chunk, start, None)
    posterior = lg[:B].argmax(-1).tolist()
    acc = 0
    while acc < B - 1 and drafts[acc] == posterior[acc]:
        acc += 1
    bonus = posterior[acc]
    produced = acc + 1
    target_commit(start, produced)
    verify_times.append(time.time() - tv)
    pending_rows = hidden_rows(produced)              # target hiddens of [t_last, d_1..d_acc]
    generated.extend(drafts[:acc] + [bonus])
    acc_lens.append(produced)
    t_last = bonus
    start += produced
    if bonus in (248044, 248043):  # <|endoftext|> / <|im_end|>
        break
generated = generated[:args.max_new]
al = sum(acc_lens) / len(acc_lens)
print(f"SD: {len(generated)} tokens in {len(acc_lens)} steps, AL={al:.2f} "
      f"(verify {sum(verify_times)/len(verify_times):.2f}s, draft {sum(draft_times)/len(draft_times):.2f}s per step)")
print("SD text:", repr(tok.decode(generated))[:400], flush=True)

result = {"prompt_idx": args.prompt_idx, "n_prompt": n_prompt, "n_generated": len(generated),
          "steps": len(acc_lens), "acc_lens": acc_lens, "AL": al, "sd_tokens": generated}
# ---------------- 3) AR greedy reference (immediate model, token by token) ----------------
if not args.skip_ar:
    t2 = time.time()
    exA = A.get_example_inputs(); maskA = AttentionMask(CausalAttentionMask(1, 1, CTX))
    kA = [t.clone() for t in exA[3]]; vA = [t.clone() for t in exA[4]]
    cur = ids[0]; ar = []
    with torch.no_grad():
        for p in range(n_prompt + len(generated) - 1):
            out = A(torch.tensor([[cur]], dtype=torch.int32), list(maskA)[0], torch.tensor([[p]], dtype=torch.int32), *kA, *vA)
            for i, lt in enumerate(layer_types):
                if lt == "linear_attention": kA[i], vA[i] = out[1][i], out[2][i]
                else: kA[i][:, :, :, p:p+1] = out[1][i]; vA[i][:, :, p:p+1, :] = out[2][i]
            maskA.smart_mask_update(p, 1)
            nxt = int(out[0][0, 0].argmax())
            if p + 1 < n_prompt: cur = ids[p + 1]
            else: ar.append(nxt); cur = nxt
    match = 0
    while match < min(len(ar), len(generated)) and ar[match] == generated[match]:
        match += 1
    gate = match == min(len(ar), len(generated))
    print(f"AR reference {len(ar)} tokens in {time.time()-t2:.0f}s; prefix match {match}/{min(len(ar), len(generated))} -> {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  first mismatch at", match, "SD", generated[match:match+5], "AR", ar[match:match+5])
    result.update({"ar_tokens": ar, "identity_gate": gate, "prefix_match": match})
if args.out:
    json.dump(result, open(args.out, "w"), indent=1)
    print("saved", args.out)
