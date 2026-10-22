"""정적 DFlash 드래프트(dflash_draft.py, 변환 ckpt) vs z-lab 원본 구현(dflash/model.py) 동등성.

두 스텝을 비교한다: (1) 빈 캐시 + ctx 12행 + 블록 16, (2) 캐시에 12행 커밋 후 ctx 7행 + 블록 16.
원본은 DynamicCache 로, 정적 모델은 k_ctx/v_ctx 를 캐시 열에 써넣는 방식으로 캐시를 관리한다.
fp32 양쪽. 통과 기준: 로짓 최대 절대 오차 < 2e-2 (softmax 이전 로짓, 값 크기 ~10).
"""
import json, os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.environ["ET_DRAFT"]))  # parent of the patched checkout (scripts/env.sh)
sys.path.insert(0, os.environ["DFLASH_REF_DIR"])  # folder with z-lab dflash model.py (github.com/z-lab/dflash)
torch.manual_seed(0)
torch.set_num_threads(16)
ET_DRAFT = os.environ["ET_DRAFT"]   # ExecuTorch checkout + executorch/patches/draft.patch (scripts/env.sh)
CKPT_DIR = os.environ["CKPT_DIR"]   # qwen3_5_4b_meta.pth, qwen3_5_4b_dflash_meta.pth
SNAP = os.environ.get("DFLASH_HF_DIR") or __import__("huggingface_hub").snapshot_download("z-lab/Qwen3.5-4B-DFlash")
CKPT = CKPT_DIR + "/qwen3_5_4b_dflash_meta.pth"
PARAMS = ET_DRAFT + "/examples/models/qwen3_5_dflash/config/4b_dflash_config.json"
CTX, B, NI = 256, 16, 16

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.dflash_draft import make_dflash_model_class

t0 = time.time()
cfg = ModelArgs(**json.load(open(PARAMS)))
cfg.max_batch_size = 1; cfg.max_seq_len = CTX; cfg.max_context_len = CTX; cfg.use_kv_cache = True
cfg.enable_masked_softmax = True
static = make_dflash_model_class(8)(cfg, ar_len=B, output_new_cache_only=True, output_cache=True,
                                    n_inject=NI, n_target_layers=8)
sd = torch.load(CKPT, weights_only=True, map_location="cpu", mmap=True)
missing, unexpected = static.load_state_dict(sd, strict=False)
missing = [m for m in missing if "freqs" not in m]
assert not missing and not unexpected, (missing[:5], list(unexpected)[:5])
static = static.to(torch.float32).eval()
emb_w = static.tok_embeddings.weight.detach()
dim = cfg.dim
print(f"static draft loaded {time.time()-t0:.0f}s; causal layers {sorted(static.causal_layers)}", flush=True)

# ---- z-lab reference ----
import model as zl
from transformers import Qwen3Config
hf_cfg = json.load(open(os.path.join(SNAP, "config.json")))
ref_cfg = Qwen3Config(**hf_cfg)
ref = zl.DFlashDraftModel(ref_cfg)
from safetensors.torch import load_file
rsd = load_file(os.path.join(SNAP, "model.safetensors"))
mk, uk = ref.load_state_dict({k: v.float() for k, v in rsd.items()}, strict=False)
print("ref missing:", [m for m in mk][:5], "unexpected:", list(uk)[:5], flush=True)
ref = ref.float().eval()
print("ref target_layer_ids", ref.target_layer_ids, "block", ref.block_size, "mask", ref.mask_token_id, flush=True)
assert ref.mask_token_id == static.mask_token_id == 248077
MASK = ref.mask_token_id
head = torch.nn.Linear(dim, emb_w.shape[0], bias=False); head.weight = torch.nn.Parameter(emb_w.clone())

def ref_step(cache, H_rows, block_ids, start):
    n_ctx = H_rows.shape[0]
    pos = torch.arange(start - n_ctx, start + len(block_ids)).unsqueeze(0)
    noise = F.embedding(torch.tensor([block_ids]), emb_w)
    with torch.no_grad():
        h = ref(position_ids=pos, noise_embedding=noise, target_hidden=H_rows.unsqueeze(0),
                past_key_values=cache, use_cache=True)
        logits = ref.compute_logits(h, head)
    return logits[0]  # [B, V]

def static_step(kc, vc, cache_valid, H_rows, block_ids, start):
    r = H_rows.shape[0]
    hs = []
    for i in range(8):
        h = torch.zeros(1, NI, dim); h[0, :r] = H_rows[:, i*dim:(i+1)*dim]; hs.append(h)
    m, mc = static.make_masks(n_ctx_valid=r, n_cache_valid=cache_valid)
    pos_ctx = torch.arange(start - r, start - r + NI, dtype=torch.int32).clamp(max=CTX-1).unsqueeze(0)
    pos_blk = torch.arange(start, start + B, dtype=torch.int32).unsqueeze(0)
    blk = F.embedding(torch.tensor([block_ids]), emb_w)
    with torch.no_grad():
        out = static(*hs, blk, m, mc, pos_blk, pos_ctx, *kc, *vc)
    logits, ks, vs = out
    for i in range(len(kc)):
        kc[i][:, :, :, cache_valid:cache_valid + r] = ks[i][:, :, :, :r]
        vc[i][:, :, cache_valid:cache_valid + r, :] = vs[i][:, :, :r, :]
    return logits[0]

H = torch.randn(24, 8 * dim) * 2.0
ex = static.get_example_inputs()
kc = [t.clone() for t in ex[5]]; vc = [t.clone() for t in ex[6]]
try:
    cache = zl._make_cache(ref_cfg)
except Exception as e:
    print("note: _make_cache fallback:", repr(e)[:80]); from transformers import DynamicCache; cache = DynamicCache()

# step 1: ctx rows 0..11, block at 12
blk1 = [846] + [MASK] * (B - 1)
r1 = ref_step(cache, H[:12], blk1, 12)
s1 = static_step(kc, vc, 0, H[:12], blk1, 12)
zl._crop_to(cache, 12)
d1 = (r1 - s1).abs().max().item(); a1 = (r1.argmax(-1) == s1.argmax(-1)).float().mean().item()
print(f"[step1] max|dlogit|={d1:.4f}  argmax agree={a1:.3f}  |logit| max={r1.abs().max():.2f}", flush=True)
# step 2: ctx rows 12..18 (7 rows), block at 19
blk2 = [1675] + [MASK] * (B - 1)
r2 = ref_step(cache, H[12:19], blk2, 19)
s2 = static_step(kc, vc, 12, H[12:19], blk2, 19)
d2 = (r2 - s2).abs().max().item(); a2 = (r2.argmax(-1) == s2.argmax(-1)).float().mean().item()
print(f"[step2] max|dlogit|={d2:.4f}  argmax agree={a2:.3f}", flush=True)
ok = d1 < 2e-2 and d2 < 2e-2
print("PASS" if ok else "FAIL")
