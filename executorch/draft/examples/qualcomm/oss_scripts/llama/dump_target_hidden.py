#!/usr/bin/env python3
"""DFlash 드래프트 캘리브레이션용 타깃 hidden 덤프 (Qwen3.5-4B, 정적 fp32 모델).

QNN_Exporter 의 dump_target_hidden.py 는 HF bf16 타깃을 쓰지만, 이 서버의 transformers(5.0.0rc1)는
qwen3_5 를 모르고, 어차피 양자화되는 것은 포크의 정적 모델이므로 **정적 fp32 모델**에서 뽑는다
(단위 오프셋 folding·L2Norm 재작성 등 배포 그래프와 같은 수식). 정적 모델은 ar_len 폭의 청크로
프롬프트를 처리하며(DeltaNet 상태·KV 를 carry), 각 층 출력(residual stream)에 hook 을 걸어
target_layer_ids 층의 출력을 모은다.

DFlash 규약: target_layer_ids=[1,5,...,29] 는 HF hidden_states[i+1], 즉 디코더 층 i 의 출력이다.
저장 형식: {"entries": [{"ids": [...], "hidden": bf16 [n, n_target*dim]}], "layer_ids", "dim", ...}
프롬프트 토큰과 hidden 행이 1:1 로 정렬돼 있어 캘리브레이션이 실제 디코딩 루프를 흉내 낼 수 있다.

사용:
  python dump_target_hidden.py --calib calib_specbench.json --out hidden_q35_4b.pt [--limit N]
"""
import argparse
import json
import os
import sys
import time

import torch

# work folder holding executorch/, ckpt/qwen3_5_4b_meta.pth and models/qwen3_5_4B (the HF tokenizer files)
ROOT = os.environ.get("QWEN35_ROOT", ".")
sys.path.insert(0, ROOT + "/executorch")

from executorch.examples.models.llama.model_args import ModelArgs  # noqa: E402
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (  # noqa: E402
    AttentionMask,
    CausalAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (  # noqa: E402
    LlamaModel,
)


def render_prompts(raw, tokenizer, add_generation_prompt=True, enable_thinking=None):
    prompts = []
    for item in raw:
        if isinstance(item, str):
            prompts.append(item)
            continue
        msgs = item["messages"] if "messages" in item else item
        kw = {}
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        prompts.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw
            )
        )
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="JSON list of {messages:[...]} or strings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=ROOT + "/ckpt/qwen3_5_4b_meta.pth")
    ap.add_argument("--params", default=ROOT + "/executorch/examples/models/qwen3_5/config/4b_config.json")
    ap.add_argument("--tokenizer", default=ROOT + "/models/qwen3_5_4B")
    ap.add_argument("--layer-ids", default="1,5,9,13,17,21,25,29")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=128, help="static model ar_len (chunk width)")
    ap.add_argument("--ctx", type=int, default=1152)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-generation-prompt", action="store_true")
    ap.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "16")))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("QNN_DELTA_KERNEL", "0")  # pure-torch unrolled recurrence

    layer_ids = [int(x) for x in args.layer_ids.split(",")]
    raw = json.load(open(args.calib))
    if isinstance(raw, dict):
        raw = raw.get("prompts", list(raw.values()))
    if args.limit:
        raw = raw[: args.limit]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    thinking = {"default": None, "on": True, "off": False}[args.thinking]
    prompts = render_prompts(raw, tok, not args.no_generation_prompt, thinking)

    cfg = ModelArgs(**json.load(open(args.params)))
    cfg.max_batch_size = 1
    cfg.max_seq_len = args.ctx
    cfg.max_context_len = args.ctx
    cfg.use_kv_cache = True
    L = args.chunk
    t0 = time.time()
    model = LlamaModel(cfg, ar_len=L, output_new_cache_only=True, output_cache=True)
    sd = torch.load(args.ckpt, weights_only=True, map_location="cpu", mmap=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "freqs" not in m]
    assert not missing and not unexpected, (missing[:5], list(unexpected)[:5])
    model = model.to(torch.float32).eval()
    print(f"static target loaded in {time.time()-t0:.0f}s (ar_len={L}, ctx={args.ctx})", flush=True)

    captured = {}

    def make_hook(i):
        def hook(mod, inp, out):
            captured[i] = out[0].detach()  # residual stream after layer i: [1, L, dim]
        return hook

    for i in layer_ids:
        model.layers[i].register_forward_hook(make_hook(i))

    ex = model.get_example_inputs()
    layer_types = cfg.layer_types
    entries = []
    stats_min, stats_max, stats_abs, n_rows = None, None, 0.0, 0
    with torch.no_grad():
        for pi, p in enumerate(prompts):
            ids = tok(p, add_special_tokens=False, truncation=True, max_length=args.max_len)["input_ids"]
            n = len(ids)
            if n == 0:
                continue
            mask_obj = AttentionMask(CausalAttentionMask(1, L, args.ctx))
            kA = [t.clone() for t in ex[3]]
            kB = [t.clone() for t in ex[4]]
            rows = []
            for start in range(0, n, L):
                chunk = ids[start : start + L]
                valid = len(chunk)
                chunk = chunk + [0] * (L - valid)
                tokens = torch.tensor([chunk], dtype=torch.int32)
                pos_t = torch.arange(start, start + L, dtype=torch.int32).unsqueeze(0)
                live_mask = list(mask_obj)[0]
                out = model(tokens, live_mask, pos_t, *kA, *kB)
                if len(out) == 3 and isinstance(out[1], (list, tuple)):
                    newA, newB = list(out[1]), list(out[2])
                else:
                    _, *caches = out
                    newA, newB = caches[: len(kA)], caches[len(kA) :]
                for li, lt in enumerate(layer_types):
                    if lt == "linear_attention":
                        kA[li], kB[li] = newA[li], newB[li]
                    else:
                        kA[li][:, :, :, start : start + L] = newA[li]
                        kB[li][:, :, start : start + L, :] = newB[li]
                mask_obj.smart_mask_update(start, L)
                h = torch.cat([captured[i][0, :valid] for i in layer_ids], dim=-1)  # [valid, n*dim]
                rows.append(h)
            H = torch.cat(rows, dim=0)  # [n, n_target*dim]
            a = H.abs()
            stats_min = H.min() if stats_min is None else torch.minimum(stats_min, H.min())
            stats_max = H.max() if stats_max is None else torch.maximum(stats_max, H.max())
            stats_abs += a.sum().item()
            n_rows += H.numel()
            entries.append({"ids": ids, "hidden": H.to(torch.bfloat16)})
            print(f"  [{pi+1}/{len(prompts)}] {n:4d} tokens -> {tuple(H.shape)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "entries": entries,
            "layer_ids": layer_ids,
            "dim": cfg.dim,
            "target": args.ckpt,
            "tokenizer": args.tokenizer,
            "add_generation_prompt": not args.no_generation_prompt,
            "thinking": args.thinking,
        },
        args.out,
    )
    print(f"\nwrote {args.out}: {len(entries)} prompts, {sum(len(e['ids']) for e in entries)} rows")
    print(f"  range [{float(stats_min):.3f}, {float(stats_max):.3f}]  mean|x| {stats_abs/max(n_rows,1):.4f}")


if __name__ == "__main__":
    main()
