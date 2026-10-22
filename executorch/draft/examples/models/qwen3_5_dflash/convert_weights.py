"""z-lab/Qwen3.5-4B-DFlash 드래프트 체크포인트를 정적 디코더(static_llama) 파라미터 이름으로 바꾼다.

2026-09-15 KST. Qwen3.5-4B 용 DFlash 드래프트는 Qwen3-4B 용과 두 가지가 다르다.
  1. 타깃 hidden 을 8층에서 읽는다(target_layer_ids [1,5,9,13,17,21,25,29]) -> fc 가 [2560, 8*2560].
     모델은 fc 를 층별 블록 8개로 나눠 합산하므로(fc.{i}.weight) 여기서 쪼갠다.
  2. 체크포인트에 임베딩/lm_head 가 없다. 원본 구현은 타깃의 embed_tokens / lm_head 를 그대로
     쓴다(Qwen3.5-4B 는 tied). 그래서 타깃 메타 체크포인트(ckpt/qwen3_5_4b_meta.pth)의
     tok_embeddings.weight 를 tok_embeddings.weight 와 output.weight 로 넣는다.
     (임베딩은 그래프 밖(호스트 fp16 테이블)이고 output 만 그래프에 들어간다.)

사용:
  python convert_weights.py <draft_hf_dir> <out.pth> --target-ckpt <qwen3_5_4b_meta.pth>
"""

import argparse
import os
from typing import Dict

import torch

_DFLASH_FROM_HF = {
    "norm.weight": "norm.weight",
    "hidden_norm.weight": "hidden_norm.weight",
}

_DFLASH_LAYER_FROM_HF = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "self_attn.q_norm.weight": "attention.q_norm_fn.weight",
    "self_attn.k_norm.weight": "attention.k_norm_fn.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # Qwen3 MLP: down(silu(gate(x)) * up(x)) -> static FeedForward: w2(silu(w1(x)) * w3(x))
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}


def load_draft_checkpoint(input_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = os.path.join(input_dir, "model.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model.safetensors not found in {input_dir}")
    return load_file(path)


def dflash_hf_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key == "fc.weight":
            dim = value.shape[0]
            n = value.shape[1] // dim
            assert value.shape[1] == dim * n, f"fc is not a whole number of blocks: {value.shape}"
            for i in range(n):
                out[f"fc.{i}.weight"] = value[:, i * dim : (i + 1) * dim].contiguous()
            continue
        if key in _DFLASH_FROM_HF:
            out[_DFLASH_FROM_HF[key]] = value
            continue
        if key.startswith("layers."):
            _, idx, rest = key.split(".", 2)
            if rest in _DFLASH_LAYER_FROM_HF:
                out[f"layers.{idx}.{_DFLASH_LAYER_FROM_HF[rest]}"] = value
                continue
        raise KeyError(f"unmapped DFlash tensor: {key}")
    return out


def convert_weights(input_dir: str, output_file: str, target_ckpt: str = None) -> None:
    target_ckpt = target_ckpt or os.environ.get(
        "QNN_DFLASH_TARGET_CKPT", "qwen3_5_4b_meta.pth"
    )
    print(f"Loading draft checkpoint from {input_dir} ...")
    sd = dflash_hf_to_meta(load_draft_checkpoint(input_dir))
    print(f"  {len(sd)} tensors, fc blocks: {sum(k.startswith('fc.') for k in sd)}")
    print(f"Loading tied embedding from target checkpoint {target_ckpt} ...")
    tgt = torch.load(target_ckpt, weights_only=True, map_location="cpu", mmap=True)
    emb = tgt["tok_embeddings.weight"]
    if emb.shape[0] != 248320 or emb.shape[1] != sd["norm.weight"].shape[0]:
        raise ValueError(f"unexpected embedding shape {tuple(emb.shape)}")
    emb = emb.clone()  # detach from mmap
    sd["tok_embeddings.weight"] = emb
    sd["output.weight"] = emb  # same storage: tied
    n_params = sum(v.numel() for k, v in sd.items() if k != "output.weight")
    print(f"Saving {output_file} ({n_params/1e6:.0f}M params incl. embedding, dtype {emb.dtype}) ...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="Convert a Qwen3.5 DFlash draft checkpoint.")
    p.add_argument("input_dir", help="HF snapshot dir holding model.safetensors")
    p.add_argument("output", help="output .pth")
    p.add_argument("--target-ckpt", default=None, help="target meta checkpoint providing the tied embedding")
    a = p.parse_args()
    convert_weights(a.input_dir, a.output, a.target_ckpt)


if __name__ == "__main__":
    main()
