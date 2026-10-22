#!/usr/bin/env python3
"""런타임(MobiSpec.cpp llama-speculative-dflash-qnn-npu)이 컨텍스트 옆에서 읽는 params.json.

QNN_Exporter/scripts/make_params.py 를 Qwen3.5 용으로 바꿨다. export 는 이 파일을 만들지 않는다.
  --target : Qwen3.5-4B 타깃 (DeltaNet 상태 슬롯 정보를 함께 적는다 — 런타임 포팅용 확장 필드)
  --draft  : Qwen3.5-4B-DFlash 드래프트 (n_inject / n_target_layers / block_size / mask_token_id)
사용:
  make_params.py --target --out ctx_target/params.json
  make_params.py --draft  --out ctx_draft/params.json
"""
import argparse, json, os

# model configs inside the patched ExecuTorch checkouts (scripts/env.sh: ET = target, ET_DRAFT = draft)
TARGET_CFG = os.path.join(os.environ.get("ET", "executorch"), "examples/models/qwen3_5/config/4b_config.json")
DRAFT_CFG = os.path.join(os.environ.get("ET_DRAFT", os.environ.get("ET", "executorch")),
                         "examples/models/qwen3_5_dflash/config/4b_dflash_config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target", action="store_true")
    g.add_argument("--draft", action="store_true")
    ap.add_argument("--n-inject", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--shard-layers", default="2,6,10,14,18,22,26,30")
    # 2B 등 다른 크기 타깃용 (2026-09-28 KST). 기본값은 4B 그대로.
    ap.add_argument("--target-config", default=TARGET_CFG)
    ap.add_argument("--target-layer-ids", default="1,5,9,13,17,21,25,29")
    ap.add_argument("--verify-ar-len", type=int, default=16, help="target prefill/verify graph width")
    ap.add_argument("--defer-state", type=int, default=1, help="target exported with QNN_DEFER_STATE")
    ap.add_argument("--defer-masked", type=int, default=1,
                    help="2026-10-01 masked-op deferral: int32 valid_prev, then valid_cur and conv_idx_prev inputs")
    a = ap.parse_args()

    if a.target:
        c = json.load(open(a.target_config))
        p = {
            "dim": c["dim"], "n_layers": c["n_layers"], "n_heads": c["n_heads"],
            "n_kv_heads": c["n_kv_heads"], "head_dim": c["head_dim"], "vocab_size": c["vocab_size"],
            "ffn_dim_multiplier": 1, "multiple_of": 256, "norm_eps": c["norm_eps"],
            "rope_theta": c["rope_theta"], "use_scaled_rope": False,
            # ---- Qwen3.5 hybrid extensions (runtime port) ----
            "model_arch": "qwen3_5_hybrid",
            "layer_types": c["layer_types"],
            "n_full_attention_layers": sum(t == "full_attention" for t in c["layer_types"]),
            "linear_num_key_heads": c["linear_num_key_heads"],
            "linear_num_value_heads": c["linear_num_value_heads"],
            "linear_key_head_dim": c["linear_key_head_dim"],
            "linear_value_head_dim": c["linear_value_head_dim"],
            "linear_conv_kernel_dim": c["linear_conv_kernel_dim"],
            "conv_dim": 2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
                        + c["linear_num_value_heads"] * c["linear_value_head_dim"],
            "shard_layers": [int(x) for x in a.shard_layers.split(",")],
            "dflash_target_layer_ids": [int(x) for x in a.target_layer_ids.split(",")],
            "verify_ar_len": a.verify_ar_len,
            "defer_state": a.defer_state,
            "defer_masked": a.defer_masked,
        }
    else:
        c = json.load(open(DRAFT_CFG))
        akw = c.get("attention_kwargs", {})
        p = {
            "dim": c["dim"], "n_layers": c["n_layers"], "n_heads": c["n_heads"],
            "n_kv_heads": c["n_kv_heads"], "head_dim": c["head_dim"], "vocab_size": c["vocab_size"],
            "ffn_dim_multiplier": 1, "multiple_of": 256, "norm_eps": c["norm_eps"],
            "rope_theta": c["rope_theta"], "use_scaled_rope": False,
            "n_inject": a.n_inject,
            "n_target_layers": len(akw.get("dflash_target_layer_ids", [])),
            "block_size": a.block_size,
            "mask_token_id": akw.get("dflash_mask_token_id"),
            # ---- Qwen3.5 draft extensions ----
            "target_layer_ids": akw.get("dflash_target_layer_ids"),
            "causal_block_layers": akw.get("dflash_causal_layers"),
            "n_attn_masks": 2,
        }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(p, open(a.out, "w"), indent=1)
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
