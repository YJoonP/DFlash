# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DFlash speculative-decoding draft head as one fused graph per decoding step.

2026-09-15 KST — Qwen3.5-4B 포팅판. QNN_Exporter(mkjsym) 의 Qwen3-4B 용 dflash_draft.py 를
기준으로 다음을 일반화했다.

  * 타깃 hidden 입력 수가 5 로 고정돼 있던 것을 n_target_layers 개(Qwen3.5-4B: 8)로.
    KV 캐시는 반드시 *args 로 남아야 하므로(파이프라인이 placeholder 이름 args_<i> 로
    층 번호를 읽는다) hidden 입력은 h_0..h_{n-1} 이라는 이름 붙은 인자로 만든다.
    인자 수가 모델마다 다르므로 forward 는 클래스 생성 시 exec 로 만든다(make_dflash_model_class).
  * z-lab/Qwen3.5-4B-DFlash 는 6층인데 layer_types 가 sliding×5 + full×1 이고, 원본
    (dflash/model.py Qwen3DFlashAttention) 에서 sliding 층은 is_causal=True, full 층은
    is_causal=False 다. 즉 앞 5층은 블록 안에서 인과 마스크, 마지막 층은 블록 안 양방향이다.
    (sliding_window 4096 은 우리 문맥 1152 보다 커서 무의미.) 그래서 마스크 입력이 두 개다:
    atten_mask(블록 양방향, 원본 규약)과 atten_mask_causal(블록 안 인과). 런타임은 둘 다 채운다.
  * rope_theta 1e7, head_dim 128, q/k norm, 임베딩은 타깃 tied 테이블(그래프 밖).

원본 DFlash 구조(dflash_generate):
  block = [마지막 확정 토큰, MASK×(B-1)], 위치 start..start+B-1
  ctx   = 타깃 hidden(target_layer_ids 층 출력을 concat) -> fc -> hidden_norm, 위치 start-ctx..start-1
  각 층: Q 는 블록에서만, K/V 는 ctx 행과 블록 행 둘 다에서(같은 k_proj/v_proj), k_norm 은 둘 다,
        RoPE 는 각자 위치. 캐시에는 ctx 행만 남긴다(블록 행 K/V 는 버림).
  logits = lm_head(norm(h)) 를 블록 위치 1..B-1 에서 사용.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    PADDING_MASK_VALUE,
)

from .static_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    NORM_REGISTRY,
    repeat_kv,
)


class DFlashAttention(LlamaAttention):
    """LlamaAttention with a second, layer-invariant KV source (the projected target hidden)."""

    def _ctx_kv(self, g_ctx, freqs_cos_ctx, freqs_sin_ctx, conv: bool):
        bsz, ctx_len, _ = g_ctx.shape
        if conv:
            g = torch.reshape(g_ctx, (bsz, ctx_len, 1, self.dim)).transpose(1, 3)
            k = self.wk_conv(g).permute(0, 3, 1, 2).squeeze(-1)
            v = self.wv_conv(g).permute(0, 3, 1, 2).squeeze(-1)
        else:
            k = self.wk(g_ctx)
            v = self.wv(g_ctx)
        k = k.view(bsz, ctx_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, ctx_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm and self.qk_norm_before_rope:
            k = self.k_norm_fn(k)
        if self.use_rope:
            k = self.apply_rope_emb(k, freqs_cos_ctx, freqs_sin_ctx)
        if self.use_qk_norm and not self.qk_norm_before_rope:
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            k = torch.matmul(k, self.r3_weight)
        return k.transpose(2, 3), v

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        g_ctx: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        freqs_cos_ctx: torch.Tensor,
        freqs_sin_ctx: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        conv: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        if conv:
            h = torch.reshape(hidden_states, (bsz, seq_len, 1, self.dim)).transpose(1, 3)
            q = self.wq_conv(h).permute(0, 3, 1, 2).squeeze(-1)
            k = self.wk_conv(h).permute(0, 3, 1, 2).squeeze(-1)
            v = self.wv_conv(h).permute(0, 3, 1, 2).squeeze(-1)
        else:
            q = self.wq(hidden_states)
            k = self.wk(hidden_states)
            v = self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)
        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        k_ctx, v_ctx = self._ctx_kv(g_ctx, freqs_cos_ctx, freqs_sin_ctx, conv)

        # key order [ past cache | committed rows of this step | block ] — the caller lays the
        # masks out the same way and the runtime appends k_ctx/v_ctx to the cache.
        kh = torch.cat([k_caches, k_ctx, k], dim=-1)
        vh = torch.cat([v_caches, v_ctx, v], dim=2)

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            attn = torch.where(atten_mask == 0, attn, attn_min + (-20))
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2)
        if conv:
            y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
            y = self.wo_conv(y).transpose(1, 3).reshape(bsz, seq_len, -1)
        else:
            y = y.reshape(bsz, seq_len, -1)
            y = self.wo(y)

        # only the committed rows persist; the block's own K/V die with this call.
        # (names k_ctx / v_ctx matter: the wrapper finds cache outputs by the "k_"/"v_" prefix)
        return y, [k_ctx], [v_ctx]

    def forward(self, hidden_states, g_ctx, freqs_cos, freqs_sin, freqs_cos_ctx,
                freqs_sin_ctx, atten_mask, k_caches, v_caches):
        return self._forward_impl(hidden_states, g_ctx, freqs_cos, freqs_sin,
                                  freqs_cos_ctx, freqs_sin_ctx, atten_mask,
                                  k_caches, v_caches, conv=False)

    def forward_attention_conv(self, hidden_states, g_ctx, freqs_cos, freqs_sin,
                               freqs_cos_ctx, freqs_sin_ctx, atten_mask, k_caches, v_caches):
        return self._forward_impl(hidden_states, g_ctx, freqs_cos, freqs_sin,
                                  freqs_cos_ctx, freqs_sin_ctx, atten_mask,
                                  k_caches, v_caches, conv=True)


class DFlashDecoderLayer(LlamaDecoderLayer):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__(layer_idx, config, output_new_cache_only)
        self.attention = DFlashAttention(
            layer_idx=layer_idx, config=config, output_new_cache_only=output_new_cache_only
        )

    def forward(self, x, g_ctx, freqs_cos, freqs_sin, freqs_cos_ctx, freqs_sin_ctx,
                atten_mask, k_caches, v_caches):
        hidden_states = self.attention_norm(x)
        h, k_cache, v_cache = self.attention(
            hidden_states=hidden_states,
            g_ctx=g_ctx,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            freqs_cos_ctx=freqs_cos_ctx,
            freqs_sin_ctx=freqs_sin_ctx,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        h = x + h
        out = self.feed_forward(self.ffn_norm(h))
        return h + out, k_cache, v_cache


class DFlashDraftModelBase(LlamaModel):
    """One graph per decoding step: commit the accepted rows and propose the next block.

    ar_len = block width (B: one known token + B-1 mask positions). n_inject = fixed number of
    committed rows carried in alongside; the caller pads up to it and masks the remainder.
    """

    def __init__(self, config: ModelArgs, ar_len=1, output_new_cache_only=True,
                 output_cache=True, use_i64_token=False, n_inject=16,
                 n_target_layers=8, **kwargs):
        super().__init__(
            config=config, ar_len=ar_len, output_new_cache_only=output_new_cache_only,
            output_cache=output_cache, use_i64_token=use_i64_token, **kwargs
        )
        self.n_inject = n_inject
        self.n_target_layers = n_target_layers
        akw = getattr(config, "attention_kwargs", None) or {}
        self.causal_layers = set(akw.get("dflash_causal_layers", []))
        self.target_layer_ids = list(akw.get("dflash_target_layer_ids", []))
        self.block_size = int(akw.get("dflash_block_size", ar_len))
        self.mask_token_id = akw.get("dflash_mask_token_id", None)
        if self.target_layer_ids and len(self.target_layer_ids) != n_target_layers:
            raise ValueError(
                f"n_target_layers={n_target_layers} but config lists "
                f"{len(self.target_layer_ids)} target layer ids"
            )
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(i, config, output_new_cache_only) for i in range(config.n_layers)]
        )
        # one block per target layer, summed in forward: sum_i fc_i(h_i) == fc(cat(h_i)).
        # Each target layer then keeps its own activation scale (their ranges differ by 100x+).
        self.fc = nn.ModuleList(
            [nn.Linear(config.dim, config.dim, bias=False) for _ in range(n_target_layers)]
        )
        self.hidden_norm = NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)

    # ---- graph body ----------------------------------------------------------------
    def _forward_impl(self, target_hidden, block_embd, atten_mask, atten_mask_causal,
                      input_pos, input_pos_ctx, args):
        hidden_states = block_embd

        output_k_cache = []
        output_v_cache = []
        freqs_cos = self.freqs_cos[input_pos][0]
        freqs_sin = self.freqs_sin[input_pos][0]
        freqs_cos_ctx = self.freqs_cos[input_pos_ctx][0]
        freqs_sin_ctx = self.freqs_sin[input_pos_ctx][0]

        g_ctx = self.hidden_norm(
            sum(fc_i(h_i) for fc_i, h_i in zip(self.fc, target_hidden))
        )

        for ind, decoder_layer in enumerate(self.layers):
            hidden_states, k, v = decoder_layer(
                hidden_states,
                g_ctx=g_ctx,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                freqs_cos_ctx=freqs_cos_ctx,
                freqs_sin_ctx=freqs_sin_ctx,
                atten_mask=atten_mask_causal if ind in self.causal_layers else atten_mask,
                k_caches=args[ind],
                v_caches=args[self.n_layers + ind],
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        logits = self.output(self.norm(hidden_states))
        if self.logits_scaling:
            logits = logits / self.logits_scaling
        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    # ---- export helpers --------------------------------------------------------------
    @property
    def cache_len(self):
        # total keys per step = past cache + this step's committed rows + the block, and every
        # mask in this pipeline is max_context_len wide.
        return self.max_context_len - self.n_inject - self.ar_len

    def get_example_inputs(self):
        b, n_ctx = self.max_batch_size, self.n_inject
        target_hidden = [
            torch.randn((b, n_ctx, self.dim), dtype=torch.float32)
            for _ in range(self.n_target_layers)
        ]
        block_embd = torch.randn((b, self.ar_len, self.dim), dtype=torch.float32)
        atten_mask = AttentionMask(
            [
                CausalAttentionMask(b, self.ar_len, self.max_context_len),  # block: bidirectional
                CausalAttentionMask(b, self.ar_len, self.max_context_len),  # block: causal
            ]
        )
        pos = torch.zeros((b, self.ar_len), dtype=torch.int32)
        pos_ctx = torch.zeros((b, n_ctx), dtype=torch.int32)
        k_cache, v_cache = [], []
        for _ in range(self.n_layers):
            k_cache.append(torch.zeros(b, self.n_kv_heads, self.head_dim, self.cache_len))
            v_cache.append(torch.zeros(b, self.n_kv_heads, self.cache_len, self.head_dim))
        # Nested, in the shape the wrapper's export_input builder indexes positionally.
        return (target_hidden, block_embd, atten_mask, pos, pos_ctx, k_cache, v_cache)

    def get_attn_mask_template(self):
        return self.get_example_inputs()[2]

    def make_masks(self, n_ctx_valid: int, n_cache_valid: int, n_block: int = None):
        """Runtime-style masks for one step (float, 0 = attend, PADDING_MASK_VALUE = masked).

        key layout: [ cache (cache_len) | committed rows (n_inject) | block (ar_len) ]
        n_cache_valid: how many cache columns hold real rows (opened from the left),
        n_ctx_valid:   how many of this step's committed rows are real (opened from the left),
        n_block:       how many block positions are real (default: all).
        Returns (mask_bidirectional, mask_causal_in_block), each (1, 1, ar_len, max_context_len).
        """
        B, CL, CI, T = 1, self.cache_len, self.n_inject, self.ar_len
        n_block = T if n_block is None else n_block
        m = torch.full((B, 1, T, self.max_context_len), PADDING_MASK_VALUE)
        m[..., :n_cache_valid] = 0.0
        m[..., CL : CL + n_ctx_valid] = 0.0
        m[..., CL + CI : CL + CI + n_block] = 0.0
        mc = m.clone()
        blk = CL + CI
        for i in range(T):
            mc[..., i, blk + i + 1 : blk + T] = PADDING_MASK_VALUE
        if n_block < T:  # padded block rows attend nothing but themselves (keeps softmax finite)
            for i in range(n_block, T):
                m[..., i, :] = PADDING_MASK_VALUE
                m[..., i, blk + i] = 0.0
                mc[..., i, :] = PADDING_MASK_VALUE
                mc[..., i, blk + i] = 0.0
        return m, mc

    def get_metadata(self):
        meta = super().get_metadata()
        meta["n_inject"] = self.n_inject
        meta["n_target_layers"] = self.n_target_layers
        meta["block_size"] = self.block_size
        if self.mask_token_id is not None:
            meta["mask_token_id"] = int(self.mask_token_id)
        meta["dflash_causal_layers"] = ",".join(str(i) for i in sorted(self.causal_layers))
        meta["dflash_target_layer_ids"] = ",".join(str(i) for i in self.target_layer_ids)
        return meta


def _make_forward(n_target: int):
    """forward(self, h_0..h_{n-1}, block_embd, atten_mask, atten_mask_causal, input_pos,
    input_pos_ctx, *args) — named hidden inputs, KV caches as the varargs tail."""
    hs = ", ".join(f"h_{i}" for i in range(n_target))
    src = (
        f"def forward(self, {hs}, block_embd, atten_mask, atten_mask_causal, "
        f"input_pos, input_pos_ctx, *args):\n"
        f"    return self._forward_impl([{hs}], block_embd, atten_mask, atten_mask_causal, "
        f"input_pos, input_pos_ctx, args)\n"
    )
    ns = {}
    exec(src, ns)  # noqa: S102 — signature must carry the input names
    return ns["forward"]


_CLASS_CACHE = {}


def make_dflash_model_class(n_target: int):
    """A DFlashDraftModel subclass whose forward names exactly n_target hidden inputs."""
    if n_target not in _CLASS_CACHE:
        _CLASS_CACHE[n_target] = type(
            f"DFlashDraftModel{n_target}",
            (DFlashDraftModelBase,),
            {"forward": _make_forward(n_target), "__module__": __name__},
        )
    return _CLASS_CACHE[n_target]


DFlashDraftModel8 = make_dflash_model_class(8)
DFlashDraftModel5 = make_dflash_model_class(5)
