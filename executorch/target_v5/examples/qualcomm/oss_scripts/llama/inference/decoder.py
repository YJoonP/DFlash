# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import os

import torch
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask


def merge_modality_embeddings(
    input_ids: torch.Tensor,
    hidden_states: Tuple[torch.Tensor, ...],
    tok_embedding: torch.nn.Module,
    token_id: int,
) -> torch.Tensor:
    """Merge vision/audio hidden states into token embeddings at the special token positions."""
    special_image_mask = (input_ids == token_id).unsqueeze(-1)
    inputs_embeds = tok_embedding(input_ids)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
        inputs_embeds.device
    )
    image_hidden_states = torch.cat(hidden_states, dim=1).to(
        inputs_embeds.device, inputs_embeds.dtype
    )
    return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)


@dataclass
class DecoderInputs:
    pos_ids: torch.Tensor
    atten_mask: AttentionMask
    k_caches: List[torch.Tensor]
    v_caches: List[torch.Tensor]
    input_ids: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    # QNN_PREFILL_VALID prefill graphs: (valid, conv_idx) after the caches.
    extras: List[torch.Tensor] = field(default_factory=list)

    def __iter__(self):
        # Yield order must match the compiled model's input signature:
        # (input_ids | inputs_embeds, *atten_mask, [pos_ids], *k_caches, *v_caches, *extras)
        yield self.input_ids if self.input_ids is not None else self.inputs_embeds
        yield from self.atten_mask
        yield self.pos_ids
        yield from self.k_caches
        yield from self.v_caches
        yield from self.extras

    def __post_init__(self):
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be set.")


class DecoderInference:
    """
    Single forward pass execution for LLM/MLLM decoders.

    Holds the execution context (pos_ids, KV cache placeholders, default
    attention mask) and handles multimodal embedding merging.

    tok_embedding is NOT stored here — it is passed at call time so callers
    always use the live (possibly re-prepared) module rather than a stale
    reference captured at construction.
    """

    def __init__(
        self,
        get_example_inputs: Callable,
        audio_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
        max_context_len: int = 1024,
        max_batch_size: int = 1,
        use_i64_token: bool = False,
        layer_types: Optional[list] = None,
        stepwise_calibration: bool = False,
        defer_n_delta: int = 0,
    ):
        # QNN_DEFER_STATE prefill (verify) graph: number of DeltaNet layers; extras are then
        # [valid_prev, (x,k,v,beta,decay) x defer_n_delta, valid_cur, conv_idx_prev].
        self.defer_n_delta = int(defer_n_delta)
        # Qwen3.5 hybrid: which cache slots are DeltaNet states (full replace)
        # vs KV caches (append at position).
        self.layer_types = layer_types
        self.stepwise_calibration = stepwise_calibration
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.max_context_len = max_context_len
        self.max_batch_size = max_batch_size
        self.pos_ids = (
            torch.arange(max_context_len, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, -1)
            .contiguous()
        )
        self._dtype = torch.int64 if use_i64_token else torch.int32
        # Always require the with-KV-cache variant of get_example_inputs so
        # pos_ids and KV cache placeholders are available.  This is required
        # for encoding override: scale/zp collected on the CALIBRATE graph
        # must be copied to the DECODE graph, and both graphs share the same
        # input signature when KV caches are present.
        example_inputs = get_example_inputs()
        assert len(example_inputs) in (5, 6), (
            f"DecoderInference requires get_example_inputs to return "
            f"(tokens, attn_mask, pos_ids, k_caches, v_caches[, extras]) (5 or 6 elements); "
            f"got {len(example_inputs)}.  Pass the with KV-cache version."
        )
        _, self._default_attn_mask, _, self._k_caches, self._v_caches = example_inputs[:5]
        # All-valid prefill inputs: the masked graph then computes exactly the unmasked math.
        self._extras = list(example_inputs[5]) if len(example_inputs) == 6 else []

    def get_inputs(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None,
        hidden_states: Tuple[torch.Tensor, ...] = (),
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ) -> "DecoderInputs":
        if attn_mask is None:
            attn_mask = self._default_attn_mask
        raw_input_ids: Optional[torch.Tensor] = None
        inputs_embeds: Optional[torch.Tensor] = None
        if all(
            (
                hidden_states,
                tok_embedding,
                self.audio_token_id or self.image_token_id,
            )
        ):
            inputs_embeds = merge_modality_embeddings(
                input_ids,
                hidden_states,
                tok_embedding,
                self.audio_token_id or self.image_token_id,
            )
        else:
            raw_input_ids = input_ids
        return DecoderInputs(
            input_ids=raw_input_ids,
            inputs_embeds=inputs_embeds,
            pos_ids=self.pos_ids,
            atten_mask=attn_mask,
            k_caches=self._k_caches,
            v_caches=self._v_caches,
            extras=self._extras,
        )

    @torch.no_grad()
    def predict_step(
        self,
        module: Union[torch.nn.Module, torch.fx.GraphModule],
        input_ids: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None,
        hidden_states: Tuple[torch.Tensor, ...] = (),
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ) -> torch.Tensor:
        """Single forward pass through the decoder; returns logits."""
        logits, *_ = module(
            *self.get_inputs(input_ids, attn_mask, hidden_states, tok_embedding)
        )
        return logits

    @torch.no_grad()
    def predict_sequence_stepwise(
        self,
        module,
        input_ids: torch.Tensor,
    ) -> None:
        """Calibrate an ar_len=1 graph by stepping over the sequence,
        carrying KV caches, DeltaNet states, and the attention mask —
        mirroring the on-device decode loop. Used when calibrate_ar1=True."""
        from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
            AttentionMask,
            CausalAttentionMask,
        )

        # Pure-attention models (e.g. Qwen3 with calibrate_ar1, 2026-10-01) carry no layer_types:
        # every cache slot is then a KV cache. Hybrid models (Qwen3.5) pass their layer_types.
        layer_types = self.layer_types or ["full_attention"] * len(self._k_caches)
        bsz = 1
        cl = self.max_context_len
        # Chunk length L = the graph's ar_len, derived from any full-attention
        # KV cache slot (cache columns = CL - L). Works for both the ar=1
        # calibrate graph and the ar=N prefill graph (self-calibration).
        # 층 -> 슬롯 지도. QNN_DELTA_MUTBUF=1 이면 DeltaNet 층은 상태를 mutable buffer 로
        # 들고 있어 캐시 슬롯을 받지 않는다 (2026-09-17 KST).
        delta_mutbuf = os.environ.get("QNN_DELTA_MUTBUF", "0") == "1"
        slot_of, slot = [], 0
        for lt in layer_types:
            if delta_mutbuf and lt == "linear_attention":
                slot_of.append(None)
                continue
            slot_of.append(slot)
            slot += 1
        full_idx = next(
            i for i, lt in enumerate(layer_types) if lt != "linear_attention"
        )
        L = cl - self._k_caches[slot_of[full_idx]].shape[-1]
        mask_obj = AttentionMask(CausalAttentionMask(bsz, L, cl))
        masks = list(mask_obj)
        kA = [t.clone() for t in self._k_caches]
        kB = [t.clone() for t in self._v_caches]
        # KV caches hold CL - L columns; only tokens whose positions fit there
        # can write their cache, so process complete chunks within that range.
        usable = cl - L
        seq = input_ids[0].tolist()[:usable]
        # QNN_DEFER_STATE: run the real protocol. The first call commits nothing (valid_prev = 0,
        # conv_idx_prev = [0, 1, 2], zero ingredients); every later call commits the whole previous
        # chunk (valid_prev = 1, conv_idx_prev = [L, L+1, L+2]) from the ingredients it returned.
        defer = self.defer_n_delta > 0
        if defer:
            nd = self.defer_n_delta
            ex = self._extras
            assert len(ex) == 3 + 5 * nd, (len(ex), nd)
            n_conv = ex[2 + 5 * nd].numel()
            cur_extras = [torch.zeros_like(ex[0]), *[torch.zeros_like(t) for t in ex[1 : 1 + 5 * nd]],
                          ex[1 + 5 * nd], torch.arange(0, n_conv, dtype=torch.int32)]
        for start in range(0, len(seq) - L + 1, L):
            chunk = seq[start : start + L]
            tokens = torch.tensor([chunk], dtype=self._dtype)
            pos_t = torch.arange(
                start, start + L, dtype=torch.int32
            ).unsqueeze(0)
            out = module(tokens, masks[0], pos_t, *kA, *kB, *(cur_extras if defer else self._extras))
            # exported modules preserve the eager output pytree
            # (logits, [k...], [v...][, [ingredients...]]); a flattened variant is also accepted.
            new_ing = []
            if len(out) in (3, 4) and isinstance(out[1], (list, tuple)):
                newA, newB = list(out[1]), list(out[2])
                if len(out) == 4:
                    new_ing = list(out[3])
            else:
                _, *caches = out
                newA = caches[: len(kA)]
                newB = caches[len(kA) : 2 * len(kA)]
                new_ing = list(caches[2 * len(kA) :])
            if defer and os.environ.get("QNN_STATE_SNAPSHOTS") != "1":
                # QNN_DEFER_CHECKPOINT_ROW (2026-10-12 23:4x KST (experiments/ckpt10_w16_20261012)): per-group checkpoint states follow the 5 * nd ingredients;
                # this host calibration protocol commits whole chunks from the committed state and ignores them.
                n_ckpt = len(new_ing) - 5 * nd
                assert n_ckpt == 0 or (int(os.environ.get("QNN_DEFER_CHECKPOINT_ROW", "0")) and n_ckpt % nd == 0), (
                    len(new_ing), nd)
                new_ing = new_ing[: 5 * nd]
                cur_extras = [torch.ones_like(ex[0]), *[t.detach() for t in new_ing], ex[1 + 5 * nd],
                              torch.arange(L, L + n_conv, dtype=torch.int32)]
            for i, lt in enumerate(layer_types):
                j = slot_of[i]
                if j is None:
                    continue
                if lt == "linear_attention":
                    if defer and os.environ.get("QNN_STATE_SNAPSHOTS") == "1":
                        kA[j] = newA[j][:, :, -n_conv:].clone()
                        kB[j] = newB[j][-1:].clone()
                    else:
                        kA[j], kB[j] = newA[j], newB[j]
                else:
                    kA[j][:, :, :, start : start + L] = newA[j]
                    kB[j][:, :, start : start + L, :] = newB[j]
            mask_obj.smart_mask_update(start, L)

    def make_forward_fn(self) -> Callable:
        k = self._k_caches
        v = self._v_caches
        pos = self.pos_ids
        extras = self._extras

        def _forward(
            model: Union[torch.nn.Module, torch.fx.GraphModule],
            model_inputs: dict,
        ) -> torch.Tensor:
            input_ids = model_inputs["input_ids"]
            attn_masks = model_inputs["attention_mask"]
            out = model(input_ids, *attn_masks, pos, *k, *v, *extras)
            return out[0] if isinstance(out, (tuple, list)) else out

        return _forward
