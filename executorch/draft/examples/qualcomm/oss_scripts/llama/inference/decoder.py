# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

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
    # QNN_DEFER_STATE (ar_len>1 DeltaNet graphs): previous-chunk acceptance mask + ingredients
    valid_prev: Optional[torch.Tensor] = None
    ingredients: Optional[List[torch.Tensor]] = None

    def __iter__(self):
        # Yield order must match the compiled model's input signature:
        # (input_ids | inputs_embeds, *atten_mask, [pos_ids], *k_caches, *v_caches[, valid_prev, *ing])
        yield self.input_ids if self.input_ids is not None else self.inputs_embeds
        yield from self.atten_mask
        yield self.pos_ids
        yield from self.k_caches
        yield from self.v_caches
        if self.valid_prev is not None:
            yield self.valid_prev
            yield from self.ingredients

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
    ):
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
        self.deferred = False
        self._valid_prev = None
        self._ingredients = None
        if len(example_inputs) == 7:
            # deferred-state graph: (tokens, attn_mask, pos_ids, k, v, valid_prev, ingredients)
            (_, self._default_attn_mask, _, self._k_caches, self._v_caches,
             self._valid_prev, self._ingredients) = example_inputs
            self.deferred = True
        else:
            assert len(example_inputs) == 5, (
                f"DecoderInference requires get_example_inputs to return "
                f"(tokens, attn_mask, pos_ids, k_caches, v_caches) (5 elements); "
                f"got {len(example_inputs)}.  Pass the with KV-cache version."
            )
            _, self._default_attn_mask, _, self._k_caches, self._v_caches = example_inputs

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
            valid_prev=self._valid_prev,
            ingredients=self._ingredients,
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

        assert self.layer_types is not None, "stepwise calibration needs layer_types"
        bsz = 1
        cl = self.max_context_len
        # Chunk length L = the graph's ar_len, derived from any full-attention
        # KV cache slot (cache columns = CL - L). Works for both the ar=1
        # calibrate graph and the ar=N prefill graph (self-calibration).
        full_idx = next(
            i for i, lt in enumerate(self.layer_types) if lt != "linear_attention"
        )
        L = cl - self._k_caches[full_idx].shape[-1]
        mask_obj = AttentionMask(CausalAttentionMask(bsz, L, cl))
        masks = list(mask_obj)
        kA = [t.clone() for t in self._k_caches]
        kB = [t.clone() for t in self._v_caches]
        # KV caches hold CL - L columns; only tokens whose positions fit there
        # can write their cache, so process complete chunks within that range.
        usable = cl - L
        seq = input_ids[0].tolist()[:usable]
        # deferred-state graph (QNN_DEFER_STATE, L>1): the previous chunk is committed inside
        # the next call. Calibrate with the real protocol: first call commits nothing
        # (valid_prev=0), later calls commit the full previous chunk (valid_prev=1).
        valid_prev = torch.zeros_like(self._valid_prev) if self.deferred else None
        ing = [t.clone() for t in self._ingredients] if self.deferred else []
        for start in range(0, len(seq) - L + 1, L):
            chunk = seq[start : start + L]
            tokens = torch.tensor([chunk], dtype=self._dtype)
            pos_t = torch.arange(
                start, start + L, dtype=torch.int32
            ).unsqueeze(0)
            if self.deferred:
                out = module(tokens, masks[0], pos_t, *kA, *kB, valid_prev, *ing)
            else:
                out = module(tokens, masks[0], pos_t, *kA, *kB)
            # exported modules preserve the eager output pytree
            # (logits, [k...], [v...][, [ing...]]); a flattened variant is also accepted.
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
            for i, lt in enumerate(self.layer_types):
                if lt == "linear_attention":
                    kA[i], kB[i] = newA[i], newB[i]
                else:
                    kA[i][:, :, :, start : start + L] = newA[i]
                    kB[i][:, :, start : start + L, :] = newB[i]
            if self.deferred:
                ing = [t.detach() for t in new_ing]
                valid_prev = torch.ones_like(self._valid_prev)
            mask_obj.smart_mask_update(start, L)

    def make_forward_fn(self) -> Callable:
        k = self._k_caches
        v = self._v_caches
        pos = self.pos_ids

        extra = ([self._valid_prev, *self._ingredients] if self.deferred else [])

        def _forward(
            model: Union[torch.nn.Module, torch.fx.GraphModule],
            model_inputs: dict,
        ) -> torch.Tensor:
            input_ids = model_inputs["input_ids"]
            attn_masks = model_inputs["attention_mask"]
            out = model(input_ids, *attn_masks, pos, *k, *v, *extra)
            return out[0] if isinstance(out, (tuple, list)) else out

        return _forward


class DFlashDecoderInference(DecoderInference):
    """Calibration driver for a DFlash draft head (Qwen3.5 port, 2026-09-15 KST).

    The draft's inputs are the TARGET model's hidden states, which this loop cannot produce, so
    they are replayed from a dump (dump_target_hidden.py). The dump keeps every prompt's token ids
    and the per-token target hidden rows aligned, which lets calibration mimic the deployed loop:

        cache <- committed rows (K/V projected from target hidden)   [written back every step]
        step  <- inject the next r rows (1..n_inject) + propose a block [tok, MASK*(B-1)]

    The block is calibrated as it is decoded (one real token, then mask tokens): a diffusion draft
    never sees a run of real text in its block, and the observers' ranges are copied straight onto
    the deployed graphs. r alternates between full rows (prompt injection) and short random runs
    (generation-time acceptance), so both fill patterns of the committed rows are observed.
    """

    dflash_calibration = True

    def __init__(self, get_example_inputs: Callable, hidden_dump: str, n_inject: int = 16,
                 n_target_layers: int = 8, mask_token_id: int = None, embedding_weight=None,
                 make_masks: Callable = None, max_context_len: int = 1024,
                 max_batch_size: int = 1, seed: int = 0, **kwargs):
        ex = get_example_inputs()
        assert len(ex) == 7, (
            "DFlashDecoderInference expects (target_hidden, block_embd, atten_mask, pos, pos_ctx, "
            f"k_caches, v_caches); got {len(ex)} elements."
        )
        target_hidden, block_embd, self._default_attn_mask, _, _, self._k_caches, self._v_caches = ex
        self.max_context_len = max_context_len
        self.max_batch_size = max_batch_size
        self.audio_token_id = None
        self.image_token_id = None
        self.layer_types = None
        self.stepwise_calibration = False
        self._dtype = torch.int32
        self.n_inject = n_inject
        self.n_target_layers = n_target_layers
        self.ar_len = block_embd.shape[1]
        self.dim = block_embd.shape[-1]
        self._mask_token_id = mask_token_id
        self._embedding_weight = embedding_weight
        self._make_masks = make_masks
        self._seed = seed
        blob = torch.load(hidden_dump, weights_only=False)
        if isinstance(blob, dict) and "entries" in blob:
            self._entries = blob["entries"]
            self._dump_layer_ids = blob.get("layer_ids")
        else:  # legacy flat dump: one big [rows, n*dim] tensor, no token ids
            H = blob["hidden"] if isinstance(blob, dict) else blob
            self._entries = [{"ids": None, "hidden": H}]
            self._dump_layer_ids = None
        rows = sum(e["hidden"].shape[0] for e in self._entries)
        hd = self._entries[0]["hidden"].shape[-1]
        assert hd == n_target_layers * self.dim, (
            f"dump rows have width {hd}, expected n_target_layers*dim={n_target_layers * self.dim}"
        )
        print(f"[DFlash calib] dump: {len(self._entries)} prompts, {rows} rows, "
              f"layer_ids={self._dump_layer_ids}, block={self.ar_len}, n_inject={n_inject}")

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        if self._embedding_weight is None:
            raise RuntimeError("DFlashDecoderInference needs the embedding table (host lookup)")
        return torch.nn.functional.embedding(ids, self._embedding_weight.detach()).float()

    def _split_hidden(self, rows: torch.Tensor):
        """[r, n*dim] -> n tensors [1, n_inject, dim], zero-padded past r."""
        r = rows.shape[0]
        out = []
        for i in range(self.n_target_layers):
            h = torch.zeros(1, self.n_inject, self.dim)
            h[0, :r] = rows[:, i * self.dim : (i + 1) * self.dim].float()
            out.append(h)
        return out

    @staticmethod
    def _unpack(out, n_layers):
        if len(out) == 3 and isinstance(out[1], (list, tuple)):
            return out[0], list(out[1]), list(out[2])
        logits, *caches = out
        return logits, caches[:n_layers], caches[n_layers:]

    @torch.no_grad()
    def calibrate_from_dump(self, module) -> None:
        import random

        rng = random.Random(self._seed)
        B, NI = self.ar_len, self.n_inject
        cache_len = self._k_caches[0].shape[-1]
        n_layers = len(self._k_caches)
        steps = 0
        for ei, e in enumerate(self._entries):
            H = e["hidden"]
            ids = e["ids"] if e["ids"] is not None else [0] * H.shape[0]
            n = min(len(ids), H.shape[0], cache_len - 1)
            if n < 2:
                continue
            k = [t.clone() for t in self._k_caches]
            v = [t.clone() for t in self._v_caches]
            cache_pos = 0
            pos = 0
            n_prompt_phase = max(1, int(n * 0.6))
            while pos < n:
                if pos < n_prompt_phase:
                    r = min(NI, n - pos)                  # prompt injection: full rows
                else:
                    r = min(rng.randint(1, max(1, min(NI, 8))), n - pos)  # generation-like runs
                start = pos + r                           # block begins at the next position
                if start + B > self.max_context_len:
                    break
                tok0 = ids[start] if start < n else ids[n - 1]
                blk_ids = [tok0] + [self._mask_token_id if self._mask_token_id is not None else tok0] * (B - 1)
                block_embd = self._embed(torch.tensor([blk_ids], dtype=torch.int64))
                m, mc = self._make_masks(n_ctx_valid=r, n_cache_valid=cache_pos)
                pos_ctx = torch.arange(pos, pos + NI, dtype=torch.int32).clamp(
                    max=self.max_context_len - 1).unsqueeze(0)
                pos_blk = torch.arange(start, start + B, dtype=torch.int32).unsqueeze(0)
                hs = self._split_hidden(H[pos : pos + r])
                out = module(*hs, block_embd, m, mc, pos_blk, pos_ctx, *k, *v)
                _, new_k, new_v = self._unpack(out, n_layers)
                for i in range(n_layers):
                    k[i][:, :, :, cache_pos : cache_pos + r] = new_k[i][:, :, :, :r].detach()
                    v[i][:, :, cache_pos : cache_pos + r, :] = new_v[i][:, :, :r, :].detach()
                cache_pos += r
                pos += r
                steps += 1
            if (ei + 1) % 5 == 0 or ei + 1 == len(self._entries):
                print(f"[DFlash calib] prompt {ei + 1}/{len(self._entries)} done, {steps} steps so far",
                      flush=True)

    @torch.no_grad()
    def predict_step(self, module, input_ids, attn_mask=None, hidden_states=(), tok_embedding=None):
        """Single block step with the first dump rows as context (used only by eval hooks)."""
        rows = self._entries[0]["hidden"][: self.n_inject]
        r = rows.shape[0]
        m, mc = self._make_masks(n_ctx_valid=r, n_cache_valid=0)
        B = input_ids.shape[-1]
        pos_blk = torch.arange(r, r + B, dtype=torch.int32).unsqueeze(0)
        pos_ctx = torch.arange(0, self.n_inject, dtype=torch.int32).unsqueeze(0)
        out = module(*self._split_hidden(rows), self._embed(input_ids.to(torch.int64)), m, mc,
                     pos_blk, pos_ctx, *self._k_caches, *self._v_caches)
        return out[0] if isinstance(out, (tuple, list)) else out
