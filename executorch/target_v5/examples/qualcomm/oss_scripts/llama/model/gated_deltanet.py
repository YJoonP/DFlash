# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen3.5 Gated DeltaNet (linear attention) for the static QNN export path.
# (2026-09-01 port) Mirrors examples/models/llama/attention.py::AttentionGatedDeltaNet
# but with:
#   - explicit state I/O (conv_state / recurrent_state passed in and returned,
#     occupying the same per-layer cache slots that full-attention layers use
#     for k/v caches), instead of mutable module buffers
#   - the causal depthwise conv1d expressed as a depthwise conv2d
#     (QNN has no conv1d builder)
#   - softplus decomposed into where/log/exp/clamp (no QNN softplus builder)
#   - the delta-rule recurrence unrolled over ar_len steps (static graph)

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.llama.model_args import ModelArgs

# QNN_DELTA_KERNEL=1: 델타 규칙 재귀를 커스텀 HTP op (op_packages/DeltaRuleOpPackage) 하나로 내린다.
# 기본값 0 = 기존처럼 ar_len 만큼 펼친 elementwise 그래프.
_USE_DELTA_KERNEL = os.environ.get("QNN_DELTA_KERNEL", "0") == "1"
# QNN_DELTA_HEAD_GROUPS: 델타 규칙 op 을 head 축으로 몇 조각 낼지. 1 이면 안 자른다.
_DELTA_HEAD_GROUPS = int(os.environ.get("QNN_DELTA_HEAD_GROUPS", "1"))
_DECODE_KERNEL_MODE = os.environ.get("QNN_DELTA_DECODE", "off")
if _DECODE_KERNEL_MODE not in ("off", "packed", "unique"):
    raise ValueError("QNN_DELTA_DECODE must be off, packed, or unique")
if _DECODE_KERNEL_MODE != "off" and not _USE_DELTA_KERNEL:
    raise ValueError("QNN_DELTA_DECODE requires QNN_DELTA_KERNEL=1")
# Keep the shipped graph unchanged until device accuracy/performance validation.
_RMS_L2NORM_MODE = os.environ.get("QNN_L2NORM_RMS", "0")
if _RMS_L2NORM_MODE not in ("0", "1", "direct", "decode_direct"):
    raise ValueError("QNN_L2NORM_RMS must be 0, 1 (prescaled), direct, or decode_direct")
_USE_RMS_L2NORM = _RMS_L2NORM_MODE != "0"
# QNN_DELTA_MUTBUF=1: 상태를 KV 슬롯(그래프 입출력)이 아니라 mutable buffer 로 들고
# 제자리 갱신한다. 델리게이트가 메모리를 쥐고 입력과 출력을 같은 주소로 묶으므로
# 러너 복사(1.1ms)와 QNN 입력 복사(1.6~2.7ms)가 사라진다 (2026-09-17 KST).
_USE_MUTBUF = os.environ.get("QNN_DELTA_MUTBUF", "0") == "1"
# QNN_CONV_TMAJOR=1: conv 상태를 (B, K-1, C) 시간 우선으로 두고 depthwise causal conv 를
# 시간축으로 밀린 입력 x 채널별 가중치 K 개의 합으로 계산한다. (B, C, K) <-> (B, K, C) 전치
# (HTP 에서 층당 약 87 us 실제 복사)를 없애기 위함 (2026-09-28 KST). 기본값은 기존 동작.
_CONV_TMAJOR = os.environ.get("QNN_CONV_TMAJOR", "0") == "1"
if _USE_DELTA_KERNEL:
    from executorch.examples.qualcomm.oss_scripts.llama.model import (  # noqa: F401
        delta_rule_op,  # torch.ops.qwen35.delta_rule 등록
    )
    delta_rule_op.validate_package_configuration(_DELTA_HEAD_GROUPS)
# QNN_PREFILL_VALID=1: the prefill graph takes valid (B,T,1,1) and conv_idx (K-1,) int32 inputs so
# padded rows of the last prompt chunk touch neither the recurrent state (masked kernel) nor the
# conv state (index_select of the last K-1 valid columns) (2026-09-29 KST).
_PREFILL_VALID = os.environ.get("QNN_PREFILL_VALID", "0") == "1"
# QNN_DEFER_STATE=1 (2026-10-01 KST, DFlash verify graph; fork_dflash 27016ac re-expressed on the masked op):
#   a DeltaNet state cannot be rolled back after a rejected draft, so the ar_len>1 (prefill = verify)
#   graph commits the PREVIOUS chunk's accepted rows at the start of a call and computes the current
#   chunk from that committed state without committing it.
#     inputs  after the caches: valid_prev (B,T,1,1) int32 = previous chunk's accepted rows,
#             per DeltaNet layer the previous chunk's ingredients x (B,C,T), k (B,T,Hv,dk), v (B,T,Hv,dv),
#             beta (B,T,Hv,1), decay (B,T,Hv,1) (exactly the tensors the delta op consumed),
#             then valid_cur (B,T,1,1) int32 (rows of this chunk to compute) and
#             conv_idx_prev (K-1,) int32 = [a, a+1, a+2], a = accepted rows of the previous chunk
#     commit  conv: index_select(cat(conv_base, x_prev), conv_idx_prev) (same Gather as QNN_PREFILL_VALID);
#             state: masked delta op with valid_prev -> rejected rows are skipped exactly, accepted rows run
#             the same per-row integer step as a T=1 decode (v1p package with QNN_DELTA_STATE_SHARED_ENC=2)
#     block   current chunk: a second masked op per head group fed straight from the commit op's group
#             state (no cat/slice in between); its state output is discarded
#     outputs committed conv/recurrent state in the state slots, then this chunk's ingredients
#   T=1 graphs (decode, calibrate) keep immediate commits; forward() is unchanged.
_DEFER_STATE = os.environ.get("QNN_DEFER_STATE", "0") == "1"
# QNN_DEFER_CHECKPOINT_ROW=c (2026-10-12 23:4x KST (experiments/ckpt10_w16_20261012), port of experiments/ckpt5_w8_20261010 onto the exact-recompute source):
# the current chunk's masked op is split at row c (rows [0, c) from the committed state, rows [c, T) from that op's
# state); the state after row c of every head group becomes an extra graph output after all ingredients. When the next
# call accepts a >= c rows the runtime feeds that checkpoint as the base state and commits only rows [c, a); otherwise
# it commits rows [0, a) from the committed state as before. All ingredients are still output.
_DEFER_CHECKPOINT_ROW = int(os.environ.get("QNN_DEFER_CHECKPOINT_ROW", "0"))
if _DEFER_CHECKPOINT_ROW and not _DEFER_STATE:
    raise ValueError("QNN_DEFER_CHECKPOINT_ROW needs QNN_DEFER_STATE=1")
_STATE_SNAPSHOTS = os.environ.get("QNN_STATE_SNAPSHOTS", "0") == "1"
if _STATE_SNAPSHOTS and not _DEFER_STATE:
    raise ValueError("QNN_STATE_SNAPSHOTS needs QNN_DEFER_STATE=1")
N_INGREDIENTS = 5
if _DEFER_STATE and _PREFILL_VALID:
    raise ValueError("QNN_DEFER_STATE=1 and QNN_PREFILL_VALID=1 are exclusive (deferral already handles padding)")
if _DEFER_STATE and (_USE_MUTBUF or _CONV_TMAJOR):
    raise ValueError("QNN_DEFER_STATE=1 supports neither QNN_DELTA_MUTBUF nor QNN_CONV_TMAJOR")
if _PREFILL_VALID and not (_USE_DELTA_KERNEL and delta_rule_op.prefill_valid_enabled()):
    raise ValueError("QNN_PREFILL_VALID=1 requires QNN_DELTA_KERNEL=1")
if _DEFER_STATE and not _USE_DELTA_KERNEL:
    raise ValueError("QNN_DEFER_STATE=1 requires QNN_DELTA_KERNEL=1 (masked delta op)")
if _PREFILL_VALID and os.environ.get("QNN_DELTA_KERNEL_FLOAT_IO", "0") == "1":
    raise ValueError("QNN_PREFILL_VALID=1 needs uint16 state edges; unset QNN_DELTA_KERNEL_FLOAT_IO")


def _stable_softplus(x: torch.Tensor) -> torch.Tensor:
    # softplus(x) = relu(x) + log(1 + exp(-|x|)).
    # log(1 + exp(x)) 를 그대로 쓰면 exp 의 동적 범위가 [0, e^20] 까지 벌어져,
    # QNN 의 per-tensor 활성화 양자화에서 작은 성분이 전부 0 으로 뭉개지고
    # log(0) = -inf 가 되어 recurrent state 가 첫 스텝부터 NaN 이 된다.
    # 아래 형태는 exp 의 출력이 (0, 1], log 의 입력이 [1, 2] 로 묶여 양자화에 안정적이다.
    return torch.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))


# 헤드별 최댓값의 하한. 0 벡터인 헤드에서 0 나눗셈을 막기 위한 것으로,
# 정상 구간(실측 최소 |x|max ~ 0.08)은 건드리지 않는다.
_L2NORM_MIN_SCALE = 1e-4


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """헤드별 L2 정규화. 양자화를 견디도록 스케일 불변성을 이용해 재작성했다.

    소박하게 x / sqrt(sum(x^2)) 로 쓰면 QNN 에서 무너진다. 이유는 제곱이 동적 범위를
    제곱시키기 때문이다. 실측(Qwen3.5-4B layer0): 정규화 직전 key 의 헤드별 크기가
    100 배 차이나는데, 제곱하면 10,000 배가 된다. per-tensor 16bit(65,536 단계)로는
    담기지 않아 작은 헤드의 제곱합이 몇 단계로 뭉개진다
    (head18: sum(x^2)=0.074, 양자화 스텝 0.0123 → 6 단계 → 노름이 0.272 대신 0.111).
    입력 자체는 41.97 dB 로 멀쩡한데 정규화 출력만 -3.22 dB 가 되는 이유가 이것이다.

    제곱하기 전에 헤드별 최댓값으로 나눠 모든 헤드를 [-1, 1] 로 맞춘다.
    epsilon이 0일 때 L2 정규화는 스케일 불변이다. 현재처럼 스케일 후 epsilon을
    더하면 원래 x 기준 분모는 sqrt(sum(x*x) + eps*scale*scale)이므로,
    RMSNorm 치환에서도 사전 스케일과 epsilon의 위치를 함께 보존해야 한다.
    """
    scale = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=_L2NORM_MIN_SCALE)
    xs = x / scale  # 헤드별로 [-1, 1] — 이후 x*x, sum, sqrt 가 모두 균일한 범위
    return xs / torch.sqrt((xs * xs).sum(dim=-1, keepdim=True) + eps)


def _repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # (B, T, H, D) -> (B, T, H*n_rep, D), interleaved like repeat_interleave(dim=2).
    if n_rep == 1:
        return x
    b, t, h, d = x.shape
    return x[:, :, :, None, :].expand(b, t, h, n_rep, d).reshape(b, t, h * n_rep, d)


def _l2norm_rms(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    prescale: bool = True,
) -> torch.Tensor:
    """Fuse the square/reduce/add/sqrt/divide, preserving baseline prescaling.

    With z=x/clamp(amax(abs(x))), RMSNorm(z, w, eps/d) equals
    z/sqrt(sum(z*z)+eps) when w=1/sqrt(d). Query uses w=1/d to
    absorb its additional 1/sqrt(d). Removing prescaling would change
    both quantization behavior and the effective epsilon near zero. With
    prescale=False this instead matches the original, unscaled reference L2.
    """
    if prescale:
        scale = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=_L2NORM_MIN_SCALE)
        x = x / scale
    return F.rms_norm(x, (x.shape[-1],), weight, eps / x.shape[-1])


class GatedDeltaNetAttention(nn.Module):
    """Static-shape Gated DeltaNet block with explicit state I/O.

    forward(hidden_states, freqs_cos, freqs_sin, atten_mask, conv_state, recurrent_state)
        -> (y, [new_conv_state], [new_recurrent_state])

    conv_state:      (B, conv_dim, conv_kernel_size - 1)   last K-1 input columns
                     ((B, conv_kernel_size - 1, conv_dim) with QNN_CONV_TMAJOR=1)
    recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim)
    freqs_*/atten_mask are accepted (same layer interface as LlamaAttention) but unused.
    """

    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=True):
        super().__init__()
        self.delta_kernel = delta_rule_op.get_delta_rule_op() if _USE_DELTA_KERNEL else None
        self.decode_kernel = delta_rule_op.get_delta_decode_op() if _DECODE_KERNEL_MODE != "off" else None
        self.masked_kernel = (
            delta_rule_op.get_delta_rule_masked_op() if (_PREFILL_VALID or _DEFER_STATE) else None
        )
        self.decode_unique = _DECODE_KERNEL_MODE == "unique"
        del output_new_cache_only  # states are always full replacements
        self.layer_idx = layer_idx
        self.hidden_size = config.dim

        assert config.linear_num_key_heads is not None
        assert config.linear_num_value_heads is not None
        assert config.linear_key_head_dim is not None
        assert config.linear_value_head_dim is not None

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        if _USE_DELTA_KERNEL and self.num_v_heads % _DELTA_HEAD_GROUPS:
            raise ValueError("QNN_DELTA_HEAD_GROUPS must divide linear_num_value_heads")
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        assert self.num_v_heads % self.num_k_heads == 0
        self.head_repeat = self.num_v_heads // self.num_k_heads
        if self.decode_unique and self.num_k_heads % _DELTA_HEAD_GROUPS:
            raise ValueError("unique Q/K heads must divide evenly into the kernel groups")
        self.conv_dim = self.key_dim * 2 + self.value_dim
        if _USE_MUTBUF:
            bsz = getattr(config, 'max_batch_size', 1) or 1
            # persistent=True 여야 torch.export 가 mutable buffer 로 분류한다.
            # (non-persistent 는 constant 로 올라가 copy_ 갱신이 사라진다 - 2026-09-17 실측)
            # 체크포인트에는 없는 값이므로 아래 로드 훅이 0 으로 채워 넣는다.
            self.register_buffer(
                'conv_state_buf',
                torch.zeros(self._conv_state_shape(bsz)),
            )
            self.register_buffer(
                'recurrent_state_buf',
                torch.zeros(bsz, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            )
        self.norm_eps = config.norm_eps
        self.use_rms_l2norm = _USE_RMS_L2NORM
        self.prescale_rms_l2norm = _RMS_L2NORM_MODE not in ("direct", "decode_direct")
        self.rms_decode_only = _RMS_L2NORM_MODE == "decode_direct"
        if self.use_rms_l2norm:
            # Constants exported as RMSNorm weights, not checkpoint parameters.
            self.register_buffer(
                "query_l2_weight", torch.full((self.head_k_dim,), 1.0 / self.head_k_dim),
                persistent=False,
            )
            self.register_buffer(
                "key_l2_weight", torch.full((self.head_k_dim,), self.head_k_dim**-0.5),
                persistent=False,
            )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Depthwise causal conv. Checkpoint stores conv1d.weight (C, 1, K);
        # we run it as a conv2d with weight (C, 1, 1, K) — see the load hook below.
        self.conv1d = nn.Conv2d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=(1, self.conv_kernel_size),
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        # RMSNormGated weight (checkpoint key: attention.norm.weight)
        self.norm = nn.ParameterDict(
            {"weight": nn.Parameter(torch.ones(self.head_v_dim))}
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.register_load_state_dict_pre_hook(self._reshape_conv1d_weight_hook)
        # QNN_DEFER_STATE: only the ar_len>1 graph defers (config.ar_len is set by process_model_args)
        self.ar_len = int(getattr(config, "ar_len", 1) or 1)
        self.deferred = _DEFER_STATE and self.ar_len > 1

    @staticmethod
    def _reshape_conv1d_weight_hook(module, state_dict, prefix, *args, **kwargs):
        key = prefix + "conv1d.weight"
        if key in state_dict and state_dict[key].dim() == 3:
            state_dict[key] = state_dict[key].unsqueeze(2)  # (C,1,K) -> (C,1,1,K)
        # mutable buffer 모드의 상태 버퍼는 체크포인트에 없다. 0 으로 채워 strict 로드를
        # 통과시킨다 (모델 시작 상태는 어차피 0 이다).
        for name in ("conv_state_buf", "recurrent_state_buf"):
            ref = getattr(module, name, None)
            if ref is not None and prefix + name not in state_dict:
                state_dict[prefix + name] = torch.zeros_like(ref)

    def _conv_state_shape(self, bsz: int):
        if _CONV_TMAJOR:
            return (bsz, self.conv_kernel_size - 1, self.conv_dim)
        return (bsz, self.conv_dim, self.conv_kernel_size - 1)

    def ingredient_shapes(self, max_batch_size: int, ar_len: int = None):
        """QNN_DEFER_STATE ingredients of one chunk (x, k, v, beta, decay): the delta op's own inputs."""
        T = ar_len or self.ar_len
        b = max_batch_size
        return (
            (b, self.conv_dim, T),
            (b, T, self.num_v_heads, self.head_k_dim),
            (b, T, self.num_v_heads, self.head_v_dim),
            (b, T, self.num_v_heads, 1),
            (b, T, self.num_v_heads, 1),
        )

    def state_shapes(self, max_batch_size: int):
        return (
            self._conv_state_shape(max_batch_size),
            (max_batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: torch.Tensor,  # conv_state (slot A)
        v_caches: torch.Tensor,  # recurrent_state (slot B)
        valid: Optional[torch.Tensor] = None,  # (B, T, 1, 1) int32, prefill only
        conv_idx: Optional[torch.Tensor] = None,  # (K-1,) int32 = [n, n+1, ...], prefill only
        deferred: Optional[dict] = None,  # QNN_DEFER_STATE: valid_prev, ing_prev, valid_cur, conv_idx_prev
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        del freqs_cos, freqs_sin, atten_mask
        if deferred is not None:
            if not self.deferred or valid is not None:
                raise ValueError("deferred inputs need QNN_DEFER_STATE=1 on an ar_len>1 graph (no prefill valid)")
            return self.forward_deferred(hidden_states, k_caches, v_caches, **deferred)
        if (valid is None) != (conv_idx is None) or (valid is not None and self.masked_kernel is None):
            raise ValueError("valid and conv_idx go together and need QNN_PREFILL_VALID=1")
        if _USE_MUTBUF:
            conv_state = self.conv_state_buf
            recurrent_state = self.recurrent_state_buf
        else:
            conv_state = k_caches
            recurrent_state = v_caches
        bsz, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)  # (B, T, C)
        z = self.in_proj_z(hidden_states).reshape(
            bsz, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)  # (B, T, Hv)
        a = self.in_proj_a(hidden_states)  # (B, T, Hv)

        # ---- causal depthwise conv over time ----
        if _CONV_TMAJOR:
            # out[t] = sum_k w[:, k] * full[t + k]  (conv2d is cross-correlation)
            full = torch.cat([conv_state, mixed_qkv], dim=1)  # (B, K-1+T, C)
            # last K-1 rows, or the K-1 rows after the last valid row
            new_conv_state = full[:, seq_len:, :] if conv_idx is None else torch.index_select(full, 1, conv_idx)
            w = self.conv1d.weight.reshape(self.conv_dim, self.conv_kernel_size)
            conv_out = full[:, 0:seq_len, :] * w[:, 0]
            for k in range(1, self.conv_kernel_size):
                conv_out = conv_out + full[:, k : k + seq_len, :] * w[:, k]
            conv_out = F.silu(conv_out)  # (B, T, C)
        else:
            # conv1d as conv2d
            xt = mixed_qkv.transpose(1, 2)  # (B, C, T)
            full = torch.cat([conv_state, xt], dim=-1)  # (B, C, T+K-1)
            # last K-1 columns, or the K-1 columns after the last valid one
            new_conv_state = full[:, :, seq_len:] if conv_idx is None else torch.index_select(full, 2, conv_idx)
            conv_out = self.conv1d(full.unsqueeze(2))  # (B, C, 1, T)
            conv_out = F.silu(conv_out.squeeze(2)).transpose(1, 2)  # (B, T, C)

        query, key, value = torch.split(
            conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        # Direct RMS is per head: normalize each unique Q/K head just once.
        # Keep the calibrated graph order of existing export modes unchanged.
        use_rms = self.use_rms_l2norm and (not self.rms_decode_only or seq_len == 1)
        norm_before_repeat = use_rms and not self.prescale_rms_l2norm
        keep_unique = self.decode_kernel is not None and seq_len == 1 and self.decode_unique
        if not norm_before_repeat and not keep_unique:
            query = _repeat_heads(query, self.head_repeat)
            key = _repeat_heads(key, self.head_repeat)

        if use_rms:
            query = _l2norm_rms(
                query, self.query_l2_weight, prescale=self.prescale_rms_l2norm
            )
            key = _l2norm_rms(
                key, self.key_l2_weight, prescale=self.prescale_rms_l2norm
            )
        else:
            query = _l2norm(query) * (self.head_k_dim**-0.5)
            key = _l2norm(key)

        if norm_before_repeat and not keep_unique:
            query = _repeat_heads(query, self.head_repeat)
            key = _repeat_heads(key, self.head_repeat)

        beta = torch.sigmoid(b)  # (B, T, Hv)
        g = -torch.exp(self.A_log) * _stable_softplus(a + self.dt_bias)
        decay = torch.exp(g)  # (B, T, Hv), in (0, 1)

        # ---- delta rule, unrolled over seq_len (static graph) ----
        # 델타 규칙.  (B,Hv,dk,1) * (B,Hv,1,dv) 처럼 두 피연산자가 서로 다른 축으로
        # 동시에 브로드캐스트되는 곱은 HTP 가 처리하지 못한다(기기 실측: 헤드 0 만
        # 계산되고 헤드 1~31 이 상수로 채워짐).  matmul 로 바꾸면 그 문제는 사라지지만
        # HTP 는 16bit x 16bit matmul 을 지원하지 않아 두 번째 피연산자(= recurrent
        # state)가 8bit 로 떨어진다.  state 는 매 스텝 자기 자신으로 되먹임되므로
        # 8bit 화는 오차를 복리로 키운다.
        # → elementwise 를 유지하되, 이중 브로드캐스트만 expand 로 명시적으로 펼친다.
        #   (단일 축 브로드캐스트는 HTP 에서 정상 동작하므로 그대로 둔다)
        dk, dv = self.head_k_dim, self.head_v_dim
        if _USE_DELTA_KERNEL:
            delta_op = (self.decode_kernel if seq_len == 1 and self.decode_kernel is not None
                        else self.delta_kernel or torch.ops.qwen35.delta_rule.default)
            extra = ()
            if valid is not None:
                delta_op, extra = self.masked_kernel, (valid,)
            # 재귀 전체를 커널 하나로. state 는 커널 안에서 float 로만 갱신된다.
            b_ = beta.unsqueeze(-1)
            d_ = decay.unsqueeze(-1)
            if _DELTA_HEAD_GROUPS > 1 and self.num_v_heads % _DELTA_HEAD_GROUPS == 0:
                # head 축으로 쪼개 op 인스턴스를 여러 개 만든다. 32 개 head 는 서로
                # 완전히 독립이고(순차인 것은 t 뿐), QNN 스케줄러가 그룹들을
                # 병렬 실행할 수 있다. 패키지별 독립 workspace 검증은 위에서 한다.
                # state 는 head 축이 바깥이라 slice/concat 이 연속 구간 조작이다.
                g = self.num_v_heads // _DELTA_HEAD_GROUPS
                cores, states = [], []
                for gi in range(_DELTA_HEAD_GROUPS):
                    lo, hi = gi * g, (gi + 1) * g
                    qlo, qhi = (lo // self.head_repeat, hi // self.head_repeat) if keep_unique else (lo, hi)
                    c_i, s_i = delta_op(
                        query[:, :, qlo:qhi],
                        key[:, :, qlo:qhi],
                        value[:, :, lo:hi],
                        b_[:, :, lo:hi],
                        d_[:, :, lo:hi],
                        recurrent_state[:, lo:hi],
                        *extra,
                    )[:2]
                    cores.append(c_i)
                    states.append(s_i)
                core = torch.cat(cores, dim=2)
                state = torch.cat(states, dim=1)
            else:
                core, state = delta_op(
                    query, key, value, b_, d_, recurrent_state, *extra
                )[:2]
            return self._finish(core, z, state, new_conv_state, bsz, seq_len)
        state = recurrent_state  # (B, Hv, dk, dv)
        outs = []
        for t in range(seq_len):
            q_t = query[:, t].unsqueeze(-1)  # (B, Hv, dk, 1)
            k_t = key[:, t].unsqueeze(-1)  # (B, Hv, dk, 1)
            v_t = value[:, t]  # (B, Hv, dv)
            # decay 는 헤드당 스칼라 → dk, dv 두 축으로 퍼지므로 펼쳐 준다
            decay_t = decay[:, t].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dk, dv)
            beta_t = beta[:, t].unsqueeze(-1)  # (B, Hv, 1)

            state = state * decay_t
            kv_mem = (state * k_t).sum(dim=-2)  # (B, Hv, dv)
            delta = (v_t - kv_mem) * beta_t
            # 외적: 양쪽을 (B,Hv,dk,dv) 로 펼쳐 평범한 elementwise 곱으로 만든다
            state = state + k_t.expand(-1, -1, -1, dv) * delta.unsqueeze(-2).expand(
                -1, -1, dk, -1
            )
            outs.append((state * q_t).sum(dim=-2))  # (B, Hv, dv)

        core = torch.stack(outs, dim=1)  # (B, T, Hv, dv)
        return self._finish(core, z, state, new_conv_state, bsz, seq_len)

    # ------------------------------------------------------------------ deferred (QNN_DEFER_STATE)
    def _defer_groups(self):
        G = _DELTA_HEAD_GROUPS if _DELTA_HEAD_GROUPS > 1 and self.num_v_heads % _DELTA_HEAD_GROUPS == 0 else 1
        g = self.num_v_heads // G
        return [(gi * g, (gi + 1) * g) for gi in range(G)]

    def forward_deferred(self, hidden_states, conv_base, state_base, valid_prev, ing_prev, valid_cur,
                         conv_idx_prev):
        bsz, seq_len, _ = hidden_states.shape
        if seq_len != self.ar_len:
            raise ValueError(f"deferred graph built for ar_len {self.ar_len}, got {seq_len} rows")
        x_prev, k_prev, v_prev, b_prev, d_prev = ing_prev
        groups = self._defer_groups()

        if _STATE_SNAPSHOTS:
            conv_committed = conv_base
            committed = [state_base[:, lo:hi] for lo, hi in groups]
            state_committed = state_base
        else:
            # 1) commit the previous chunk's accepted rows
            full_prev = torch.cat([conv_base, x_prev], dim=-1)  # (B, C, K-1+T)
            conv_committed = torch.index_select(full_prev, 2, conv_idx_prev)  # (B, C, K-1)
            committed = []
            for lo, hi in groups:
                # query is unused here (the core output is dropped); its own slice keeps q and k separate tensors
                committed.append(
                    self.masked_kernel(
                        k_prev[:, :, lo:hi], k_prev[:, :, lo:hi], v_prev[:, :, lo:hi],
                        b_prev[:, :, lo:hi], d_prev[:, :, lo:hi], state_base[:, lo:hi], valid_prev,
                    )[1]
                )
            state_committed = torch.cat(committed, dim=1)

        # 2) the current chunk from the committed state (forward() math for seq_len > 1, non-TMAJOR conv)
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (B, T, C)
        z = self.in_proj_z(hidden_states).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # (B, T, Hv)
        a = self.in_proj_a(hidden_states)  # (B, T, Hv)
        xt = mixed_qkv.transpose(1, 2)  # (B, C, T)
        full = torch.cat([conv_committed, xt], dim=-1)  # (B, C, T+K-1)
        conv_out = self.conv1d(full.unsqueeze(2))  # (B, C, 1, T)
        conv_out = F.silu(conv_out.squeeze(2)).transpose(1, 2)  # (B, T, C)
        query, key, value = torch.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        use_rms = self.use_rms_l2norm and not self.rms_decode_only  # seq_len > 1
        norm_before_repeat = use_rms and not self.prescale_rms_l2norm
        if not norm_before_repeat:
            query = _repeat_heads(query, self.head_repeat)
            key = _repeat_heads(key, self.head_repeat)
        if use_rms:
            query = _l2norm_rms(query, self.query_l2_weight, prescale=self.prescale_rms_l2norm)
            key = _l2norm_rms(key, self.key_l2_weight, prescale=self.prescale_rms_l2norm)
        else:
            query = _l2norm(query) * (self.head_k_dim**-0.5)
            key = _l2norm(key)
        if norm_before_repeat:
            query = _repeat_heads(query, self.head_repeat)
            key = _repeat_heads(key, self.head_repeat)
        beta = torch.sigmoid(b)  # (B, T, Hv)
        g = -torch.exp(self.A_log) * _stable_softplus(a + self.dt_bias)
        decay = torch.exp(g)  # (B, T, Hv)
        b_, d_ = beta.unsqueeze(-1), decay.unsqueeze(-1)
        cores, histories, checkpoints = [], [], []
        c = _DEFER_CHECKPOINT_ROW
        if c and (_STATE_SNAPSHOTS or not 0 < c < seq_len):
            raise ValueError(f"QNN_DEFER_CHECKPOINT_ROW={c} needs the masked op and must lie inside the {seq_len}-row chunk")
        for (lo, hi), s_g in zip(groups, committed):
            if c:
                group_in = (query[:, :, lo:hi], key[:, :, lo:hi], value[:, :, lo:hi], b_[:, :, lo:hi], d_[:, :, lo:hi])
                head = self.masked_kernel(*(t[:, :c] for t in group_in), s_g, valid_cur[:, :c])
                tail = self.masked_kernel(*(t[:, c:] for t in group_in), head[1], valid_cur[:, c:])
                cores.append(torch.cat([head[0], tail[0]], dim=1))
                checkpoints.append(head[1])  # (B, Hg, dk, dv): state after row c of this head group
                continue
            op = torch.ops.qwen35.delta_rule_snapshots16.default if _STATE_SNAPSHOTS else self.masked_kernel
            result = op(query[:, :, lo:hi], key[:, :, lo:hi], value[:, :, lo:hi],
                        b_[:, :, lo:hi], d_[:, :, lo:hi], s_g, valid_cur)
            cores.append(result[0])
            if _STATE_SNAPSHOTS:
                histories.append(result[1])
        core = torch.cat(cores, dim=2)
        core = F.rms_norm(core, (self.head_v_dim,), self.norm["weight"], self.norm_eps)
        core = core * F.silu(z)
        y = self.out_proj(core.reshape(bsz, seq_len, self.value_dim))
        # 3) ingredients for the next call: the very tensors this chunk's delta op consumed
        if _STATE_SNAPSHOTS:
            return y, [full], [torch.cat(histories, dim=1)], []
        return y, [conv_committed], [state_committed], [xt, key, value, b_, d_] + checkpoints

    def _finish(self, core, z, state, new_conv_state, bsz, seq_len):
        # ---- gated RMS norm: rmsnorm(core) * silu(z), then out_proj ----
        core = F.rms_norm(core, (self.head_v_dim,), self.norm["weight"], self.norm_eps)
        core = core * F.silu(z)
        core = core.reshape(bsz, seq_len, self.value_dim)
        y = self.out_proj(core)

        if _USE_MUTBUF:
            self.conv_state_buf.copy_(new_conv_state)
            self.recurrent_state_buf.copy_(state)
            return y, [], []
        return y, [new_conv_state], [state]
