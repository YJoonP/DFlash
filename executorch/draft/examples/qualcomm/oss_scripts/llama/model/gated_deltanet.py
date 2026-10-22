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
#
# (2026-09-15 KST) 추측 디코딩용 "지연 커밋(deferred state commit)" 모드 추가 — QNN_DEFER_STATE=1.
#   DeltaNet 상태는 KV 캐시처럼 위치별로 되돌릴 수 없다. T 토큰을 검증한 뒤 앞 k+1 개만 수락되면
#   상태는 "k+1 토큰까지" 로 되돌아가야 하는데, 수락 길이는 그래프가 끝나야 안다.
#   그래서 ar_len>1 그래프는 상태를 **한 청크 늦게** 커밋한다:
#     입력: S_base(직전 청크 이전 상태), 직전 청크의 재료(x_prev, k_prev, v_prev, beta_prev, decay_prev),
#           valid_prev(직전 청크에서 수락된 위치 = 1)
#     계산: S_committed = DeltaRule(재료 ⊙ valid_prev, S_base)  — 거부 위치는 beta=0, decay=1 로 무효화
#           conv_committed = cat(conv_base, x_prev) 의 마지막 유효 K-1 열 (선택 행렬)
#           현재 청크는 S_committed/conv_committed 에서 계산, 그 상태 결과는 버린다
#     출력: S_committed, conv_committed (상태 슬롯), 현재 청크의 재료(다음 호출 입력)
#   T=1 그래프(decode, calibrate)는 기존 즉시 커밋 그대로다.

import os
from typing import List, Tuple

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
# QNN_DEFER_STATE=1: ar_len>1 그래프에서 상태를 한 청크 늦게 커밋한다 (추측 디코딩 검증용).
DEFER_STATE = os.environ.get("QNN_DEFER_STATE", "0") == "1"
# conv 상태 커밋 위치: graph = 그래프 안 선택 행렬(기본), runtime = 런타임이 x_prev 로 직접 조립.
DEFER_CONV_SELECT = os.environ.get("QNN_DEFER_CONV_SELECT", "graph")
if DEFER_CONV_SELECT not in ("graph", "runtime"):
    raise ValueError("QNN_DEFER_CONV_SELECT must be graph or runtime")
# 층당 재료 텐서 수 (x, k, v, beta, decay). runtime 모드에서는 x 가 입력에 없다(출력만).
N_INGREDIENTS = 5
if _USE_DELTA_KERNEL:
    from executorch.examples.qualcomm.oss_scripts.llama.model import (  # noqa: F401
        delta_rule_op,  # torch.ops.qwen35.delta_rule 등록
    )
    delta_rule_op.validate_package_configuration(_DELTA_HEAD_GROUPS)


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
    recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim)
    freqs_*/atten_mask are accepted (same layer interface as LlamaAttention) but unused.

    forward_deferred(...) — QNN_DEFER_STATE=1, ar_len>1 그래프 전용 (파일 머리말 참고).
    """

    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=True):
        super().__init__()
        self.delta_kernel = delta_rule_op.get_delta_rule_op() if _USE_DELTA_KERNEL else None
        self.decode_kernel = delta_rule_op.get_delta_decode_op() if _DECODE_KERNEL_MODE != "off" else None
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

        # ---- deferred-commit constants (ar_len>1 graphs only) ----
        self.ar_len = int(getattr(config, "ar_len", 1) or 1)
        self.deferred = DEFER_STATE and self.ar_len > 1
        if self.deferred and DEFER_CONV_SELECT == "graph":
            L = self.conv_kernel_size - 1 + self.ar_len
            # inclusive cumsum as a matmul with a constant (QNN has no cumsum builder):
            # cum[t] = sum_{s<=t} v[s]  <=>  v @ M with M[s,t] = 1 for s <= t (upper triangular)
            self.register_buffer(
                "defer_tril", torch.triu(torch.ones(L, L)), persistent=False
            )
            self.register_buffer(
                "defer_ones", torch.ones(1, 1, self.conv_kernel_size - 1), persistent=False
            )

    @staticmethod
    def _reshape_conv1d_weight_hook(module, state_dict, prefix, *args, **kwargs):
        key = prefix + "conv1d.weight"
        if key in state_dict and state_dict[key].dim() == 3:
            state_dict[key] = state_dict[key].unsqueeze(2)  # (C,1,K) -> (C,1,1,K)

    def state_shapes(self, max_batch_size: int):
        return (
            (max_batch_size, self.conv_dim, self.conv_kernel_size - 1),
            (max_batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )

    def ingredient_shapes(self, max_batch_size: int, ar_len: int = None):
        """Deferred-commit ingredient tensors of one chunk: (x, k_unique, v, beta, decay)."""
        T = ar_len or self.ar_len
        b = max_batch_size
        return (
            (b, self.conv_dim, T),
            (b, T, self.num_k_heads, self.head_k_dim),
            (b, T, self.num_v_heads, self.head_v_dim),
            (b, T, self.num_v_heads, 1),
            (b, T, self.num_v_heads, 1),
        )

    # ------------------------------------------------------------------ pieces
    def _project(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (B, T, C)
        z = self.in_proj_z(hidden_states).reshape(
            bsz, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)  # (B, T, Hv)
        a = self.in_proj_a(hidden_states)  # (B, T, Hv)
        return mixed_qkv, z, b, a

    def _conv_qkv(self, mixed_qkv, conv_state, keep_unique):
        """causal depthwise conv + split + L2 norm. Returns (query, key, value, xt, new_conv_state)."""
        bsz, seq_len, _ = mixed_qkv.shape
        xt = mixed_qkv.transpose(1, 2)  # (B, C, T)
        full = torch.cat([conv_state, xt], dim=-1)  # (B, C, T+K-1)
        new_conv_state = full[:, :, seq_len:]  # last K-1 columns
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
        return query, key, value, xt, new_conv_state

    def _gates(self, b, a):
        beta = torch.sigmoid(b)  # (B, T, Hv)
        g = -torch.exp(self.A_log) * _stable_softplus(a + self.dt_bias)
        decay = torch.exp(g)  # (B, T, Hv), in (0, 1)
        return beta, decay

    def _run_delta(self, query, key, value, beta, decay, recurrent_state, keep_unique=False, state_only=False):
        """델타 규칙 재귀. beta/decay: (B,T,Hv). Returns (core (B,T,Hv,dv), state)."""
        seq_len = query.shape[1]
        dk, dv = self.head_k_dim, self.head_v_dim
        if _USE_DELTA_KERNEL:
            from executorch.examples.qualcomm.oss_scripts.llama.model import delta_rule_op
            commit_only = state_only and not delta_rule_op.recurrent_state_is_quantized()
            delta_op = (self.decode_kernel if seq_len == 1 and self.decode_kernel is not None
                        else self.delta_kernel or torch.ops.qwen35.delta_rule.default)
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
                input_states = delta_rule_op.split_recurrent_state(recurrent_state, _DELTA_HEAD_GROUPS)
                for gi in range(_DELTA_HEAD_GROUPS):
                    lo, hi = gi * g, (gi + 1) * g
                    qlo, qhi = (lo // self.head_repeat, hi // self.head_repeat) if keep_unique else (lo, hi)
                    if commit_only:
                        s_i = torch.ops.qwen35.delta_commit.default(
                            key[:, :, qlo:qhi], value[:, :, lo:hi], b_[:, :, lo:hi],
                            d_[:, :, lo:hi], input_states[gi])
                    else:
                        c_i, s_i = delta_op(
                            query[:, :, qlo:qhi], key[:, :, qlo:qhi], value[:, :, lo:hi],
                            b_[:, :, lo:hi], d_[:, :, lo:hi], input_states[gi])[:2]
                        cores.append(c_i)
                    states.append(s_i)
                core = None if commit_only else torch.cat(cores, dim=2)
                state = delta_rule_op.join_recurrent_state(states)
            elif commit_only:
                core = None
                state = torch.ops.qwen35.delta_commit.default(key, value, b_, d_, recurrent_state)
            else:
                core, state = delta_op(query, key, value, b_, d_, recurrent_state)[:2]
            return core, state
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
        return core, state

    # ------------------------------------------------------------------ immediate
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: torch.Tensor,  # conv_state (slot A)
        v_caches: torch.Tensor,  # recurrent_state (slot B)
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        del freqs_cos, freqs_sin, atten_mask
        conv_state = k_caches
        recurrent_state = v_caches
        bsz, seq_len, _ = hidden_states.shape

        mixed_qkv, z, b, a = self._project(hidden_states)
        keep_unique = self.decode_kernel is not None and seq_len == 1 and self.decode_unique
        query, key, value, _, new_conv_state = self._conv_qkv(mixed_qkv, conv_state, keep_unique)
        beta, decay = self._gates(b, a)

        # ---- delta rule ----
        # 델타 규칙.  (B,Hv,dk,1) * (B,Hv,1,dv) 처럼 두 피연산자가 서로 다른 축으로
        # 동시에 브로드캐스트되는 곱은 HTP 가 처리하지 못한다(기기 실측: 헤드 0 만
        # 계산되고 헤드 1~31 이 상수로 채워짐).  matmul 로 바꾸면 그 문제는 사라지지만
        # HTP 는 16bit x 16bit matmul 을 지원하지 않아 두 번째 피연산자(= recurrent
        # state)가 8bit 로 떨어진다.  state 는 매 스텝 자기 자신으로 되먹임되므로
        # 8bit 화는 오차를 복리로 키운다.
        # → elementwise 를 유지하되, 이중 브로드캐스트만 expand 로 명시적으로 펼친다.
        #   (단일 축 브로드캐스트는 HTP 에서 정상 동작하므로 그대로 둔다)
        core, state = self._run_delta(query, key, value, beta, decay, recurrent_state, keep_unique)
        return self._finish(core, z, state, new_conv_state, bsz, seq_len)

    # ------------------------------------------------------------------ deferred
    def _commit_conv(self, conv_state_base, x_prev, valid_prev):
        """cat(conv_base, x_prev) 에서 마지막 유효 K-1 열을 고른다 (그래프 안).

        valid_prev (B,T,1,1) 0/1. 옛 상태 열 K-1 개는 항상 유효. cum = 포함 누적합(상수 행렬과 matmul),
        대상 열 j 는 cum == n_valid-(K-2)+j 인 유효 열이다. 정수 차이를 clamp(3-4|d|,0,1) 로 1/0 으로 만들어
        16bit 양자화 잡음에 견디게 하고, 열마다 가중합(원소곱 + ReduceSum)으로 뽑는다.

        선택을 matmul(full, sel) 로 쓰면 annotate_kv_8bit 가 그것을 어텐션(Q@K)으로 보고 sel 경로를
        8bit 로 묶어 valid_prev 가 8bit 가 된다 -> beta/decay 마스킹 곱과 concat 이 u8/u16 혼합으로 HTP
        검증에서 거부된다 (2026-09-15 diag8 export 실측: DeltaNet 6층 x (mul 2 + cat 1) = 18개).
        누적합 matmul 은 두 번째 피연산자가 상수(get_attr)라 그 규칙에 걸리지 않는다.
        """
        bsz, C, T = x_prev.shape
        Km1 = self.conv_kernel_size - 1
        full = torch.cat([conv_state_base, x_prev], dim=-1)  # (B, C, Km1+T)
        v = valid_prev.reshape(bsz, 1, T)
        vf = torch.cat([self.defer_ones.expand(bsz, 1, Km1), v], dim=-1)  # (B, 1, Km1+T)
        cum = torch.matmul(vf, self.defer_tril)  # (B, 1, Km1+T) inclusive cumsum
        n_valid = cum[:, :, -1:]  # (B, 1, 1)
        cols = []
        for j in range(Km1):
            d = cum - (n_valid - float(Km1 - 1 - j))  # (B, 1, Km1+T), 0 at the target column
            w = vf * torch.clamp(3.0 - 4.0 * d.abs(), 0.0, 1.0)  # (B, 1, Km1+T) one-hot
            cols.append((full * w).sum(dim=-1, keepdim=True))  # (B, C, 1)
        return torch.cat(cols, dim=-1)  # (B, C, Km1)

    def forward_deferred(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: torch.Tensor,  # conv_state_base (slot A) — 'runtime' 모드에서는 이미 커밋된 conv 상태
        v_caches: torch.Tensor,  # recurrent_state_base (slot B)
        valid_prev: torch.Tensor,  # (B, T, 1, 1)
        x_prev: torch.Tensor,  # (B, C, T)     'runtime' 모드에서는 None
        k_prev: torch.Tensor,  # (B, T, Hk, dk) unique heads, post-norm
        v_prev: torch.Tensor,  # (B, T, Hv, dv)
        beta_prev: torch.Tensor,  # (B, T, Hv, 1)
        decay_prev: torch.Tensor,  # (B, T, Hv, 1)
    ):
        del freqs_cos, freqs_sin, atten_mask
        bsz, seq_len, _ = hidden_states.shape
        assert seq_len == self.ar_len, (seq_len, self.ar_len)

        # 1) 직전 청크 커밋 (수락된 위치만)
        if DEFER_CONV_SELECT == "graph":
            conv_committed = self._commit_conv(k_caches, x_prev, valid_prev)
        else:
            conv_committed = k_caches
        k_prev_rep = _repeat_heads(k_prev, self.head_repeat)
        beta_m = (beta_prev * valid_prev).squeeze(-1)  # (B, T, Hv)
        decay_m = (1.0 - valid_prev * (1.0 - decay_prev)).squeeze(-1)  # 거부 위치: decay 1 (무효)
        _, state_committed = self._run_delta(
            k_prev_rep, k_prev_rep, v_prev, beta_m, decay_m, v_caches, state_only=True
        )

        # 2) 현재 청크 (커밋된 상태에서 계산; 그 상태 결과는 버린다)
        mixed_qkv, z, b, a = self._project(hidden_states)
        query, key, value, xt, _ = self._conv_qkv(mixed_qkv, conv_committed, keep_unique=False)
        beta, decay = self._gates(b, a)
        core, _state_after = self._run_delta(query, key, value, beta, decay, state_committed)
        y, _, _ = self._finish(core, z, state_committed, conv_committed, bsz, seq_len)

        # 3) 다음 호출이 커밋할 재료. key 는 반복 헤드에서 unique 헤드만 (interleaved: h*n_rep).
        ing_x_cur = xt
        ing_k_cur = key[:, :, :: self.head_repeat] if self.head_repeat > 1 else key
        ing_v_cur = value
        ing_beta_cur = beta.unsqueeze(-1)
        ing_decay_cur = decay.unsqueeze(-1)
        return (
            y,
            [conv_committed],
            [state_committed],
            [ing_x_cur, ing_k_cur, ing_v_cur, ing_beta_cur, ing_decay_cur],
        )

    def _finish(self, core, z, state, new_conv_state, bsz, seq_len):
        # ---- gated RMS norm: rmsnorm(core) * silu(z), then out_proj ----
        core = F.rms_norm(core, (self.head_v_dim,), self.norm["weight"], self.norm_eps)
        core = core * F.silu(z)
        core = core.reshape(bsz, seq_len, self.value_dim)
        y = self.out_proj(core)

        return y, [new_conv_state], [state]
