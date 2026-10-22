# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen3.5 Gated DeltaNet 의 델타 규칙 재귀를 하나의 커스텀 연산으로 묶는다.
#
# 왜 묶느냐:
#   현재 static 그래프는 이 재귀를 ar_len(프리필 32, 디코드 1) 만큼 풀어서 펼친다.
#   한 스텝이 5개 텐서 연산이고 state 가 (1,32,128,128)=524,288 원소이므로,
#   프리필 그래프에는 (1,32,128,128) 크기의 중간 텐서가 32*5 = 160 개 생긴다.
#   또한 매 스텝의 state 가 그래프 텐서라서 스텝마다 16bit 로 재양자화되고,
#   state 는 자기 자신으로 되먹임되므로 그 오차가 T 스텝 동안 복리로 커진다.
#
#   융합은 그래프 중간 텐서와 반복 재양자화를 줄인다. 하지만 그래프 경계가 매번
#   DDR 왕복을 뜻하지는 않는다. 기존 패키지의 scalar 입출력과 전역 scratch는
#   QHPI 패키지의 HVX 입출력 및 독립 출력 workspace로 개선했다.
#   float 상태 누산의 정확도 이득과 기기 실행 속도는 별도로 검증해야 한다.

from typing import Tuple

import operator
import os

import torch
from torch.library import Library, impl

delta_rule_lib = Library("qwen35", "DEF")

# query/key 는 이미 l2norm 과 스케일링이 끝난 값, decay/beta 도 이미 exp/sigmoid 가 끝난 값이다.
# 커널은 순수하게 재귀만 담당한다 (앞단 전처리는 QNN 기본 연산이 잘 처리하므로 그대로 둔다).
delta_rule_lib.define(
    "delta_rule("
    "Tensor query, Tensor key, Tensor value, "
    "Tensor beta, Tensor decay, Tensor state_in"
    ") -> (Tensor, Tensor)"
)


def _delta_rule_reference(
    query: torch.Tensor,  # (B, T, Hv, dk)
    key: torch.Tensor,  # (B, T, Hv, dk)
    value: torch.Tensor,  # (B, T, Hv, dv)
    beta: torch.Tensor,  # (B, T, Hv, 1)   QNN 커스텀 op 가 4D 를 요구해 마지막 축을 붙였다
    decay: torch.Tensor,  # (B, T, Hv, 1)
    state_in: torch.Tensor,  # (B, Hv, dk, dv)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """gated_deltanet.py 의 펼친 루프와 원소 단위로 동일한 참조 구현.

    커널의 정답지이자, export 시 그래프에 남는 연산의 shape/dtype 전파용이다.
    """
    _, seq_len, _, _ = query.shape
    dk = query.shape[-1]
    dv = value.shape[-1]
    state = state_in
    outs = []
    for t in range(seq_len):
        q_t = query[:, t].unsqueeze(-1)  # (B, Hv, dk, 1)
        k_t = key[:, t].unsqueeze(-1)  # (B, Hv, dk, 1)
        v_t = value[:, t]  # (B, Hv, dv)
        decay_t = decay[:, t].unsqueeze(-1).expand(-1, -1, dk, dv)  # (B,Hv,1)->(B,Hv,dk,dv)
        beta_t = beta[:, t]  # (B, Hv, 1)

        state = state * decay_t
        kv_mem = (state * k_t).sum(dim=-2)  # (B, Hv, dv)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.expand(-1, -1, -1, dv) * delta.unsqueeze(-2).expand(
            -1, -1, dk, -1
        )
        outs.append((state * q_t).sum(dim=-2))  # (B, Hv, dv)
    return torch.stack(outs, dim=1), state


@impl(delta_rule_lib, "delta_rule", dispatch_key="CompositeExplicitAutograd")
def delta_rule_impl(query, key, value, beta, decay, state_in):
    return _delta_rule_reference(query, key, value, beta, decay, state_in)


delta_rule_lib.define(
    "delta_rule.out("
    "Tensor query, Tensor key, Tensor value, "
    "Tensor beta, Tensor decay, Tensor state_in, "
    "*, Tensor(a!) out, Tensor(b!) state_out"
    ") -> (Tensor(a!), Tensor(b!))"
)


@impl(delta_rule_lib, "delta_rule.out", dispatch_key="CompositeExplicitAutograd")
def delta_rule_out_impl(
    query, key, value, beta, decay, state_in, *, out, state_out
):
    o, s = _delta_rule_reference(query, key, value, beta, decay, state_in)
    out.copy_(o)
    state_out.copy_(s)
    return out, state_out


# Explicit private workspace: QNN owns one float head (64 KiB for dk=dv=128)
# per op. It is not a global buffer and is not exported as model state I/O.
delta_rule_lib.define(
    "delta_rule_state16(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in) -> (Tensor, Tensor, Tensor)"
)


@impl(delta_rule_lib, "delta_rule_state16", dispatch_key="CompositeExplicitAutograd")
def delta_rule_state16_impl(query, key, value, beta, decay, state_in):
    output, state = _delta_rule_reference(query, key, value, beta, decay, state_in)
    return output, state, state[-1:, -1:].clone()


delta_rule_lib.define(
    "delta_rule_state16.out(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in, *, Tensor(a!) out, Tensor(b!) state_out, "
    "Tensor(c!) workspace) -> (Tensor(a!), Tensor(b!), Tensor(c!))"
)


@impl(delta_rule_lib, "delta_rule_state16.out", dispatch_key="CompositeExplicitAutograd")
def delta_rule_state16_out_impl(query, key, value, beta, decay, state_in, *, out, state_out, workspace):
    output, state, work = delta_rule_state16_impl(query, key, value, beta, decay, state_in)
    out.copy_(output)
    state_out.copy_(state)
    workspace.copy_(work)
    return out, state_out, workspace


# Prompt padding: the prefill graph always has T rows, but only the first n are prompt tokens.
# valid (B,T,1,1) int32 marks them. A row with valid == 0 leaves the state unchanged and emits a
# zero output row, so a padded call computes exactly what a call on the valid rows would.
delta_rule_lib.define(
    "delta_rule_state16_masked(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in, Tensor valid) -> (Tensor, Tensor, Tensor)"
)


def _masked_reference(query, key, value, beta, decay, state_in, valid):
    dk, dv = query.shape[-1], value.shape[-1]
    state = state_in
    outs = []
    for t in range(query.shape[1]):
        keep = valid[:, t].reshape(-1, 1, 1, 1) != 0
        q_t = query[:, t].unsqueeze(-1)
        k_t = key[:, t].unsqueeze(-1)
        stepped = state * decay[:, t].unsqueeze(-1).expand(-1, -1, dk, dv)
        delta = (value[:, t] - (stepped * k_t).sum(dim=-2)) * beta[:, t]
        stepped = stepped + k_t.expand(-1, -1, -1, dv) * delta.unsqueeze(-2).expand(-1, -1, dk, -1)
        state = torch.where(keep, stepped, state)
        o_t = (state * q_t).sum(dim=-2)
        outs.append(torch.where(keep.reshape(-1, 1, 1), o_t, torch.zeros_like(o_t)))
    return torch.stack(outs, dim=1), state


@impl(delta_rule_lib, "delta_rule_state16_masked", dispatch_key="CompositeExplicitAutograd")
def delta_rule_state16_masked_impl(query, key, value, beta, decay, state_in, valid):
    output, state = _masked_reference(query, key, value, beta, decay, state_in, valid)
    return output, state, state[-1:, -1:].clone()


delta_rule_lib.define(
    "delta_rule_state16_masked.out(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in, Tensor valid, *, Tensor(a!) out, Tensor(b!) state_out, "
    "Tensor(c!) workspace) -> (Tensor(a!), Tensor(b!), Tensor(c!))"
)


@impl(delta_rule_lib, "delta_rule_state16_masked.out", dispatch_key="CompositeExplicitAutograd")
def delta_rule_state16_masked_out_impl(query, key, value, beta, decay, state_in, valid, *, out, state_out,
                                       workspace):
    output, state, work = delta_rule_state16_masked_impl(query, key, value, beta, decay, state_in, valid)
    out.copy_(output)
    state_out.copy_(state)
    workspace.copy_(work)
    return out, state_out, workspace


# Snapshot output replaces the final-state output; float reference is for PTQ only.
delta_rule_lib.define(
    "delta_rule_snapshots16(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in, Tensor valid) -> (Tensor, Tensor, Tensor)"
)

@impl(delta_rule_lib, "delta_rule_snapshots16", dispatch_key="CompositeExplicitAutograd")
def delta_rule_snapshots16_impl(query, key, value, beta, decay, state_in, valid):
    outs, states = [], []
    state = state_in
    for t in range(query.shape[1]):
        o, state = _masked_reference(query[:, t:t+1], key[:, t:t+1], value[:, t:t+1],
                                    beta[:, t:t+1], decay[:, t:t+1], state, valid[:, t:t+1])
        outs.append(o)
        states.append(state)
    B, T, H, _ = query.shape
    history = torch.stack(states, dim=1).reshape(B*T, H, query.shape[-1], value.shape[-1])
    return torch.cat(outs, dim=1), history, query.new_empty((1, 1, 128, 64))


def _delta_decode_reference(query, key, value, beta, decay, state_in):
    """One step, packed state boundary, optional grouped-query inputs."""
    if query.ndim != 4 or value.ndim != 4 or query.shape[1] != 1 or value.shape[1] != 1:
        raise ValueError("DeltaRuleDecode16 requires T=1 and rank-4 inputs")
    b, _, hq, dk = query.shape
    hv, dv = value.shape[2:]
    if not b or not hq or not hv or hv % hq or key.shape != query.shape or value.shape[0] != b:
        raise ValueError("value heads must be divisible by matching query/key heads")
    if beta.shape != (b, 1, hv, 1) or decay.shape != beta.shape or state_in.shape != (b, hv, dk, dv):
        raise ValueError("DeltaRuleDecode16 state/gate shapes do not match value heads")
    if not 0 < dk <= 128 or dv not in (64, 128):
        raise ValueError("DeltaRuleDecode16 supports 1..128 key dimensions and 64/128 value dimensions")
    # CPU reference only: the DSP kernel indexes unique heads without copying.
    repeat = hv // hq
    return _delta_rule_reference(query.repeat_interleave(repeat, dim=2),
                                 key.repeat_interleave(repeat, dim=2),
                                 value, beta, decay, state_in)


delta_rule_lib.define(
    "delta_rule_decode16(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in) -> (Tensor, Tensor)"
)


@impl(delta_rule_lib, "delta_rule_decode16", dispatch_key="CompositeExplicitAutograd")
def delta_rule_decode16_impl(query, key, value, beta, decay, state_in):
    return _delta_decode_reference(query, key, value, beta, decay, state_in)


delta_rule_lib.define(
    "delta_rule_decode16.out(Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor decay, Tensor state_in, *, Tensor(a!) out, Tensor(b!) state_out) "
    "-> (Tensor(a!), Tensor(b!))"
)


@impl(delta_rule_lib, "delta_rule_decode16.out", dispatch_key="CompositeExplicitAutograd")
def delta_rule_decode16_out_impl(query, key, value, beta, decay, state_in, *, out, state_out):
    output, state = _delta_decode_reference(query, key, value, beta, decay, state_in)
    out.copy_(output)
    state_out.copy_(state)
    return out, state_out


# ---------------------------------------------------------------------------
# QNN 통합 헬퍼. llama.py(컴파일 스펙) 와 llm_wrappers.py(양자화 어노테이션) 가 쓴다.
# ---------------------------------------------------------------------------
OP_PACKAGE_DIR = os.environ.get(
    "QNN_DELTA_OPPKG_DIR", "DeltaRuleOpPackage"
)
# 기기에서 러너가 .so 를 찾을 디렉터리 (run_ppl_qwen35.sh 의 EW 워크스페이스와 같아야 한다)
OP_PACKAGE_DEVICE_DIR = os.environ.get(
    "QNN_DELTA_OPPKG_DEVICE_DIR", "/data/local/tmp/executorch/static_llm"
)


def package_capabilities():
    import json
    from pathlib import Path

    path = Path(OP_PACKAGE_DIR) / "config/delta_rule_capabilities.json"
    return json.loads(path.read_text()) if path.exists() else {}


def get_delta_rule_op():
    if os.environ.get("QNN_STATE_IO_16BIT", "0") == "1" and package_capabilities().get("state16_workspace_op"):
        return torch.ops.qwen35.delta_rule_state16.default
    return torch.ops.qwen35.delta_rule.default


def get_delta_decode_op():
    if os.environ.get("QNN_STATE_IO_16BIT", "0") != "1" or not package_capabilities().get("decode16_unique_op"):
        raise ValueError("decode specialization requires uint16 state and a decode16_unique_op package")
    return torch.ops.qwen35.delta_rule_decode16.default


def prefill_valid_enabled():
    flag = os.environ.get("QNN_PREFILL_VALID", "0")
    if flag not in ("0", "1"):
        raise ValueError(f"QNN_PREFILL_VALID must be 0 or 1, got {flag!r}")
    return flag == "1"


def defer_state_enabled():
    """QNN_DEFER_STATE=1: DFlash verify graph with deferred commits through the masked op (2026-10-01)."""
    flag = os.environ.get("QNN_DEFER_STATE", "0")
    if flag not in ("0", "1"):
        raise ValueError(f"QNN_DEFER_STATE must be 0 or 1, got {flag!r}")
    return flag == "1"


def masked_op_needed():
    return prefill_valid_enabled() or defer_state_enabled()


def get_delta_rule_masked_op():
    if os.environ.get("QNN_STATE_IO_16BIT", "0") != "1" or not package_capabilities().get("state16_masked_op"):
        raise ValueError("QNN_PREFILL_VALID=1 requires QNN_STATE_IO_16BIT=1 and a state16_masked_op package")
    return torch.ops.qwen35.delta_rule_state16_masked.default


def validate_package_configuration(head_groups=1):
    capabilities = package_capabilities()
    if masked_op_needed() and not capabilities.get("state16_masked_op"):
        raise ValueError("QNN_PREFILL_VALID=1 / QNN_DEFER_STATE=1 require a package with state16_masked_op (QNN_DELTA_OPPKG_DIR)")
    if head_groups < 1:
        raise ValueError("QNN_DELTA_HEAD_GROUPS must be positive")
    if head_groups != 1 and not capabilities.get("independent_head_workspace", False):
        raise ValueError(
            "Head groups require a package with independently owned workspace. "
            "The legacy package shares global scratch; build qhpi/build.sh into "
            "a separate directory and set QNN_DELTA_OPPKG_DIR to that directory."
        )
    if (capabilities.get("implementation") == "qhpi"
            and os.environ.get("QNN_STATE_IO_16BIT", "0") == "1"
            and "uint16" not in capabilities.get("state_dtypes", [])):
        raise ValueError("This QHPI build requires QNN_STATE_IO_16BIT=0; its state edges are float32")


def get_op_package_options(htp_arch: str = "79"):
    """DeltaRule op package 를 컴파일 스펙에 등록할 옵션과, 기기에 push 할 .so 목록."""
    from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchOpPackagePlatform,
        QnnExecuTorchOpPackageTarget,
    )

    validate_package_configuration(int(os.environ.get("QNN_DELTA_HEAD_GROUPS", "1")))
    xml = f"{OP_PACKAGE_DIR}/config/delta_rule_op_package_htp.xml"
    op_map = {"DeltaRule": torch.ops.qwen35.delta_rule.default}
    if package_capabilities().get("state16_workspace_op"):
        op_map["DeltaRuleState16"] = torch.ops.qwen35.delta_rule_state16.default
    if package_capabilities().get("decode16_unique_op"):
        op_map["DeltaRuleDecode16"] = torch.ops.qwen35.delta_rule_decode16.default
    if masked_op_needed():
        op_map["DeltaRuleState16Masked"] = torch.ops.qwen35.delta_rule_state16_masked.default
    if os.environ.get("QNN_STATE_SNAPSHOTS") == "1":
        if not package_capabilities().get("snapshots_workspace_v2"):
            raise ValueError("Snapshot export requires the snapshot op package")
        op_map["DeltaRuleSnapshots16"] = torch.ops.qwen35.delta_rule_snapshots16.default
    b = QnnCustomOpPackageBuilder(
        xml_path=xml,
        torch_op_name_map=op_map,
    )
    lib = f"libQnn{b.op_package_name}"
    b.register_implementation(
        target=QnnExecuTorchOpPackageTarget.HTP,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{OP_PACKAGE_DEVICE_DIR}/{lib}_HTP.so",
    )
    b.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{OP_PACKAGE_DEVICE_DIR}/{lib}.so",
    )
    b.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=f"{OP_PACKAGE_DIR}/build/x86_64-linux-clang/{lib}.so",
    )
    device_files = [
        (f"{OP_PACKAGE_DIR}/build/hexagon-v{htp_arch}/{lib}.so", f"{lib}_HTP.so"),
        (f"{OP_PACKAGE_DIR}/build/aarch64-android/{lib}.so", f"{lib}.so"),
    ]
    return b.get_op_package_options(), device_files


def build_annotation_fn(act_observer=None):
    """16bit activations; state follows the selected package and I/O mode."""
    from executorch.backends.qualcomm.custom_op.annotator import (
        CustomOpsQuantAnnotator,
        IOQuantConfig,
    )
    from executorch.backends.qualcomm.quantizer.qconfig import get_16a4w_qnn_ptq_config
    from torchao.quantization.pt2e.observer import MinMaxObserver

    cfg = get_16a4w_qnn_ptq_config(act_observer=act_observer or MinMaxObserver)
    # state 입출력 dtype 은 export 설정을 따른다.
    #   QNN_STATE_IO_16BIT=1 -> state_out 도 16bit 로 관측해 자기 인코딩을 갖는다.
    #     (러너가 uint16 상태 버퍼를 그 zero_point 로 채워야 하므로 실제 관측값이 필요하다)
    #   =0 -> state_out 은 float 유지. output_qspec=None + _annotated=True = 관측기 없음.
    state_16 = os.environ.get("QNN_STATE_IO_16BIT", "0") == "1"
    state_spec = cfg.output_activation if state_16 else None
    ann = CustomOpsQuantAnnotator()
    ann.register_annotation(
        torch.ops.qwen35.delta_rule.default,
        IOQuantConfig(
            input_quant_specs={i: cfg.input_activation for i in range(5)},  # state_in(5) 제외
            # 튜플 출력: 스펙이 2개 이상이어야 어노테이터가 getitem 사용자에 붙인다.
            output_quant_specs={0: cfg.output_activation, 1: state_spec},
        ),
    )
    ann.register_annotation(
        torch.ops.qwen35.delta_rule_state16.default,
        IOQuantConfig(
            input_quant_specs={i: cfg.input_activation for i in range(6)},
            output_quant_specs={0: cfg.output_activation, 1: cfg.output_activation, 2: None},
        ),
    )
    ann.register_annotation(
        torch.ops.qwen35.delta_rule_state16_masked.default,
        # valid (index 6) is int32: the annotator skips non-float inputs.
        IOQuantConfig(
            input_quant_specs={i: cfg.input_activation for i in range(6)},
            output_quant_specs={0: cfg.output_activation, 1: cfg.output_activation, 2: None},
        ),
    )
    ann.register_annotation(
        torch.ops.qwen35.delta_rule_snapshots16.default,
        IOQuantConfig(input_quant_specs={i: cfg.input_activation for i in range(6)},
                      output_quant_specs={0: cfg.output_activation, 1: cfg.output_activation, 2: None}),
    )
    ann.register_annotation(
        torch.ops.qwen35.delta_rule_decode16.default,
        IOQuantConfig(input_quant_specs={i: cfg.input_activation for i in range(6)},
                      output_quant_specs={0: cfg.output_activation, 1: cfg.output_activation}),
    )
    # 레시피가 먼저 getitem 을 기본 16bit 로 붙이지만, 커스텀 어노테이션이 뒤에 돌며
    # 위 스펙(None 포함)으로 덮어쓴다 — annotator.py 가 명시적 None 도 기록하도록
    # 고친 뒤부터 성립한다 (2026-09-10). 그 전에는 여기서 수동으로 되돌려야 했다.
    annotate = ann.build_annotation_fn()
    if state_16:
        return annotate

    def annotate_float_state(gm):
        import operator
        from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
        from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

        annotate(gm)
        # Head partitioning adds slice/cat nodes. The generic recipe would
        # quantize these state-only transport nodes even with float state I/O.
        transport = {torch.ops.aten.slice.Tensor, torch.ops.aten.cat.default}
        visited = set()

        def keep_float(node):
            if node in visited or node.target not in transport:
                return
            visited.add(node)
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(_annotated=True)
            for arg in node.all_input_nodes:
                keep_float(arg)

        for node in gm.graph.nodes:
            if node.target != torch.ops.qwen35.delta_rule.default:
                continue
            keep_float(node.args[5])
            for user in node.users:
                if user.target == operator.getitem and user.args[1] == 1:
                    for consumer in user.users:
                        if consumer.target in transport:
                            keep_float(consumer)

    return annotate_float_state


STATE16_OPS = (torch.ops.qwen35.delta_rule_state16.default, torch.ops.qwen35.delta_rule_state16_masked.default, torch.ops.qwen35.delta_rule_snapshots16.default)


def shared_state_encoding_mode():
    """QNN_DELTA_STATE_SHARED_ENC: 0 off, 1 = state_out takes the op's state_in encoding (per head
    group), 2 = every state edge of the op (head-group slice -> op -> cat) takes the graph IO state
    encoding, so slice and cat carry no requantization at all."""
    flag = os.environ.get("QNN_DELTA_STATE_SHARED_ENC", "0")
    if flag not in ("0", "1", "2"):
        raise ValueError(f"QNN_DELTA_STATE_SHARED_ENC must be 0, 1 or 2, got {flag!r}")
    return int(flag)


STATE_IO_FIXED = set()   # QNN_STATE_IO_FROM_Q: state input dequantizes moved onto their quantize's encoding


def _io_state_encoding(state_dq, q, dq):
    """Graph-level encoding behind an op's state_in: through the head-group slice when present."""
    state_q = state_dq.args[0]
    if not isinstance(state_q, torch.fx.Node) or state_q.target is not q:
        raise ValueError(f"state_in {state_dq.name} is not fed by a per-tensor quantize")
    src = state_q.args[0]
    if src.target in (torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor):
        io = src.args[0]
        if not isinstance(io, torch.fx.Node) or io.target is not dq:
            raise ValueError(f"head-group slice {src.name} does not read a per-tensor dequantize")
        # 2026-10-13 10:3x KST (exact_oneenc_20261013): QNN_STATE_IO_FROM_Q=1. In the full32 exports the state_rec input
        # Q of layers 16 and 21 carries the overridden IO encoding while 3 of its 4 dequantizes (groups 0-2) kept the
        # calibrated one (scale 2.386371e-05 vs 2.386369e-05, offset 22049 vs 22048), so those groups requantized the
        # state on every call and their checkpoints came out on a grid 1 code off (runner restore inexact at the bottom
        # of the range). The stored input encoding is the Q's: take it, and put the dequantize on it as well.
        io_q = io.args[0]
        if (os.environ.get("QNN_STATE_IO_FROM_Q", "0") == "1" and isinstance(io_q, torch.fx.Node)
                and io_q.target is q):
            if tuple(io.args[1:3]) != tuple(io_q.args[1:3]):
                io.args = (io.args[0], *io_q.args[1:3], *io.args[3:])
                STATE_IO_FIXED.add(io.name)
            return state_q, (io_q.args[1], io_q.args[2])
        return state_q, (io.args[1], io.args[2])
    # QNN_DEFER_STATE: the verify op reads the commit op's group state directly
    # (getitem(state16 op, 1) -> q -> dq); its IO encoding is the commit op's state_in IO encoding.
    if (src.target is operator.getitem and len(src.args) > 1 and src.args[1] == 1
            and isinstance(src.args[0], torch.fx.Node) and src.args[0].target in STATE16_OPS):
        producer_in = src.args[0].args[5]
        if isinstance(producer_in, torch.fx.Node) and producer_in.target is dq:
            return state_q, _io_state_encoding(producer_in, q, dq)[1]
    return state_q, (state_q.args[1], state_q.args[2])


def force_shared_state_encoding(gm, mode=1):
    """QNN_DELTA_STATE_SHARED_ENC (after calibration and IO overrides).
    mode 1: give every uint16-state delta op's state_out the encoding of its state_in. The op then
    writes the state on the grid it read it from, which the integer decode kernel requires. With
    head groups the slice (IO -> group) and cat (group -> IO) still requantize.
    mode 2: state_in edge and state_out both take the graph IO state encoding, so the state is
    rounded once per step (in the kernel) and slice/cat are pure data movement.
    Returns (ops forced, ops whose calibrated state_in already equalled the IO encoding)."""
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    forced = already_io = 0
    for node in gm.graph.nodes:
        if node.target not in STATE16_OPS:
            continue
        src = node.args[5]
        if not isinstance(src, torch.fx.Node) or src.target is not dq:
            raise ValueError(f"{node.name}: state_in is not a per-tensor dequantize ({src})")
        scale, zp = src.args[1], src.args[2]
        state_q, io = _io_state_encoding(src, q, dq)
        already_io += (scale, zp) == io
        if mode == 2:
            scale, zp = io
            state_q.args = (state_q.args[0], scale, zp, *state_q.args[3:])
            for dn in list(state_q.users):
                if dn.target is dq:
                    dn.args = (dn.args[0], scale, zp, *dn.args[3:])
        for user in node.users:
            if user.target is not operator.getitem or user.args[1] != 1:
                continue
            for qn in list(user.users):
                if qn.target is not q:
                    continue
                qn.args = (qn.args[0], scale, zp, *qn.args[3:])
                for dn in list(qn.users):
                    if dn.target is dq:
                        dn.args = (dn.args[0], scale, zp, *dn.args[3:])
                forced += 1
    gm.recompile()
    return forced, already_io


def force_exact_ingredient_io(gm):
    """QNN_DEFER_EXACT=1 (2026-10-11 KST, after calibration and the IO / shared-state overrides): every deferred-commit
    ingredient input (x, k, v, beta, decay of the previous chunk) takes the encoding of the graph output it is fed
    from, so its codes cross the call boundary unchanged. The commit op then reads exactly the codes the verify op read
    (the group requantization after the head-group slice is already identical on both sides; checked here).
    Found by the diag8 QNN_QDQ_DUMP comparison: v and beta inputs were calibrated on their own (e.g. layer 0 v: output
    0.000342/814, input 0.000155/1792), so the commit replayed rounded copies. Returns a dict of counts."""
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    masked = torch.ops.qwen35.delta_rule_state16_masked.default
    slices = (torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor)
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    outputs = set()
    torch.fx.node.map_arg(out_node.args, lambda n: outputs.add(n))

    def back(node, what):
        if not isinstance(node, torch.fx.Node) or node.target is not what:
            raise ValueError(f"force_exact_ingredient_io: expected {what} at {node}")
        return node.args[0]

    def output_q(src):
        qs = [u for u in src.users if u.target is q and any(d in outputs for d in u.users)]
        if len(qs) != 1:
            raise ValueError(f"force_exact_ingredient_io: {src.name} has {len(qs)} output quantizes")
        return qs[0]

    def take(dst_q, src_q, src_dq):
        """dst placeholder Q := src_q encoding; the dequantizes after dst_q := src_dq encoding."""
        changed = tuple(dst_q.args[1:3]) != tuple(src_q.args[1:3])
        if tuple(dst_q.args[3:]) != tuple(src_q.args[3:]):
            raise ValueError(f"force_exact_ingredient_io: {dst_q.name} range/dtype differs from {src_q.name}")
        dst_q.args = (dst_q.args[0], *src_q.args[1:])
        for d in list(dst_q.users):
            if d.target is dq:
                d.args = (d.args[0], *src_dq.args[1:])
        return changed

    def to_source(dn):
        """From an op input dequantize: walk DQ <- Q <- (slice <- DQ <- Q)* back to the tensor in front of the first
        slice; returns (that tensor, its quantize, the dequantize after it)."""
        d = dn
        while True:
            qn = back(d, dq)
            src = qn.args[0]
            if src.target in slices and isinstance(src.args[0], torch.fx.Node) and src.args[0].target is dq:
                d = src.args[0]
                continue
            return src, qn, d

    counts = {"pairs": 0, "group_changed": 0, "group_already": 0, "kvbd_changed": 0, "kvbd_already": 0,
              "x_changed": 0, "x_already": 0}
    done = set()
    # 2026-10-13 00:0x KST (ckpt10, found by check_exact_ckpt.py on the diag8 test export): with the checkpoint split the
    # head / tail ops read their rows through a second slice (rows [0, c) / [c, T) of the head-group slice) whose
    # requantization was calibrated on those rows alone. The verify side then rounds twice (output -> group -> rows,
    # 19 of 96 diag8 chains with a group encoding equal to neither end) while the commit rounds once (output -> its
    # group encoding), so the replay could differ by a code; and the tail inputs (86 of 96 on their own encoding) would
    # be clipped to the head's rows if they simply took the head's. Every row-slice requantization therefore takes the
    # encoding of the head-group slice in front of it (calibrated on all T rows, = what the unsplit verify op reads):
    # rows -> group is then the identity on codes, and the pair step below puts the commit on that group encoding too.
    # 2026-10-13 05:0x KST (experiments/exact_oneenc_20261013): QNN_DEFER_EXACT_ONE_ENC=1. The ckpt10 device test found
    # the checkpoint continuation inexact (committed states 0.2-0.3% of codes off by 1-3 vs the full replay, every shard)
    # although every QDQ-level chain is replayable: 343 of 384 verify-side chains requantize output -> group encoding,
    # and the compiled graph need not round the verify path the way the replay rounds its stored output codes (e.g. a
    # requantize chain folded into one). With this switch every requantization between an ingredient's producer and a
    # delta op (group slice, row slice; verify, head, tail and commit sides) takes the producer's output encoding, so
    # the only stored codes are the ingredient outputs themselves and every reader takes them unchanged.
    one_enc = os.environ.get("QNN_DEFER_EXACT_ONE_ENC", "0") == "1"
    if one_enc:
        for v in list(gm.graph.nodes):
            if v.target is not masked:
                continue
            st = back(back(v.args[5], dq), q)
            if not (st.target is operator.getitem and st.args[1] == 1 and st.args[0].target is masked):
                continue  # commit op: the pair step below puts it on the verify side's (now output) encoding
            for i in (1, 2, 3, 4):
                d = v.args[i]
                chain = []
                while True:
                    qn = back(d, dq)
                    chain.append((qn, d))
                    src = qn.args[0]
                    if src.target in slices and isinstance(src.args[0], torch.fx.Node) and src.args[0].target is dq:
                        d = src.args[0]
                        continue
                    break
                q_out, dq_out = chain[-1]
                for qn, dn in chain[:-1]:
                    if tuple(qn.args[3:]) != tuple(q_out.args[3:]):
                        raise ValueError(f"force_exact_ingredient_io: one-encoding range/dtype differs at {v.name} input {i}")
                    if tuple(qn.args[1:3]) != tuple(q_out.args[1:3]):
                        counts["one_enc_changed"] = counts.get("one_enc_changed", 0) + 1
                    else:
                        counts["one_enc_already"] = counts.get("one_enc_already", 0) + 1
                    qn.args = (qn.args[0], *q_out.args[1:])
                    for u in list(qn.users):
                        if u.target is dq:
                            u.args = (u.args[0], *dq_out.args[1:])
    # 2026-10-13 10:4x KST: QNN_DEFER_SPLIT_Q_ON_GROUP=1 also puts the query (input 0) row slices of the split head / tail
    # ops on the group slice encoding. The query never enters the state, but its row slices were calibrated per sub-block
    # (rows 0-9 / 10-15), so the split verify computed slightly different outputs than the unsplit export (10-13 S26:
    # ckpt10-off vs plain recompute tokens 7/15); on the group encoding the split reads q exactly as the unsplit op does.
    row_inputs = (0, 1, 2, 3, 4) if os.environ.get("QNN_DEFER_SPLIT_Q_ON_GROUP", "0") == "1" else (1, 2, 3, 4)
    for v in list(gm.graph.nodes):
        if v.target is not masked:
            continue
        for i in row_inputs:
            dv = v.args[i]
            qv = back(dv, dq)
            rs = qv.args[0]
            if not (rs.target in slices and isinstance(rs.args[0], torch.fx.Node) and rs.args[0].target is dq):
                continue  # one slice level (commit op, or an unsplit verify op)
            dg = rs.args[0]
            qg = back(dg, dq)
            if not (isinstance(qg.args[0], torch.fx.Node) and qg.args[0].target in slices):
                continue  # the rows are sliced straight off the ingredient tensor, no group requantization in between
            if tuple(qv.args[3:]) != tuple(qg.args[3:]):
                raise ValueError(f"force_exact_ingredient_io: row/group range differs at {v.name} input {i}")
            if tuple(qv.args[1:]) != tuple(qg.args[1:]) or tuple(dv.args[1:]) != tuple(dg.args[1:]):
                counts["rows_on_group_changed"] = counts.get("rows_on_group_changed", 0) + 1
                qv.args = (qv.args[0], *qg.args[1:])
                for d in list(qv.users):
                    if d.target is dq:
                        d.args = (d.args[0], *dg.args[1:])
            else:
                counts["rows_on_group_already"] = counts.get("rows_on_group_already", 0) + 1
    for v in list(gm.graph.nodes):
        if v.target is not masked:
            continue
        st = back(v.args[5], dq)
        st = back(st, q)
        if not (st.target is operator.getitem and st.args[1] == 1 and st.args[0].target is masked):
            continue  # a commit op (its state comes from the graph input)
        c = st.args[0]
        cs = back(back(c.args[5], dq), q)
        if cs.target is operator.getitem and cs.args[1] == 1 and cs.args[0].target is masked:
            # QNN_DEFER_CHECKPOINT_ROW (2026-10-12 23:4x KST (experiments/ckpt10_w16_20261012)): v is the tail (rows [c, T)) of a split verify op and c its head.
            # The next call commits rows from one ingredient tensor with the head's group encodings (set on the commit
            # by the (commit, head) pair), so the tail must read with the head's group encodings too.
            for i in (1, 2, 3, 4):
                dv, dh = v.args[i], c.args[i]
                qv, qh = back(dv, dq), back(dh, dq)
                if tuple(qv.args[3:]) != tuple(qh.args[3:]):
                    raise ValueError(f"force_exact_ingredient_io: tail/head range differs at {v.name} input {i}")
                if tuple(qv.args[1:]) != tuple(qh.args[1:]) or tuple(dv.args[1:]) != tuple(dh.args[1:]):
                    counts["tail_changed"] = counts.get("tail_changed", 0) + 1
                    qv.args = (qv.args[0], *qh.args[1:])
                    for d in list(qv.users):
                        if d.target is dq:
                            d.args = (d.args[0], *dh.args[1:])
                else:
                    counts["tail_already"] = counts.get("tail_already", 0) + 1
            continue
        counts["pairs"] += 1
        for i in (1, 2, 3, 4):
            dv, dc = v.args[i], c.args[i]
            qv, qc = back(dv, dq), back(dc, dq)
            # the commit op must dequantize with the encoding the verify op used (equal in diag8, not always in full32:
            # full32 w16 decay of one group differed, 15:21 KST)
            if tuple(qv.args[3:]) != tuple(qc.args[3:]):
                raise ValueError(f"force_exact_ingredient_io: group range/dtype differs at {v.name}/{c.name} input {i}")
            if tuple(qv.args[1:]) != tuple(qc.args[1:]) or tuple(dv.args[1:]) != tuple(dc.args[1:]):
                counts["group_changed"] += 1
                qc.args = (qc.args[0], *qv.args[1:])
                for d in list(qc.users):
                    if d.target is dq:
                        d.args = (d.args[0], *dv.args[1:])
            else:
                counts["group_already"] += 1
            # 2026-10-12 23:5x KST (ckpt10): with the checkpoint split the verify input is sliced twice (head group,
            # then rows [0, c)); walk back through every slice to the tensor the ingredient output is taken from.
            producer, qav, dav = to_source(dv)
            ph, qac, dac = to_source(dc)
            if back(qv, q).target not in slices or back(qc, q).target not in slices:
                raise ValueError(f"force_exact_ingredient_io: {v.name}/{c.name} input {i} is not a head-group slice")
            if ph.op != "placeholder" or ph in done:
                if ph.op != "placeholder":
                    raise ValueError(f"force_exact_ingredient_io: commit input {i} of {c.name} is not a graph input")
                continue
            done.add(ph)
            oq = output_q(producer)
            if tuple(oq.args[1:]) != tuple(qav.args[1:]):
                raise ValueError(f"force_exact_ingredient_io: output {oq.name} is not on the verify encoding")
            counts["kvbd_changed" if take(qac, qav, dav) else "kvbd_already"] += 1
    # x: commit cat (feeds the conv-state gather) input 1 <- x_prev; verify cat input 1 <- x (the same tensor as the output)
    for g in list(gm.graph.nodes):
        if g.target is not torch.ops.aten.index_select.default:
            continue
        cat_c = back(back(g.args[0], dq), q)
        if cat_c.target is not torch.ops.aten.cat.default:
            raise ValueError(f"force_exact_ingredient_io: gather {g.name} does not read a cat")
        dxin = cat_c.args[0][1]
        qxin = back(dxin, dq)
        ph = qxin.args[0]
        cat_v, frontier = None, list(g.users)
        for _ in range(8):   # g -> q -> dq (-> q -> dq of the explicit gather requant) -> verify cat
            nxt = []
            for u in frontier:
                if u.target is torch.ops.aten.cat.default and u is not cat_c:
                    cat_v = cat_v or u
                elif u.target in (q, dq):
                    nxt.extend(u.users)
            frontier = nxt
        if cat_v is None or ph.op != "placeholder":
            raise ValueError(f"force_exact_ingredient_io: verify cat / x_prev not found for {g.name}")
        dxt = cat_v.args[0][1]
        qxt = back(dxt, dq)
        oq = output_q(qxt.args[0])
        if tuple(oq.args[1:]) != tuple(qxt.args[1:]):
            raise ValueError(f"force_exact_ingredient_io: x output {oq.name} is not on the verify cat input encoding")
        counts["x_changed" if take(qxin, qxt, dxt) else "x_already"] += 1
        # Conv window as in the snapshot export (2026-10-11 19:xx KST, full32 check): there the verify cat is the graph
        # output of the conv history, so it is stored on the conv-state IO encoding E_io and the conv reads it back on the
        # cat's calibrated encoding (a requantize; up to a 1-code difference in 7-8 layers where the two differ). Make
        # both cats store E_io, keep the conv's read encoding, and carry the committed window on E_io, so the conv input
        # and the carried conv state are the snapshot's codes.
        if os.environ.get("QNN_DEFER_EXACT_CONV", "1") == "1":
            qbase = back(cat_c.args[0][0], dq)
            e_io = tuple(qbase.args[1:3])
            qcat = back(g.args[0], dq)
            def put(n, enc):
                if tuple(n.args[1:3]) != enc:
                    n.args = (n.args[0], *enc, *n.args[3:])
                    return 1
                return 0
            changed = put(qcat, e_io)
            for d in list(qcat.users):
                if d.target is dq:
                    changed += put(d, e_io)
            frontier = list(g.users)                 # g -> q/dq ... -> (verify cat, output)
            while frontier:
                nxt = []
                for u in frontier:
                    if u.target in (q, dq):
                        changed += put(u, e_io)
                        nxt.extend(u.users)
                frontier = nxt
            for u in list(cat_v.users):              # store only; the dequantize keeps the conv's read encoding
                if u.target is q:
                    changed += put(u, e_io)
            # QNN_DEFER_EXACT_CONV_ONEGRID=1 (2026-10-13 KST): the conv
            # also reads the window on E_io, so the window is stored and read on one grid and QNN inserts no requantize.
            # Store-only left E_io 0.0005449663/38190 vs read 0.0005449633/38190 in layer 4 of the 10-11 export (5 ppm apart)
            # and cost one 19 x 8192 conversion op (aten__to_copy, 0.42 M cycles) per DeltaNet layer on the S25.
            if os.environ.get("QNN_DEFER_EXACT_CONV_ONEGRID", "0") == "1":
                views = {t for t in (getattr(getattr(torch.ops.aten, name, None), ov, None) for name, ov in (
                    ("unsqueeze", "default"), ("unsqueeze_copy", "default"), ("view", "default"), ("view_copy", "default"),
                    ("reshape", "default"), ("squeeze", "dims"), ("squeeze", "dim"), ("squeeze_copy", "dims"),
                    ("permute", "default"), ("permute_copy", "default"))) if t is not None}
                path, frontier, ends = [], [u for u in cat_v.users if u.target is q], []
                for _ in range(16):                  # Q -> DQ -> view ops -> Q -> DQ -> conv
                    nxt = []
                    for u in frontier:
                        if u.target in (q, dq):
                            path.append(u); nxt.extend(u.users)
                        elif u.target in views:
                            nxt.extend(u.users)
                        else:
                            ends.append(u)
                    frontier = nxt
                    if not frontier:
                        break
                if frontier or not ends or not path:
                    raise ValueError(f"force_exact_ingredient_io: window read path of {cat_v.name} not resolved "
                                     f"(ends {[e.name for e in ends]}, left {[f.name for f in frontier]})")
                og = sum(put(n, e_io) for n in path)
                bad = [n.name for n in path if tuple(n.args[1:3]) != e_io]
                if bad:
                    raise ValueError(f"force_exact_ingredient_io: window read path not on E_io: {bad}")
                counts["conv_onegrid_layers"] = counts.get("conv_onegrid_layers", 0) + 1
                counts["conv_onegrid_changed"] = counts.get("conv_onegrid_changed", 0) + og
                counts["conv_onegrid_ends"] = sorted({*counts.get("conv_onegrid_ends", []), *(str(e.target) for e in ends)})
            counts["conv_layers"] = counts.get("conv_layers", 0) + 1
            counts["conv_changed"] = counts.get("conv_changed", 0) + changed
    gm.recompile()
    return counts
