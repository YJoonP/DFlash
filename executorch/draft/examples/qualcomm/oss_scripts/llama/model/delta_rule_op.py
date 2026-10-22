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

import os

import torch
from torch.library import Library, impl

from executorch.examples.qualcomm.oss_scripts.llama.model.state_precision import (
    recurrent_state_is_quantized,
)

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


delta_rule_lib.define("delta_commit(Tensor key, Tensor value, Tensor beta, Tensor decay, Tensor state_in) -> Tensor")


@impl(delta_rule_lib, "delta_commit", dispatch_key="CompositeExplicitAutograd")
def delta_commit_impl(key, value, beta, decay, state_in):
    state = state_in
    for t in range(key.shape[1]):
        k = key[:, t].unsqueeze(-1)
        state = state * decay[:, t].unsqueeze(-1)
        delta = (value[:, t] - (state * k).sum(dim=-2)) * beta[:, t]
        state = state + k * delta.unsqueeze(-2)
    return state


delta_rule_lib.define("delta_commit.out(Tensor key, Tensor value, Tensor beta, Tensor decay, Tensor state_in, *, Tensor(a!) out) -> Tensor(a!)")


@impl(delta_rule_lib, "delta_commit.out", dispatch_key="CompositeExplicitAutograd")
def delta_commit_out_impl(key, value, beta, decay, state_in, *, out):
    out.copy_(delta_commit_impl(key, value, beta, decay, state_in))
    return out


# HTP native float Slice/Concat can round FP32 tensors through FP16. These
# transport ops copy raw FP32 words on HVX and have no arithmetic or quantization.
def _register_state_transport(n):
    split_name, join_name = f"state_split{n}", f"state_join{n}"
    returns = ", ".join(["Tensor"] * n)
    delta_rule_lib.define(f"{split_name}(Tensor state) -> ({returns})")
    delta_rule_lib.define(f"{join_name}(" + ", ".join(f"Tensor state{i}" for i in range(n)) + ") -> Tensor")

    def split(state):
        if state.ndim != 4 or state.shape[1] % n:
            raise ValueError("FP32 state heads must divide evenly into groups")
        return tuple(x.clone() for x in state.chunk(n, dim=1))

    def join(*states):
        return torch.cat(states, dim=1)

    impl(delta_rule_lib, split_name, "CompositeExplicitAutograd")(split)
    impl(delta_rule_lib, join_name, "CompositeExplicitAutograd")(join)
    aliases = [f"Tensor({chr(97+i)}!)" for i in range(n)]
    delta_rule_lib.define(f"{split_name}.out(Tensor state, *, " +
                          ", ".join(f"{a} out{i}" for i, a in enumerate(aliases)) +
                          ") -> (" + ", ".join(aliases) + ")")
    delta_rule_lib.define(f"{join_name}.out(" + ", ".join(f"Tensor state{i}" for i in range(n)) +
                          ", *, Tensor(a!) out) -> Tensor(a!)")

    def split_out(state, **outputs):
        for i, x in enumerate(split(state)):
            outputs[f"out{i}"].copy_(x)
        return tuple(outputs[f"out{i}"] for i in range(n))

    def join_out(*states, out):
        out.copy_(join(*states))
        return out

    impl(delta_rule_lib, split_name + ".out", "CompositeExplicitAutograd")(split_out)
    impl(delta_rule_lib, join_name + ".out", "CompositeExplicitAutograd")(join_out)


for _state_groups in (2, 4, 8):
    _register_state_transport(_state_groups)


def split_recurrent_state(state, groups):
    if recurrent_state_is_quantized():
        size = state.shape[1] // groups
        return tuple(state[:, i * size:(i + 1) * size] for i in range(groups))
    if groups not in (2, 4, 8):
        raise ValueError("Exact FP32 transport supports 2, 4 or 8 head groups")
    return getattr(torch.ops.qwen35, f"state_split{groups}").default(state)


def join_recurrent_state(states):
    if recurrent_state_is_quantized():
        return torch.cat(states, dim=1)
    return getattr(torch.ops.qwen35, f"state_join{len(states)}").default(*states)


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
    if recurrent_state_is_quantized() and package_capabilities().get("state16_workspace_op"):
        return torch.ops.qwen35.delta_rule_state16.default
    return torch.ops.qwen35.delta_rule.default


def get_delta_decode_op():
    if not recurrent_state_is_quantized() or not package_capabilities().get("decode16_unique_op"):
        raise ValueError("decode specialization requires uint16 state and a decode16_unique_op package")
    return torch.ops.qwen35.delta_rule_decode16.default


def validate_package_configuration(head_groups=1):
    capabilities = package_capabilities()
    if head_groups < 1:
        raise ValueError("QNN_DELTA_HEAD_GROUPS must be positive")
    if head_groups != 1 and not capabilities.get("independent_head_workspace", False):
        raise ValueError(
            "Head groups require a package with independently owned workspace. "
            "The legacy package shares global scratch; build qhpi/build.sh into "
            "a separate directory and set QNN_DELTA_OPPKG_DIR to that directory."
        )
    if (not recurrent_state_is_quantized() and head_groups > 1
            and not capabilities.get("fp32_state_transport", False)):
        raise ValueError("FP32 grouped state requires the exact state transport package; native HTP Slice/Concat can round to FP16")
    if (capabilities.get("implementation") == "qhpi"
            and recurrent_state_is_quantized()
            and "uint16" not in capabilities.get("state_dtypes", [])):
        raise ValueError("This QHPI build requires float32 recurrent state")


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
    if package_capabilities().get("fp32_state_transport", False):
        op_map["DeltaCommit"] = torch.ops.qwen35.delta_commit.default
        for groups in (2, 4, 8):
            op_map[f"StateSplit{groups}"] = getattr(torch.ops.qwen35, f"state_split{groups}").default
            op_map[f"StateJoin{groups}"] = getattr(torch.ops.qwen35, f"state_join{groups}").default
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
    # Recurrent precision may override QNN_STATE_IO_16BIT without changing conv IO.
    # state 입출력 dtype 은 export 설정을 따른다.
    #   QNN_STATE_IO_16BIT=1 -> state_out 도 16bit 로 관측해 자기 인코딩을 갖는다.
    #     (러너가 uint16 상태 버퍼를 그 zero_point 로 채워야 하므로 실제 관측값이 필요하다)
    #   =0 -> state_out 은 float 유지. output_qspec=None + _annotated=True = 관측기 없음.
    state_16 = recurrent_state_is_quantized()
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
        torch.ops.qwen35.delta_rule_decode16.default,
        IOQuantConfig(input_quant_specs={i: cfg.input_activation for i in range(6)},
                      output_quant_specs={0: cfg.output_activation, 1: cfg.output_activation}),
    )
    ann.register_annotation(
        torch.ops.qwen35.delta_commit.default,
        IOQuantConfig(input_quant_specs={**{i: cfg.input_activation for i in range(4)}, 4: None},
                      output_quant_specs={0: None}),
    )
    for groups in (2, 4, 8):
        ann.register_annotation(
            getattr(torch.ops.qwen35, f"state_split{groups}").default,
            IOQuantConfig(input_quant_specs={0: None}, output_quant_specs={i: None for i in range(groups)}),
        )
        ann.register_annotation(
            getattr(torch.ops.qwen35, f"state_join{groups}").default,
            IOQuantConfig(input_quant_specs={i: None for i in range(groups)}, output_quant_specs={0: None}),
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
