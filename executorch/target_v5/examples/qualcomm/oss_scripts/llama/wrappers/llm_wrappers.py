# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import copy
import gc
import inspect
import json
import operator
import os
import logging
import re

from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset import (
    DataConfig,
    DatasetBuilder,
    ModalityEncoderDataset,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    GraniteSpeechEncoder,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.evaluator.lm_eval_adapter import (
    run_lm_eval,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import (
    DecoderInference,
    EncoderInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.mix_precision_analyzer import (
    PerLayerSqnrAnalyzer,
    save_suggest_recipes,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import (
    TokenEmbedding,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.quantize import (
    PTQStrategy,
    QATStrategy,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.train.config import TrainingArgs
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torch.utils.data import DataLoader
from torchao.quantization.pt2e import MinMaxObserver, move_exported_model_to_train
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from transformers import AutoModel, AutoModelForSpeechSeq2Seq


def is_node_src_start_with_name(node: torch.fx.Node, kv_cache_prefix: str) -> bool:
    """
    Return True if any NodeSource in node.meta['from_node'] has a name
    starting with `kv_cache_prefix`. Used to identify K/V cache nodes by their
    "k_" or "v_" name prefix in the traced graph.
    """

    def has_source_name_prefix(
        node_src: torch.fx.traceback.NodeSource, kv_cache_prefix: str
    ) -> bool:

        name = getattr(node_src, "name", None)
        if isinstance(name, str) and name.startswith(kv_cache_prefix):
            return True

        children = getattr(node_src, "from_node", None)
        if not children:
            return False

        for src in children:
            if has_source_name_prefix(src, kv_cache_prefix):
                return True

        return False

    node_srcs = node.meta.get("from_node", None)
    if not node_srcs:
        return False

    return any(
        has_source_name_prefix(node_src, kv_cache_prefix) for node_src in node_srcs
    )


# --- mutable buffer 상태(QNN_DELTA_MUTBUF=1) 인코딩 통일 헬퍼 (2026-09-18 KST) ---
_MUTBUF_RE = re.compile(r"layers[._](\d+)[._]attention[._](conv|recurrent)_state_buf")
_MUTBUF_Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
_MUTBUF_DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default


def mutbuf_pairs(gm):
    """{(층, conv|recurrent): (버퍼 노드, 갱신값의 q 입력 노드)} — copy_(buffer, value) 에서 찾는다.
    .module() 그래프에서는 버퍼가 get_attr, edge 이전이라 copy_ 가 살아 있다."""
    pairs = {}
    for n in gm.graph.nodes:
        if n.target is not torch.ops.aten.copy_.default or len(n.args) < 2:
            continue
        ph = n.args[0]
        if not isinstance(ph, torch.fx.Node) or ph.op not in ("placeholder", "get_attr"):
            continue
        m = _MUTBUF_RE.search(str(ph.target)) or _MUTBUF_RE.search(ph.name)
        if not m:
            continue
        src, cur = None, n.args[1]
        for _ in range(3):  # dq(q(src)) 를 거슬러 q 의 입력
            if not isinstance(cur, torch.fx.Node):
                break
            if cur.target is _MUTBUF_Q:
                src = cur.args[0]
                break
            cur = cur.args[0] if cur.args else None
        pairs[(int(m.group(1)), m.group(2))] = (ph, src)
    return pairs


def mutbuf_q_args(node):
    if node is None:
        return None
    for u in node.users:
        if u.target is _MUTBUF_Q:
            return tuple(float(a) if isinstance(a, float) else a for a in u.args[1:3])
    return None


def mutbuf_force_q(node, scale, zp):
    """node 의 모든 q 사용자(와 그 dq 사용자)의 (scale, zp) 를 강제한다. 반환 = 고친 q 수."""
    n = 0
    if node is None:
        return 0
    for u in list(node.users):
        if u.target is _MUTBUF_Q:
            u.args = (u.args[0], scale, zp, *u.args[3:])
            n += 1
            for v in list(u.users):
                if v.target is _MUTBUF_DQ:
                    v.args = (v.args[0], scale, zp, *v.args[3:])
    return n


def gather_explicit_requant(gm):
    """QNN_PREFILL_VALID: the conv-state index_select reads the prefill cat encoding A, but the IO
    overrides force its output q/dq to decode's conv-state encoding B. Keep the Gather on A and
    requantize A -> B with an explicit q/dq pair after it, so no Gather ever changes encoding.
    Returns the number of index_select outputs rewritten."""
    fixed = 0
    for node in list(gm.graph.nodes):
        if node.target is not torch.ops.aten.index_select.default:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node) or src.target is not _MUTBUF_DQ:
            logging.warning(f"gather_explicit_requant: {node.name} input is not a per-tensor dequantize; left as is")
            continue
        a_scale, a_zp = src.args[1], src.args[2]
        qs = [u for u in node.users if u.target is _MUTBUF_Q]
        if not qs:
            logging.warning(f"gather_explicit_requant: {node.name} has no quantize user; left as is")
        for q in qs:
            if (q.args[1], q.args[2]) == (a_scale, a_zp):
                logging.info(f"gather_explicit_requant: {node.name} already on its input encoding ({a_scale}, {a_zp})")
                continue
            with gm.graph.inserting_before(q):
                qa = gm.graph.call_function(_MUTBUF_Q, (node, a_scale, a_zp, *src.args[3:]))
                dqa = gm.graph.call_function(_MUTBUF_DQ, (qa, a_scale, a_zp, *src.args[3:]))
            q.replace_input_with(node, dqa)
            fixed += 1
    gm.graph.lint()
    gm.recompile()
    return fixed


def node_layer_idx(node: torch.fx.Node):
    """노드가 속한 디코더 층 번호. 찾지 못하면 None.

    nn_module_stack 의 경로 문자열에서 'layers.<N>' 을 읽는다.
    KV 캐시 인코딩을 '위치' 가 아니라 '층' 으로 짝짓기 위해 필요하다.
    """
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    for path, _ in stack.values():
        m = re.search(r"layers\.(\d+)", str(path))
        if m:
            return int(m.group(1))
    return None


@contextlib.contextmanager
def _no_load_state_dict_hooks(module: torch.nn.Module):
    """load_state_dict 프리훅을 잠시 끈다.

    static_llama.py 의 _fold_unit_offset 은 Qwen3.5 의 RMSNorm 가중치(1+w 규약)에
    +1 을 더하는데 **멱등하지 않다**. fp32 가중치 공유 레버는 이미 훅을 거친
    state_dict 를 두 번째로 load 하므로, 훅이 다시 돌면 PREFILL/CALIBRATE 인스턴스가
    2+w 를 갖게 된다. 하필 양자화되는 인스턴스가 CALIBRATE 라서 모든 활성화 스케일이
    틀린 모델에서 관측되고, parameter_override 가 그 가중치를 배포 그래프로 옮긴다.
    2026-09-08 기기 실측: 생성이 통째로 깨짐(채널당 최대 86배 게인 오차).

    gated_deltanet 의 _reshape_conv1d_weight_hook 은 dim()==3 가드가 있어 멱등하므로
    끄든 켜든 무해하다. 여기서는 안전을 위해 모든 프리훅을 끈다.
    """
    saved = []
    for m in module.modules():
        if getattr(m, "_load_state_dict_pre_hooks", None):
            saved.append((m, m._load_state_dict_pre_hooks))
            m._load_state_dict_pre_hooks = OrderedDict()
    try:
        yield
    finally:
        for m, hooks in saved:
            m._load_state_dict_pre_hooks = hooks


def _malloc_trim():
    """glibc arena 를 OS 로 반환한다. gc.collect() 만으로는 RSS 가 내려가지 않는다."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class TextDecoder(Component):

    # fp32 가중치 1벌 공유용 캐시. DECODE/PREFILL/CALIBRATE 인스턴스가 각자
    # .to(fp32) 로 자기 몫을 만들면 4B 기준 16.4GiB x 3 = 49.3GiB 를 낭비한다.
    # decode/prefill 의 가중치 '값' 은 _encoding_override 가 get_attr 타깃을
    # calibration_prefill 텐서로 되돌리므로 애초에 산출물에 도달하지 않는다.
    _shared_fp32_weights = None

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.pass_manager_cls = get_qnn_pass_manager_cls()
        self.passes_job = self.pass_manager_cls.get_capture_program_passes()
        self.dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
        )
        self.meta = {}
        recipe_cls = (
            self.config.qat_recipe
            if control_args.qat and self.config.qat_recipe
            else self.config.quant_recipe
        )
        self.quant_recipe: StaticLLMQuantRecipe = (
            recipe_cls(mode == Mode.CALIBRATE) if recipe_cls else None
        )

        # For multimodal embedding
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            self.pass_manager_cls.get_capture_program_passes()
            if apply_embedding
            else None
        )
        self.tok_embedding_dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
            if apply_embedding
            else None
        )

        # load static llama model args
        if control_args.decoder_model == "gemma4-e2b":
            from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
                Gemma4Config,
            )

            params_path = config.params_path
            self.gemma4_config = Gemma4Config.from_json(params_path)
            self.gemma4_config.use_kv_cache = True
            self.gemma4_config.max_batch_size = 1
            self.gemma4_config.max_seq_len = control_args.max_seq_len
            self.gemma4_config.max_context_len = control_args.max_context_len
            self.model_args = None
        else:
            params_path = (
                config.params_path
                if control_args.params is None
                else control_args.params
            )
            with open(params_path) as f:
                self.model_args = process_model_args(
                    control_args,
                    ModelArgs(**json.load(f)),
                    self.quant_recipe,
                    config,
                    mode,
                )
        # prepare instance
        self.tok_embedding, self.decoder = self._prepare_model()

        # DeltaNet 상태 shape 은 '여기서' 계산해 캐시해야 한다.
        # self.decoder 는 torch.export.export() 시점에 GraphModule 로 교체되고,
        # 그 뒤에는 .layers 가 ModuleList 가 아니라 평범한 Module 이라 순회할 수 없다
        # ('Module' object is not iterable). _tag_ios / _save_output_state_quant_attrs
        # 는 둘 다 export 이후에 돌기 때문에 그때 계산하면 늦다.
        self._cache_linear_attention_state_shapes()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            layer_prefix_offsets = None
            if self.control_args.decoder_model == "gemma4-e2b":
                n_self = self.gemma4_config.num_self_decoder_layers
                layer_prefix_offsets = {
                    "model.self_decoder.layers": 0,
                    "model.cross_decoder.layers": n_self,
                }
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
                pattern=self._get_sharding_get_pattern(),
                layer_prefix_offsets=layer_prefix_offsets,
                shard_layers=getattr(self.config, "shard_layers", None),
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

        self._decoder_inference = (
            DecoderInference(
                get_example_inputs=self.get_example_inputs,
                audio_token_id=self.meta.get("audio_token_id", None),
                image_token_id=self.meta.get("image_token_id", None),
                max_context_len=self.meta["get_max_context_len"],
                max_batch_size=self.meta["get_max_batch_size"],
                use_i64_token=self.control_args.embedding_quantize is not None,
                layer_types=(
                    self.model_args.layer_types
                    if self.model_args is not None
                    else None
                ),
                stepwise_calibration=getattr(self.config, "calibrate_ar1", False),
                defer_n_delta=(
                    len(self.decoder.delta_layer_ids)
                    if getattr(self.decoder, "defer_state", False)
                    else 0
                ),
            )
            if self.decoder is not None
            else None
        )

    def _get_sharding_get_pattern(self):
        if self.control_args.decoder_model == "gemma4-e2b":
            prefixes = [
                "model.cross_decoder.layers",
                "model.self_decoder.layers",
            ]
        else:
            prefixes = [
                "layers",
            ]
        prefix_alt = "|".join(re.escape(p) for p in prefixes)
        return rf"^(?:{prefix_alt})\.(\d+)"

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None, None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            if self.control_args.decoder_model == "gemma4-e2b":
                from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.convert_weights import (
                    remap_keys,
                )
                from huggingface_hub import snapshot_download

                state_dict = self.config.convert_weights.__func__(
                    snapshot_download(repo_id=self.config.repo_id),
                    self.gemma4_config,
                    torch.float32,
                )
                state_dict = remap_keys(state_dict)
            else:
                checkpoint = download_and_convert_hf_checkpoint(
                    self.config.repo_id,
                    self.config.convert_weights.__func__,
                )
                state_dict = torch.load(
                    checkpoint, weights_only=True, map_location="cpu", mmap=True
                )
                if self.control_args.decoder_model in {
                    "gemma-2b",
                    "gemma2-2b",
                    "gemma3-1b",
                }:
                    for k, v in state_dict.items():
                        if "norm" not in k:
                            continue
                        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                        # See https://github.com/huggingface/transformers/pull/29402
                        state_dict[k] = v.float() + torch.ones(
                            v.shape, dtype=torch.float32
                        )
                for k, v in state_dict.items():
                    if "tok_embeddings.weight" == k:
                        state_dict[k] = (
                            v.float() * self.model_args.embedding_scale_factor
                        )
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        decoder.load_state_dict(state_dict, strict=True, assign=True)

        if any([self.config.r1, self.config.r2]):
            raise RuntimeError(
                "SpinQuant (r1/r2) is no longer supported: the "
                "torchao.prototype.spinquant module has been deleted."
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedforward_conv", None):
                layer.feed_forward.prepare_feedforward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

            # --- fp32 가중치 1벌 공유 (피크 RAM) ---------------------------
            # 아래 셋은 가중치를 인스턴스별로 *제자리에서* 고치므로 공유하면 안 된다:
            #   r1/r2            : SpinQuant 회전이 in-place -> 공유 시 3번 적용됨
            #   seq_mse_candidates: 인스턴스별 스케일을 탐색해 되쓴다
            #   qat              : 학습한다
            can_share = (
                not self.config.r1
                and not self.config.r2
                and not getattr(self.config, "seq_mse_candidates", 0)
                and not getattr(self.control_args, "qat", False)
                # QNN_SHARE_FP32=0 이면 공유를 끈다. 메모리가 충분할 때의 대조군이자,
                # 공유 경로에 의심이 생겼을 때의 안전밸브 (피크 RAM +~33 GiB).
                and os.environ.get("QNN_SHARE_FP32", "1") != "0"
            )
            if os.environ.get("QNN_SHARE_FP32", "1") == "0":
                logging.info(f"[{self.mode}] QNN_SHARE_FP32=0: fp32 weight sharing disabled")
            if can_share:
                shared = TextDecoder._shared_fp32_weights
                cur = decoder.state_dict()
                if shared is None:
                    TextDecoder._shared_fp32_weights = dict(cur)
                    logging.info(
                        f"[{self.mode}] cached fp32 weights for later instances"
                    )
                elif set(shared) == set(cur):
                    with _no_load_state_dict_hooks(decoder):
                        decoder.load_state_dict(shared, strict=True, assign=True)
                    # 사후 검증. assign=True + 훅 없음이면 모든 텐서가 캐시와
                    # **같은 storage** 를 가리켜야 한다(= 메모리 공유 성공).
                    # 훅이 state_dict 를 건드리면 새 할당이 생겨 data_ptr 이 달라지므로
                    # 여기서 걸린다. 객체 동일성(is)으로 보면 안 된다 -- state_dict() 는
                    # detach() 된 새 객체를 돌려주기 때문이다.
                    # 이 검사가 없어서 2026-09-07~08 의 export 세 번이 조용히
                    # 틀린 모델(2+w RMSNorm)을 만들었다.
                    now = decoder.state_dict()
                    tainted = [
                        k
                        for k in shared
                        if now[k].shape != shared[k].shape
                        or now[k].data_ptr() != shared[k].data_ptr()
                    ]
                    if tainted:
                        raise RuntimeError(
                            f"[{self.mode}] fp32 weight sharing was corrupted by a "
                            f"load_state_dict hook on {len(tainted)} tensors "
                            f"(e.g. {tainted[:3]}). Refusing to calibrate a model "
                            f"whose weights differ from the cached ones."
                        )
                    del cur, now
                    gc.collect()
                    _malloc_trim()
                    logging.info(f"[{self.mode}] reusing the cached fp32 weights")
                else:
                    logging.warning(
                        f"[{self.mode}] fp32 weight key mismatch; not sharing"
                    )
            # ---------------------------------------------------------------

        # check embedding fallback
        if self.control_args.embedding_quantize:
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None
        use_i64_token = self.control_args.embedding_quantize is not None

        # get embedding model
        tok_embedding = None
        if self.apply_embedding:
            if hasattr(self.config, AUDIO_ENCODER):
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
            elif hasattr(self.config, VISION_ENCODER):
                auto_model = AutoModel.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
            tok_embedding = TokenEmbedding(
                auto_model.get_input_embeddings().to(torch.float32),
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        if self.control_args.decoder_model == "gemma4-e2b":
            from executorch.examples.qualcomm.oss_scripts.gemma4.model_wrapper import (
                Gemma4TextModelWrapper,
            )

            ar_len = (
                self.control_args.prefill_ar_len
                if self.mode == Mode.PREFILL
                else (
                    self.control_args.max_context_len
                    if self.mode == Mode.CALIBRATE
                    else 1
                )
            )
            extra_kwargs = {
                # 32 is the sentinel for "unquantized KV IO"; get_kv_io_bit_width()
                # returns it too when the recipe has no default_quant_dtype.
                "kv_io_bit_width": (
                    self.quant_recipe.get_kv_io_bit_width() if self.quant_recipe else 32
                ),
            }
            decoder: Gemma4TextModelWrapper = Gemma4TextModelWrapper(
                self.gemma4_config,
                ar_len=ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                enable_masked_softmax=False,
                **extra_kwargs,
            )
        else:
            decoder: LlamaModel = LLM_VARIANT_ARCHS.get(
                self.control_args.decoder_model, LlamaModel
            )(
                self.model_args,
                ar_len=self.model_args.ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                **get_model_specific_kwargs(self.control_args, self.config),
                **(
                    {
                        "prefill_valid": True,
                        "prefill_last_row": os.environ.get("QNN_PREFILL_LAST_ROW", "0") == "1",
                    }
                    if self.mode == Mode.PREFILL and os.environ.get("QNN_PREFILL_VALID", "0") == "1"
                    else {}
                ),
                # QNN_DEFER_STATE: the prefill graph is the DFlash verify graph (deferred commit)
                **(
                    {"defer_state": True}
                    if self.mode == Mode.PREFILL and os.environ.get("QNN_DEFER_STATE", "0") == "1"
                    else {}
                ),
            )

        self.prefill_valid = getattr(decoder, "prefill_valid", False)
        self.prefill_last_row = getattr(decoder, "prefill_last_row", False)
        self.defer_state = getattr(decoder, "defer_state", False)
        # DeltaNet layer count of the deferred graph (the graph module loses LlamaModel attributes later)
        self.defer_n_delta = len(decoder.delta_layer_ids) if self.defer_state else 0
        self.meta = decoder.get_metadata()
        # get example input
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
            *(self.example_input[5] if len(self.example_input) > 5 else []),  # valid, conv_idx
        )
        self.io_shape = {
            # logit output
            (
                decoder.max_batch_size,
                1 if self.prefill_last_row else decoder.ar_len,
                decoder.vocab_size,
            ),
        }
        # shape of k caches and v caches
        if self.control_args.decoder_model == "gemma4-e2b":
            # Gemma 4 has per-layer head_dim: sliding=256, full=512
            kv_head_dims = {
                self.meta["get_head_dim"],
                self.meta["get_global_head_dim"],
            }
            self.kv_cache_shape = set()
            for head_dim in kv_head_dims:
                self.kv_cache_shape.add((head_dim, self.meta["get_max_context_len"]))
                self.kv_cache_shape.add((self.meta["get_max_context_len"], head_dim))
                self.kv_cache_shape.add((head_dim, self.meta["get_ar_len"]))
                self.kv_cache_shape.add((self.meta["get_ar_len"], head_dim))
        else:
            self.kv_cache_shape = {
                # single head, kv input
                (self.meta["get_head_dim"], self.meta["get_max_context_len"]),
                (self.meta["get_max_context_len"], self.meta["get_head_dim"]),
                # single head, kv output
                (self.meta["get_head_dim"], self.meta["get_ar_len"]),
                (self.meta["get_ar_len"], self.meta["get_head_dim"]),
            }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens

        return tok_embedding, decoder

    @property
    def attn_mask(self):
        """Attention mask for this decoder graph, used as a schema for dataset construction."""
        return self.example_input[1]

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _save_output_kv_cache_quant_attrs(self):
        kv_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            # 양자화된 KV 출력만 해당한다 (dequantize(quantize(src))). float 로 남긴 출력
            # (예: DeltaNet 커널의 state_out getitem)은 args 구조가 달라 건너뛴다.
            if (
                node.target
                != torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue
            cache_output_node = node.args[0].args[0]
            val = cache_output_node.meta.get("val")
            if not torch.is_tensor(val):
                continue
            if val.size()[-2:] in self.kv_cache_shape:
                # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
                # This meta is for attention sink feature
                self.meta[f"get_kv_output_{kv_idx}_quant_attr"] = [
                    node.args[1],
                    node.args[2],
                    node.args[3],
                    node.args[4],
                    str(node.args[5]),
                ]
                kv_idx += 1

    def _save_output_state_quant_attrs(self):
        """DeltaNet 상태 슬롯 출력의 양자화 인코딩을 메타에 남긴다.

        상태 IO 를 16bit 로 태깅하면(_tag_ios) 러너가 잡는 상태 버퍼가 uint16 이 된다.
        그런데 상태의 인코딩은 비대칭이라 zero_point 가 0 이 아니다(실측 8840~45008).
        버퍼를 0 바이트로 두면 초기 상태가 0 이 아니라 -zero_point*scale 이 되어
        첫 토큰부터 틀어진다. 러너가 zero_point 로 채울 수 있도록 여기서 노출한다.

        인덱스는 그래프 출력 순서(logits, k_0..k_{L-1}, v_0..v_{L-1})에서 logits 를 뺀
        위치다. 즉 layer L 의 k 슬롯은 L, v 슬롯은 num_layers+L 이다.
        """
        # 이 메타는 러너의 상태 버퍼 초기화용 '부가' 정보다.
        # 여기서 예외가 나도 export 전체(수십 분)를 죽여서는 안 된다.
        try:
            state_shape = self._linear_attention_state_shapes()
            if not state_shape:
                logging.info("no linear-attention state shapes; skip state quant attrs")
                return
            saved = 0
            idx = 0
            DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            QT = torch.ops.quantized_decomposed.quantize_per_tensor.default

            def _tensor_val(n, depth=5):
                """출력이 내놓는 텐서를 찾는다.

                그래프 출력은 두 종류가 섞여 있다 (2026-09-10 실측):
                  - dequantize 노드: 자기 meta["val"] 이 비어 있어 위로 거슬러 올라가야 한다
                  - getitem 노드   : 자기 meta["val"] 에 텐서가 있다 (다중 출력 커스텀 op)
                한 종류만 다루면 매번 절반만 잡힌다. 체인을 훑어 첫 텐서를 쓴다.
                """
                cur = n
                for _ in range(depth):
                    if not isinstance(cur, torch.fx.Node):
                        return None
                    v = cur.meta.get("val")
                    if torch.is_tensor(v):
                        return v
                    if not cur.args or not isinstance(cur.args[0], torch.fx.Node):
                        return None
                    cur = cur.args[0]
                return None

            def _quant_node(n, depth=4):
                """출력에서 거슬러 올라가 q/dq 노드를 찾는다.
                그래프 구조는 빌드마다 다르다 — dq(q(src)) 일 수도, q(getitem) 일 수도,
                getitem 하나만 있을 수도 있다. 패턴을 못박지 않고 체인을 훑는다."""
                cur = n
                for _ in range(depth):
                    if not isinstance(cur, torch.fx.Node):
                        return None
                    if cur.target in (DQ, QT):
                        return cur
                    if not cur.args or not isinstance(cur.args[0], torch.fx.Node):
                        return None
                    nxt = cur.args[0]
                    if isinstance(nxt.meta.get("val"), (tuple, list)):
                        # 다중 출력 노드 경계. 더 올라가면 그 op 의 '입력'을 이 출력의
                        # 양자화 노드로 착각한다 (2026-09-10 실측: DeltaRule 의
                        # state_out 이 query 입력의 scale/zero_point 를 물려받음).
                        return None
                    cur = nxt
                return None

            for node in self.decoder.graph.nodes:
                if node.op != "output":
                    continue
                for out in node.args[0]:
                    if not isinstance(out, torch.fx.Node):
                        if out is not node.args[0][0]:
                            idx += 1
                        continue
                    val = _tensor_val(out)
                    qn = _quant_node(out)
                    # 진단: 모든 그래프 출력의 실체를 남긴다. 상태 출력이 왜 안 잡히는지
                    # 구조를 보지 않고는 못 고친다 (2026-09-10).
                    _v = out.meta.get("val")
                    logging.info(
                        "[STATEMETA] idx=%d name=%s target=%s val=%s qnode=%s chain=%s",
                        idx, out.name, str(out.target)[:48],
                        (tuple(_v.size()) if torch.is_tensor(_v)
                         else f"{type(_v).__name__}[{len(_v)}]" if isinstance(_v, (tuple, list))
                         else type(_v).__name__),
                        (str(qn.target)[:32] if qn is not None else "없음"),
                        "->".join(
                            str(x.target)[:24] if isinstance(x, torch.fx.Node) else "?"
                            for x in [out,
                                      out.args[0] if out.args and isinstance(out.args[0], torch.fx.Node) else None,
                                      (out.args[0].args[0] if out.args and isinstance(out.args[0], torch.fx.Node)
                                       and out.args[0].args and isinstance(out.args[0].args[0], torch.fx.Node) else None)]
                            if x is not None),
                    )
                    if val is None and qn is not None:
                        val = _tensor_val(qn)
                    if (
                        val is not None
                        and qn is not None
                        and len(qn.args) >= 6
                        and tuple(val.size()[-2:]) in state_shape
                    ):
                        # [scale, zero_point, quant_min, quant_max, dtype]
                        self.meta[f"get_state_output_{idx}_quant_attr"] = [
                            qn.args[1],
                            qn.args[2],
                            qn.args[3],
                            qn.args[4],
                            str(qn.args[5]),
                        ]
                        # 러너가 버퍼를 채울 때 쓰는 값. 위 리스트는 이종 타입이라
                        # C++ 에서 읽기 번거로우므로 정수 하나로도 따로 내보낸다.
                        self.meta[f"get_state_output_{idx}_zero_point"] = int(qn.args[2])
                        saved += 1
                    elif val is not None and tuple(val.size()[-2:]) in state_shape:
                        logging.warning(
                            "state output %d (%s): 상태 모양인데 q/dq 노드를 못 찾음 "
                            "— zero_point 메타 누락, 기기에서 uint16 상태 초기화 실패한다",
                            idx,
                            out.name,
                        )
                    # logits 를 제외한 캐시/상태 출력만 센다
                    if out is not node.args[0][0]:
                        idx += 1
                break
            logging.info(f"state output quant attrs saved: {saved}")
            if saved == 0:
                # 상태 슬롯이 있는데 하나도 못 건졌다면 러너가 uint16 상태를 초기화하지
                # 못한다. 조용히 넘어가면 기기에서 assert 로 죽으므로 크게 남긴다.
                logging.error(
                    "state quant attrs: 0 saved despite %d state shapes — "
                    "uint16 state IO 는 이 메타 없이는 기기에서 실패한다",
                    len(state_shape),
                )
        except Exception as exc:  # noqa: BLE001
            logging.error(f"failed to save state quant attrs: {exc}", exc_info=True)

    def _linear_attention_state_shapes(self):
        """Gated DeltaNet(선형 어텐션) 층의 상태 슬롯 shape 집합.

        _tag_ios 는 shape 로 그래프 IO 를 골라 양자화 dtype 을 붙인다. KV 캐시 shape 만
        등록돼 있어 DeltaNet 상태는 태깅되지 않고 float32 로 노출되며, 러너와 NPU 사이
        전송량이 두 배가 된다(4B 모델 기준 토큰당 약 105MB). 여기서 상태 shape 을 만들어
        16bit 로 태깅한다.
        """
        # 기본값은 0(float32) 이다. 16bit 로 켜려면 러너 지원이 먼저 필요하다:
        # 상태 인코딩은 비대칭이라 zero_point 가 0 이 아니므로(실측 8840~45008),
        # 러너가 상태 버퍼를 0 바이트로 두면 초기 상태가 0 이 아니라
        # -zero_point*scale (약 -29) 이 되어 첫 토큰부터 결과가 틀어진다.
        # _save_output_state_quant_attrs() 가 zero_point 를 메타로 내보내므로
        # kv_manager 의 init_cache 가 그 값으로 채우도록 고친 뒤에 켤 것.
        if os.environ.get("QNN_STATE_IO_16BIT", "0") == "0":
            return set()
        raw = getattr(self, "_state_shapes_raw", None)
        if raw is None:
            # 아직 캐시가 없으면(호출 순서가 바뀐 경우) 한 번 더 시도해 본다.
            self._cache_linear_attention_state_shapes()
            raw = self._state_shapes_raw
        # KV 캐시와 우연히 같은 shape 이면 KV 규칙을 우선한다.
        # 뺄셈은 사용 시점에 한다 - kv_cache_shape 는 _prepare_model 뒤에 채워진다.
        return raw - set(self.kv_cache_shape)

    def _cache_linear_attention_state_shapes(self):
        """상태 슬롯 shape 을 self.decoder 가 아직 nn.Module 일 때 계산해 둔다.

        _tag_ios 는 마지막 두 축으로 그래프 IO 를 비교하므로 [-2:] 만 남긴다.
        """
        self._state_shapes_raw = set()
        self._state_shape_conv = None
        self._state_shape_rec = None
        # 층 -> 캐시 슬롯 지도. QNN_DELTA_MUTBUF=1 이면 DeltaNet 층이 슬롯을 안 쓰므로
        # args_i 의 i 가 층 번호가 아니라 슬롯 번호가 된다 (2026-09-17 KST).
        self._kv_slot_of_layer = list(
            getattr(self.decoder, "kv_slot_of_layer", []) or []
        )
        layers = getattr(self.decoder, "layers", None)
        if layers is None:
            return
        try:
            layer_list = list(layers)
        except TypeError:
            # export 이후의 GraphModule. 이 경로로 들어왔다면 호출 시점이 잘못된 것이다.
            logging.warning(
                "state shapes requested after export; cache is empty"
            )
            return
        for layer in layer_list:
            attn = getattr(layer, "attention", None)
            if attn is None or not hasattr(attn, "state_shapes"):
                continue
            conv_shape, rec_shape = attn.state_shapes(self.decoder.max_batch_size)
            # conv 상태는 k 슬롯, recurrent 상태는 v 슬롯에 실린다. 인코딩을 층별로
            # 짝지으려면 둘을 구분해야 하므로 따로도 보관한다.
            self._state_shape_conv = tuple(conv_shape[-2:])
            self._state_shape_rec = tuple(rec_shape[-2:])
            self._state_shapes_raw = {
                self._state_shape_conv,
                self._state_shape_rec,
            }
            if os.environ.get("QNN_STATE_SNAPSHOTS") == "1" and getattr(attn, "deferred", False):
                self._state_shapes_raw.add((attn.conv_dim, attn.conv_kernel_size - 1 + attn.ar_len))
            # QNN_DEFER_EXACT: the deferred-commit ingredients cross the call boundary as 16-bit graph IO
            self._ingredient_shapes_raw = set()
            # QNN_DEFER_EXACT_U16IO=0 keeps float32 ingredient IO (encodings still forced; 2026-10-11 20:2x KST test of
            # whether 16-bit ingredient graph IO is what made the exact export's NPU time +4.7 ms at width 16)
            if (os.environ.get("QNN_DEFER_EXACT", "0") == "1" and os.environ.get("QNN_DEFER_EXACT_U16IO", "1") == "1"
                    and getattr(attn, "deferred", False)):
                self._ingredient_shapes_raw = {tuple(sh[-2:]) for sh in attn.ingredient_shapes(self.decoder.max_batch_size)}
            logging.info(
                f"[{self.mode}] linear-attention state shapes cached: "
                f"{sorted(self._state_shapes_raw)}"
            )
            return

    def _tag_ios(self, node, fixed_point_type):  # noqa: C901
        if not torch.is_tensor(node.meta.get("val")):
            # 다중 출력(튜플) 노드는 자체 val 이 텐서가 아니다. getitem 사용자가 태깅된다.
            return None
        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                1,  # num_heads=1: the mask broadcasts across all heads.
                self.meta["get_ar_len"],
                self.meta["get_max_context_len"],
            ),
        }

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }
        if self.control_args.decoder_model == "gemma4-e2b":
            freq_shape.add(
                (self.meta["get_ar_len"], self.meta["get_global_head_dim"] // 2)
            )

        freq_op = {exir_ops.edge.aten.select.int, exir_ops.edge.aten.select_copy.int}
        # DeltaNet 상태 슬롯(선형 어텐션). KV 캐시와 shape 이 달라 별도로 잡아야 한다.
        state_shape = self._linear_attention_state_shapes()
        ingredient_shape = getattr(self, "_ingredient_shapes_raw", set()) - set(self.kv_cache_shape) - set(state_shape)
        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) > 0
                and "args_" in node.name
                and (torch.is_tensor(users[0].meta["val"]) and users[0].meta["val"].size()[-2:] in self.kv_cache_shape)
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif (
                len(list(node.users)) > 0
                # mutable buffer 모드(QNN_DELTA_MUTBUF=1)에서는 상태가 args_ 가 아니라
                # buffer placeholder 로 들어온다. 이름이 b_..._state_buf 라 여기서 같이
                # 잡지 않으면 입력만 float 로 남아, 델리게이트 출력(uint16)과 dtype 이
                # 어긋나 CPU copy_ 가 삽입되고 실행이 죽는다 (2026-09-17 기기 실측).
                and ("args_" in node.name or "state_buf" in node.name)
                and tuple(node.meta["val"].size()[-2:]) in state_shape
            ):
                # 상태는 매 스텝 자기 자신으로 되먹임되어 오차가 복리로 쌓이므로
                # KV 와 달리 8bit 로 내리지 않고 16bit(io_type)를 쓴다.
                #
                # KV 캐시와 달리 **placeholder 자신의 shape** 으로 판정한다.
                # KV 는 입력(max_context_len)과 첫 사용자 cat(=+ar_len)의 shape 이 달라
                # 사용자 shape 을 봐야 하지만, 상태는 입력과 출력 shape 이 같다.
                # 사용자 shape 으로 보면 conv 상태 (1,8192,3) 의 첫 사용자가
                # cat -> (1,8192,4) 라 절대 매칭되지 않아, 입력 24개가 태깅에서
                # 빠지고 출력 24개만 uint16 이 되는 비대칭이 생긴다(실측 확인).
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif len(list(node.users)) > 0 and "args_" in node.name and tuple(node.meta["val"].size()[-2:]) in ingredient_shape:
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in self.kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif tuple(node.meta["val"].size()[-2:]) in state_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif tuple(node.meta["val"].size()[-2:]) in ingredient_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # Grouped state concatenation may be partitioned into the last shard.
        # Preserve packed state on those intermediate edges as well: otherwise
        # InsertIOQDQ exports float state and requantizes it in the final shard.
        if (
            node.target == operator.getitem
            and node.args[1] == 1
            and isinstance(node.args[0], torch.fx.Node)
            and hasattr(node.args[0].target, "_schema")
            and node.args[0].target._schema.name in {
                "qwen35::delta_rule_state16", "qwen35::delta_rule_decode16",
                "qwen35::delta_rule_state16_masked", "qwen35::delta_rule_snapshots16",
            }
        ):
            quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        # Matched against stack_trace source text, so this is coupled to the
        # variable names in model forward: renaming per_layer_inputs or
        # changing how it is sliced silently drops the tag.
        if (
            self.control_args.decoder_model == "gemma4-e2b"
            and "stack_trace" in node.meta
            and (
                "per_layer_inputs[n_self:]" in node.meta["stack_trace"]
                or "per_layer_inputs[i]" in node.meta["stack_trace"]
            )
        ):
            quant_io_type = fixed_point_type["io_type"]

        # Tag the donor K/V that YOCO shared layers consume.
        #
        # Matched on full_k/full_v are the local variable names in model forward.
        # Renaming them drops the tag.
        if self.control_args.decoder_model == "gemma4-e2b" and (
            is_node_src_start_with_name(node, "full_k")
            or is_node_src_start_with_name(node, "full_v")
        ):
            quant_io_type = fixed_point_type["io_type"]
        return quant_io_type

    def _quant_recipe_suggestion(
        self,
        fp32_gm: torch.fx.GraphModule,
        qdq_gm: torch.fx.GraphModule,
        text_dataloader: DataLoader,
        recipe: StaticLLMQuantRecipe,
    ):
        """
        Compare fp32 vs QDQ intermediate outputs and write SQNR reports.

        fp32_gm: Fp32 exported GraphModule (before prepare_pt2e).
        qdq_gm: QDQ GraphModule (after convert_pt2e).

        Output files:
          ``{model_name}_quantization_error.csv``: per-group statistics
          ``{model_name}_suggest_recipe.py``: Python script containing quantization recipe classes
        based on the suggested quant recipe overrides.
        """
        model_name = self.control_args.decoder_model
        report = PerLayerSqnrAnalyzer(
            model_name=model_name,
            num_layers=self.meta["get_n_layers"],
            fp32_gm=fp32_gm,
            qdq_gm=qdq_gm,
            analysis_recipe=recipe,
        ).analyze(
            self._decoder_inference,
            text_dataloader,
        )
        report.save_analysis_summary()
        suggest_recipe_overrides = report.suggest_recipe_overrides()
        save_suggest_recipes(report, suggest_recipe_overrides)

    @log_info
    def quantize(self, request: Request):  # noqa: C901
        if self.quant_recipe is None:
            return

        if self.decoder is None or (
            self.apply_embedding and self.tok_embedding is None
        ):
            return

        data = request.method_data[TEXT_DECODER]
        # check bit width graph io
        fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
        if data.skip_quantize:
            # already init as float32
            return
        else:
            if self.quant_recipe.get_kv_io_bit_width() == 8:
                fixed_point_type["kv_type"] = torch.uint8
            elif self.quant_recipe.get_kv_io_bit_width() == 16:
                fixed_point_type["kv_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
                )

            if self.quant_recipe.get_logits_output_bit_width() == 16:
                fixed_point_type["io_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
                )

        # embedding fallback and quantization
        if self.control_args.embedding_quantize:
            self.decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(self.decoder)
            if self.apply_embedding:
                self.tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(self.tok_embedding)

        data = request.method_data[TEXT_DECODER]

        quantizer = make_quantizer(backend=data.backend, soc_model=data.soc_model)
        quantizer.set_recipe(self.quant_recipe.recipe)
        if (
            os.environ.get("QNN_DELTA_KERNEL", "0") == "1"
            and os.environ.get("QNN_DELTA_KERNEL_FLOAT_IO", "0") != "1"
        ):
            # 델타 규칙 커스텀 op 의 입출력 양자화 스펙 (state 는 float 유지).
            # QNN_DELTA_KERNEL_FLOAT_IO=1 이면 어노테이션을 생략해 op 입출력을 전부 float 로 둔다
            # (커널의 PlainFloat 등록 경로 검증용).
            from executorch.examples.qualcomm.oss_scripts.llama.model.delta_rule_op import (
                build_annotation_fn,
            )
            quantizer.add_custom_quant_annotations((build_annotation_fn(),))
        tok_embedding_quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            backend=data.backend,
            soc_model=data.soc_model,
        )

        use_qat = self.control_args.qat and self.mode == Mode.CALIBRATE
        with torch.no_grad():
            graph_module = None
            self.decoder = torch.export.export(
                self.decoder, self.export_input, strict=True
            ).module()
            if self.mode == Mode.CALIBRATE and (
                self.control_args.quant_recipe_suggestion or use_qat
            ):
                graph_module = copy.deepcopy(self.decoder)
            if self.apply_embedding:
                self.tok_embedding = torch.export.export(
                    self.tok_embedding,
                    self.tok_embedding.get_example_input(),
                    strict=True,
                ).module()

            if (
                self.control_args.verbose
                and self.mode == Mode.CALIBRATE
                and not self.apply_embedding
            ):
                run_lm_eval(
                    module=self.decoder,
                    get_example_inputs=self.get_example_inputs,
                    tokenizer=data.tokenizer,
                    max_seq_length=self.meta["get_max_context_len"],
                    tasks=self.control_args.eval_tasks,
                    use_i64_token=self.control_args.embedding_quantize is not None,
                    num_fewshot=self.control_args.eval_num_fewshot,
                    limit=self.control_args.eval_limit,
                    max_batch_size=self.meta["get_max_batch_size"],
                    event_name="export_tasks",
                )

            if use_qat:
                self.decoder = prepare_qat_pt2e(self.decoder, quantizer)
                move_exported_model_to_train(self.decoder)
            else:
                self.decoder = prepare_pt2e(self.decoder, quantizer)
            if self.apply_embedding:
                self.tok_embedding = prepare_pt2e(
                    self.tok_embedding, tok_embedding_quantizer
                )

            # calibrate_ar1 models: the prefill(ar=N) graph's unrolled op count
            # differs from the ar=1 graphs, so decode->prefill encoding copy is
            # impossible. Self-calibrate the prefill graph on the same data
            # (chunked stepping); its IO encodings are then unified with decode
            # in HybridTextDecoder.compile via _override_io_encodings.
            prefill_self_calibrate = self.mode == Mode.PREFILL and getattr(
                self.config, "calibrate_ar1", False
            )
            if self.mode == Mode.CALIBRATE or prefill_self_calibrate:
                calibration_dataloaders = {
                    AUDIO_ENCODER: request.method_data[
                        AUDIO_ENCODER
                    ].quantization_data.intermediate_outputs,
                    VISION_ENCODER: request.method_data[
                        VISION_ENCODER
                    ].quantization_data.intermediate_outputs,
                    TEXT_DECODER: data.quantization_data.calib_loader,
                }

                if use_qat:
                    training_args = TrainingArgs.from_yaml(
                        self.control_args.train_config
                    )
                    training_args.lr_config = self.control_args.lr_config
                    frozen = (
                        [".*"]
                        if self.control_args.freeze_all_params
                        # freeze_all_params: CI-only flag to verify QAT vs PTQ accuracy difference
                        # by disabling weight updates and only updating scale/zero_point.
                        else getattr(self.quant_recipe, "frozen_param_patterns", None)
                    )
                    QATStrategy(
                        inference=self._decoder_inference,
                        module=self.decoder,
                        tok_embedding=self.tok_embedding,
                        seq_mse_candidates=self.config.seq_mse_candidates,
                    ).quantize(
                        calib_loader=calibration_dataloaders,
                        training_args=training_args,
                        teacher=graph_module,
                        train_loader=data.quantization_data.train_loader,
                        val_loader=data.quantization_data.val_loader,
                        frozen_param_patterns=frozen,
                    )
                    logging.info("QAT training complete")
                else:
                    PTQStrategy(
                        inference=self._decoder_inference,
                        module=self.decoder,
                        tok_embedding=self.tok_embedding,
                        seq_mse_candidates=self.config.seq_mse_candidates,
                    ).quantize(calib_loader=calibration_dataloaders)
                    logging.info("Calibration complete")
            else:
                # one dummy inference to remove affine observer
                # error happened in convert_pt2e
                self.decoder(*self.export_input)

            self.decoder = convert_pt2e(self.decoder)

            if (
                self.mode == Mode.CALIBRATE
                and self.control_args.quant_recipe_suggestion
            ):
                self._quant_recipe_suggestion(
                    graph_module,
                    self.decoder,
                    calibration_dataloaders[TEXT_DECODER],
                    self.quant_recipe.recipe,
                )

            # FP32 model used as QAT teacher or quant-recipe-suggestion reference; release after use.
            del graph_module
            gc.collect()

            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)

            if (
                self.control_args.verbose
                and self.mode == Mode.CALIBRATE
                and not self.apply_embedding
            ):
                run_lm_eval(
                    module=self.decoder,
                    get_example_inputs=self.get_example_inputs,
                    tokenizer=data.tokenizer,
                    max_seq_length=self.meta["get_max_context_len"],
                    tasks=self.control_args.eval_tasks,
                    use_i64_token=self.control_args.embedding_quantize is not None,
                    num_fewshot=self.control_args.eval_num_fewshot,
                    limit=self.control_args.eval_limit,
                    max_batch_size=self.meta["get_max_batch_size"],
                    event_name="convert_pt2e_tasks",
                )

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = partial(
                self._tag_ios, fixed_point_type=fixed_point_type
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
    ):
        if os.environ.get("QNN_PREFILL_VALID", "0") == "1" and not getattr(config, "calibrate_ar1", False):
            # The non-calibrate_ar1 prefill encoding override pairs args_ placeholders by count.
            raise ValueError("QNN_PREFILL_VALID=1 requires a calibrate_ar1 model config")
        if os.environ.get("QNN_DEFER_STATE", "0") == "1":
            if not getattr(config, "calibrate_ar1", False):
                raise ValueError("QNN_DEFER_STATE=1 requires a calibrate_ar1 model config")
            if os.environ.get("QNN_PREFILL_LAST_ROW", "0") == "1" or os.environ.get("QNN_DELTA_MUTBUF", "0") == "1":
                raise ValueError("QNN_DEFER_STATE=1 needs all verify rows' logits and graph-IO states "
                                 "(unset QNN_PREFILL_LAST_ROW and QNN_DELTA_MUTBUF)")
        self.decode = TextDecoder(
            control_args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding,
        )
        # Full AR sequence with KV cache; used only for quantization.
        # Scales/zp collected here are propagated to decode and prefill graphs via _encoding_override.
        self.calibration_prefill = TextDecoder(
            control_args,
            config,
            Mode.CALIBRATE,
            apply_embedding=apply_embedding,
        )

        self.control_args = control_args
        self.config = config
        self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding

    def _override_io_encodings(self, src_model, dst_model):
        """Copy only the graph-interface encodings (every input placeholder's
        quantize params and every output's q/dq params) from src to dst,
        positionally. Used for calibrate_ar1 models whose prefill graph is
        self-calibrated: internal activation scales stay as calibrated, but the
        shared interface (tokens/mask/pos, KV caches, DeltaNet states, logits)
        must quantize identically across the two graphs."""
        qdq_target = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.torchao.quantize_affine,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }

        def copy_chain(src_node, dst_node):
            for su, du in zip(list(src_node.users), list(dst_node.users)):
                if su.name == "output" or du.name == "output":
                    continue
                if su.target in qdq_target and du.target == su.target:
                    du.args = (du.args[0], *su.args[1:])
                    copy_chain(su, du)

        src_ph = [n for n in src_model.graph.nodes if n.op == "placeholder"]
        dst_ph = [n for n in dst_model.graph.nodes if n.op == "placeholder"]
        n_extra = 0
        deferred = getattr(self.prefill, "defer_state", False) and dst_model is self.prefill.decoder
        n_defer_out = 0
        if getattr(self.prefill, "prefill_valid", False) and dst_model is self.prefill.decoder:
            n_extra = 3 if getattr(self.prefill, "prefill_last_row", False) else 2
        if deferred:
            # valid_prev, 5 ingredients per DeltaNet layer, valid_cur, conv_idx_prev; the ingredient
            # inputs/outputs are prefill-only float tensors and keep their own calibrated encodings.
            n_defer_out = 5 * self.prefill.defer_n_delta
            n_extra = 3 + n_defer_out
            if os.environ.get("QNN_STATE_SNAPSHOTS") == "1":
                n_defer_out = 0
        assert len(src_ph) + n_extra == len(dst_ph), (
            f"IO mismatch: {len(src_ph)} + {n_extra} vs {len(dst_ph)} placeholders"
        )
        extras_ph = dst_ph[len(src_ph):]
        int_pos = {0, len(extras_ph) - 2, len(extras_ph) - 1} if deferred else set(range(len(extras_ph)))
        for i, extra in enumerate(extras_ph):
            val = extra.meta.get("val")
            want = torch.int32 if i in int_pos else torch.float32
            assert val is not None and val.dtype == want, f"prefill extra input {extra.name} must be {want}"
        for s_n, d_n in zip(src_ph, dst_ph):
            copy_chain(s_n, d_n)

        src_out = next(n for n in src_model.graph.nodes if n.op == "output")
        dst_out = next(n for n in dst_model.graph.nodes if n.op == "output")
        src_args = list(src_out.args[0])
        dst_args = list(dst_out.args[0])
        # deferred: the trailing ingredient outputs are prefill-only; unify the common prefix
        # (logits, k/v/state slots) only
        if deferred and int(os.environ.get("QNN_DEFER_CHECKPOINT_ROW", "0")):
            # QNN_DEFER_CHECKPOINT_ROW (2026-10-12 23:4x KST (experiments/ckpt10_w16_20261012)): per-group checkpoint states trail the ingredients (prefill-only
            # outputs; force_shared_state_encoding mode 2 gives them the IO state encoding after this override).
            n_ckpt = len(dst_args) - len(src_args) - n_defer_out
            assert n_ckpt > 0 and n_ckpt % self.prefill.defer_n_delta == 0, (
                f"checkpoint outputs: {len(dst_args)} - {len(src_args)} - {n_defer_out} = {n_ckpt}")
            n_defer_out += n_ckpt
        assert len(src_args) + n_defer_out == len(dst_args), (
            f"output mismatch: {len(src_args)} + {n_defer_out} vs {len(dst_args)}"
        )
        for s_n, d_n in zip(src_args, dst_args):
            # each output is typically dequantize(quantize(x)); copy both.
            if s_n.target in qdq_target and d_n.target == s_n.target:
                d_n.args = (d_n.args[0], *s_n.args[1:])
                s_q, d_q = s_n.args[0], d_n.args[0]
                if (
                    hasattr(s_q, "target")
                    and s_q.target in qdq_target
                    and d_q.target == s_q.target
                ):
                    d_q.args = (d_q.args[0], *s_q.args[1:])
        dst_model.recompile()

    def _override_mutbuf_encodings(self, src_model, dst_model):
        """calibrate_ar1 의 AR=N prefill 은 자체 캘리브레이션이라 버퍼 상태 인코딩이 decode 와
        다르다. _override_io_encodings 는 placeholder/output 만 다루므로(버퍼는 get_attr,
        갱신값은 copy_) 여기서 decode(src) 의 값으로 prefill(dst) 의 버퍼 입력·갱신값을 강제한다.
        (2026-09-18 KST, 기기 실측: decode 33984 통일 뒤에도 prefill 32883 로 남아 첫 토큰부터 붕괴)"""
        S, Dt = mutbuf_pairs(src_model), mutbuf_pairs(dst_model)
        done = 0
        for key, (d_ph, d_src) in sorted(Dt.items()):
            src = S.get(key)
            ref = mutbuf_q_args(src[0]) if src else None
            if ref is None:
                logging.warning(f"mutbuf prefill override: no decode reference for {key}; skipped")
                continue
            scale, zp = ref
            counts = [mutbuf_force_q(d_ph, scale, zp), mutbuf_force_q(d_src, scale, zp)]
            done += 1
            logging.info(f"mutbuf prefill override L{key[0]} {key[1]}: forced {ref}; q users fixed (in,out)={counts}")
        logging.info(f"mutbuf prefill state encoding override: {done}/{len(Dt)} pairs (decode pairs {len(S)})")
        dst_model.recompile()

    def _encoding_override(  # noqa: C901
        self,
        quantized_model,
        unquantized_model,
        override_kv_cache,
    ):
        pbq_target = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.torchao.quantize_affine,
        }
        pcq_target = {
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        }
        ptq_target = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }
        qdq_target = pbq_target | pcq_target | ptq_target

        def compare_nodes(quantized_node, unquantized_node):
            def info(node):
                return node.name + (
                    str(node.meta["nn_module_stack"].values())
                    if node.op == "call_function"
                    else ""
                )

            assert info(quantized_node) == info(
                unquantized_node
            ), f"found unmatched order for ops: {quantized_node} vs {unquantized_node}"

        def resolve_param_target(node):
            return (
                node
                if node.op == "call_function" and node.target not in qdq_target
                else resolve_param_target(list(node.users)[0])
            )

        def activation_override(quantized_node, unquantized_node):
            # 두 그래프의 사용자를 '순서'로 짝지으면 그래프 모양이 조금만 달라도 어긋난다.
            # 다중 출력 커스텀 op 가 끼면 실제로 어긋난다 (2026-09-10 실측:
            # quantize_per_tensor 와 qwen35.delta_rule 이 짝이 되어 assert 로 export 중단).
            # 우리가 옮기려는 것은 q/dq 노드의 인코딩 인자뿐이므로 target 으로 짝짓는다.
            q_users = [u for u in quantized_node.users if u.name != "output"]
            u_users = [u for u in unquantized_node.users if u.name != "output"]
            paired = []
            remaining = list(u_users)
            for qu in q_users:
                match = next((x for x in remaining if x.target == qu.target), None)
                if match is None:
                    continue
                remaining.remove(match)
                paired.append((qu, match))
            if len(paired) < len(q_users):
                logging.debug(
                    "activation override: %d/%d users matched (%s vs %s)",
                    len(paired), len(q_users),
                    [str(u.target)[:24] for u in q_users],
                    [str(u.target)[:24] for u in u_users],
                )
            for quantized_user, unquantized_user in paired:
                if quantized_user.target in qdq_target:
                    unquantized_user.args = (
                        unquantized_user.args[0],
                        *quantized_user.args[1:],
                    )
                    activation_override(quantized_user, unquantized_user)

        def parameter_override(quantized_node, unquantized_node):
            # Some parameters need to be iterated over to retrieve attributes such as static_llama.tok_embedding.weight
            def _get_attr(graph_module: torch.fx.GraphModule, target: str) -> Any:
                attr: Any = graph_module
                for target_atom in target.split("."):
                    attr = getattr(attr, target_atom)
                return attr

            def _set_attr(
                graph_module: torch.fx.GraphModule, target: str, replacement: Any
            ) -> Any:
                attr: Any = graph_module
                target_list = target.split(".")
                for target_atom in target_list[:-1]:
                    attr = getattr(attr, target_atom)
                setattr(attr, target_list[-1], replacement)

            _set_attr(
                unquantized_model,
                unquantized_node.target,
                _get_attr(quantized_model, quantized_node.target),
            )
            # scale / zero point are part of op's attributes
            if list(quantized_node.users)[0].target in ptq_target:
                activation_override(quantized_node, unquantized_node)

        # copy encoding for hybrid mode
        parameters = [
            {
                n: resolve_param_target(n)
                for n in model.graph.nodes
                if n.op == "get_attr"
            }
            for model in (quantized_model, unquantized_model)
        ]
        activations = [
            [
                n
                for n in model.graph.nodes
                if n.target not in qdq_target
                and n.op in {"call_function", "placeholder"}
            ]
            for model in (quantized_model, unquantized_model)
        ]
        # check topology order by node name & nn_module_stack
        for act_quantized, act_unquantized in zip(*activations):
            compare_nodes(act_quantized, act_unquantized)

        for op_quantized, op_unquantized in zip(*[p.values() for p in parameters]):
            compare_nodes(op_quantized, op_unquantized)
        # perform encoding override
        for act_quantized, act_unquantized in zip(*activations):
            activation_override(act_quantized, act_unquantized)

        for param_quantized, param_unquantized in zip(*[p.keys() for p in parameters]):
            parameter_override(param_quantized, param_unquantized)

        if override_kv_cache:
            n_cache_args = sum(
                1
                for n in unquantized_model.graph.nodes
                if n.op == "placeholder" and "args_" in n.name
            )
            k_input_cache_nodes = []
            v_input_cache_nodes = []
            for node in unquantized_model.graph.nodes:
                if node.op != "placeholder":
                    continue

                if "args_" in node.name:
                    args_idx = int(node.name.split("_")[-1])

                    # 캐시 인자는 [K 슬롯 전부][V 슬롯 전부] 순서다. 층 수가 아니라
                    # 실제 인자 개수의 절반으로 갈라야 mutable buffer 모드에서도 맞는다.
                    n_cache_layers = n_cache_args // 2
                    if args_idx >= n_cache_layers:
                        v_input_cache_nodes.append(node)
                    else:
                        k_input_cache_nodes.append(node)

            if not k_input_cache_nodes or not v_input_cache_nodes:
                raise RuntimeError(
                    "KV cache input detection failed. This likely means the model naming "
                    "does not match expected prefixes."
                )

            # 입력 placeholder 의 층 번호. args_i 는 i<n_cache_layers 이면 층 i 의 K,
            # 아니면 층 (i-n_cache_layers) 의 V 슬롯이다.
            slot_map = getattr(self, "_kv_slot_of_layer", None) or []
            slot_to_layer = {
                slot: layer
                for layer, slot in enumerate(slot_map)
                if slot is not None
            }

            def _layer_of(slot):
                return slot_to_layer.get(slot, slot)

            k_input_by_layer = {
                _layer_of(int(n.name.split("_")[-1])): n for n in k_input_cache_nodes
            }
            v_input_by_layer = {
                _layer_of(int(n.name.split("_")[-1]) - n_cache_layers): n
                for n in v_input_cache_nodes
            }

            k_output_cache_nodes = []
            v_output_cache_nodes = []
            for node in quantized_model.graph.nodes:
                if not is_graph_output(node):
                    continue
                # Float recurrent outputs (including grouped state cat nodes)
                # are not dequantize(quantize(KV)). Do not unwrap their args.
                if node.target != torch.ops.quantized_decomposed.dequantize_per_tensor.default:
                    continue
                cache_output_node = node.args[0].args[0]
                if is_node_src_start_with_name(cache_output_node, kv_cache_prefix="k_"):
                    k_output_cache_nodes.append(cache_output_node)
                elif is_node_src_start_with_name(
                    cache_output_node, kv_cache_prefix="v_"
                ):
                    v_output_cache_nodes.append(cache_output_node)

            if not k_output_cache_nodes or not v_output_cache_nodes:
                raise RuntimeError(
                    "KV cache detection failed. This likely means the model naming "
                    "does not match expected prefixes."
                )

            # 층 번호로 짝짓는다.
            #
            # 예전 코드는 zip 으로 '위치' 를 짝지었다. k_input 은 모든 층의 K 슬롯인데
            # k_output 은 이름이 "k_" 로 시작하는 출력, 즉 **어텐션 층만** 이라 길이가
            # 다르다. Qwen3.5 같은 하이브리드에서는 args_0(층0 DeltaNet conv 상태)이
            # 층3 어텐션 K 캐시의 8bit 인코딩을 받아버렸고, 그 결과
            #   - 상태 placeholder 가 UFIXED_8 로 추론되어 QNN 이 mul/cat 을 거부
            #     (0x408 != 0x416) -> DeltaNet 앞 6개 층이 CPU 폴백
            #   - 어텐션 층도 엉뚱한 층의 KV 인코딩을 받음 (args_3 <- 층15)
            # 가 동시에 일어났다. 순수 어텐션 모델에서는 두 리스트 길이가 같아
            # 우연히 맞아떨어졌기 때문에 드러나지 않았다.
            if os.environ.get("QNN_KV_LAYER_MATCH", "1") == "0":
                # 원인 분리용: 2026-09-05 배포본과 동일한 '위치' zip 동작.
                logging.info("QNN_KV_LAYER_MATCH=0: legacy positional zip pairing")
                for i_k, o_k in zip(k_input_cache_nodes, k_output_cache_nodes):
                    activation_override(o_k, i_k)
                for i_v, o_v in zip(v_input_cache_nodes, v_output_cache_nodes):
                    activation_override(o_v, i_v)
                k_output_cache_nodes = v_output_cache_nodes = []
            overridden = 0
            for out_node, in_by_layer, tag in (
                (k_output_cache_nodes, k_input_by_layer, "k"),
                (v_output_cache_nodes, v_input_by_layer, "v"),
            ):
                for cache_out in out_node:
                    layer = node_layer_idx(cache_out)
                    cache_in = in_by_layer.get(layer)
                    if layer is None or cache_in is None:
                        logging.warning(
                            f"kv encoding override: no {tag} input slot for layer "
                            f"{layer} ({cache_out.name}); skipped"
                        )
                        continue
                    activation_override(cache_out, cache_in)
                    overridden += 1
            logging.info(
                f"kv cache encoding override: {overridden} slots matched by layer "
                f"(k_out={len(k_output_cache_nodes)}, v_out={len(v_output_cache_nodes)}, "
                f"k_in={len(k_input_cache_nodes)}, v_in={len(v_input_cache_nodes)})"
            )

            # --- DeltaNet 상태 슬롯도 입력 인코딩 = 출력 인코딩 으로 맞춘다 ---
            #
            # 러너의 update_state 는 output_buffer -> buffer 를 raw memcpy 한다
            # (runner/kv_manager.cpp). float IO 일 때는 그래프가 출력에서 dequant,
            # 입력에서 requant 를 하므로 인코딩이 달라도 값이 보존됐다. 그러나 상태 IO 를
            # 16bit 로 태깅하면 **정수를 그대로 복사**하므로 두 인코딩이 다르면
            # 값이 통째로 어긋난다(기기 실측: 생성이 전부 깨짐).
            # 어텐션 KV 가 위에서 하는 것과 같은 처리를, 각 층의 '자기' 상태 출력과
            # 짝지어 수행한다.
            conv2 = getattr(self.decode, "_state_shape_conv", None)
            rec2 = getattr(self.decode, "_state_shape_rec", None)
            # 이 오버라이드는 **상태 IO 가 uint16 일 때만** 의미가 있다.
            #   - uint16 IO: 러너가 정수를 raw memcpy 하므로 입력==출력 인코딩이 필수.
            #   - float IO : 그래프가 출력에서 dequant, 입력에서 requant 하므로 인코딩이
            #     달라도 값이 보존된다. 여기서 입력 인코딩을 출력 것으로 덮어쓰면
            #     입력 observer 가 실제로 본 범위와 다른 scale/zp 를 심는 것이라
            #     이득은 없고 해만 있다.
            # 2026-09-08 감사에서 "QNN_STATE_IO_16BIT=0 인데도 48개 오버라이드가 실행됨"
            # 이 확인됐다(kvfix 빌드). 기본값을 상태 IO 설정에 종속시킨다.
            # QNN_STATE_ENC_OVERRIDE 를 명시하면 그 값이 우선한다(원인 분리 실험용).
            state_io_16 = os.environ.get("QNN_STATE_IO_16BIT", "0") == "1"
            enc_override = os.environ.get(
                "QNN_STATE_ENC_OVERRIDE", "1" if state_io_16 else "0"
            )
            if enc_override == "0":
                logging.info(
                    "deltanet state encoding override: skipped "
                    f"(QNN_STATE_IO_16BIT={int(state_io_16)}, "
                    f"QNN_STATE_ENC_OVERRIDE={enc_override})"
                )
                conv2 = rec2 = None
            elif not state_io_16:
                # 명시적으로 켰지만 float IO 인 경우: 실험 의도로 보고 경고만 남긴다.
                logging.warning(
                    "deltanet state encoding override forced ON with float state IO; "
                    "this rewrites input encodings the observer did not see"
                )
            if conv2 is not None and rec2 is not None:
                state_overridden = 0
                for node in quantized_model.graph.nodes:
                    if not is_graph_output(node):
                        continue
                    # dq(q(src)) 형태의 양자화된 출력만 대상이다. 커스텀 커널을 쓰면
                    # state 출력이 getitem 을 거쳐 오거나 float 로 남아 args 구조가
                    # 달라진다 (2026-09-10: KeyError 'val' 로 export 중단).
                    if (
                        node.target
                        != torch.ops.quantized_decomposed.dequantize_per_tensor.default
                        or not node.args
                        or not isinstance(node.args[0], torch.fx.Node)
                        or not node.args[0].args
                        or not isinstance(node.args[0].args[0], torch.fx.Node)
                    ):
                        continue
                    state_out = node.args[0].args[0]
                    val = state_out.meta.get("val")
                    if not torch.is_tensor(val):
                        continue
                    shape = tuple(val.size()[-2:])
                    if shape == conv2:
                        slot_map = k_input_by_layer
                    elif shape == rec2:
                        slot_map = v_input_by_layer
                    else:
                        continue
                    layer = node_layer_idx(state_out)
                    state_in = slot_map.get(layer)
                    if layer is None or state_in is None:
                        logging.warning(
                            f"state encoding override: no input slot for layer "
                            f"{layer} ({state_out.name}); skipped"
                        )
                        continue
                    activation_override(state_out, state_in)
                    state_overridden += 1
                logging.info(
                    f"deltanet state encoding override: {state_overridden} slots "
                    f"matched by layer (conv={conv2}, rec={rec2})"
                )

        # --- mutable buffer 상태 (QNN_DELTA_MUTBUF=1): 인코딩 네 개를 하나로 (2026-09-17 KST) ---
        # 상태가 버퍼 placeholder 로 들어오면 위의 args_/k_/v_ 짝짓기에서 빠진다. 그러면
        # prefill 입력·출력, decode 입력·출력이 각자 관측한 scale/zp 를 갖고, 같은 u16 버퍼를
        # 네 가지 인코딩으로 읽고 써서 첫 토큰부터 깨진다 (32층 기기 실측: zp 34003/33984/
        # 32907/32907). 각 층의 copy_(buffer, value) 에서 버퍼와 값의 q 입력을 찾아
        # prefill 출력 인코딩으로 나머지 셋을 덮어쓴다.
        if override_kv_cache and os.environ.get("QNN_DELTA_MUTBUF", "0") == "1":
            P, Dm = mutbuf_pairs(quantized_model), mutbuf_pairs(unquantized_model)
            done = 0
            for key, (p_ph, p_src) in sorted(P.items()):
                ref = mutbuf_q_args(p_src)
                if ref is None:
                    logging.warning(f"mutbuf override: prefill {key} value has no q user; skipped")
                    continue
                before = (mutbuf_q_args(p_ph), ref)
                scale, zp = ref
                counts = [mutbuf_force_q(p_ph, scale, zp), mutbuf_force_q(p_src, scale, zp)]
                d = Dm.get(key)
                if d is not None:
                    counts += [mutbuf_force_q(d[0], scale, zp), mutbuf_force_q(d[1], scale, zp)]
                done += 1
                logging.info(
                    f"mutbuf override L{key[0]} {key[1]}: before {before} -> forced {ref}; "
                    f"q users fixed (p_in,p_out,d_in,d_out)={counts}"
                )
            logging.info(
                f"mutbuf state encoding override: {done}/{len(P)} prefill pairs, decode pairs {len(Dm)}"
            )

        unquantized_model.recompile()

    def quantize(self, request: Request):
        if request.method_data[TEXT_DECODER].skip_quantize:
            return
        self.calibration_prefill.quantize(request)

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # perform encoding override for models to compile
        # ---
        # theoretically decode & prefill model should share the same encoding
        # given that they are using the identical calibration dataset.
        #
        #
        # here we use a mechanism to make sure the encoding align correctly and
        # save AoT quantization time as well.
        # ---
        if not request.method_data[TEXT_DECODER].skip_quantize:
            self._encoding_override(
                quantized_model=self.calibration_prefill.decoder,
                unquantized_model=self.decode.decoder,
                override_kv_cache=True,
            )

            # save logit's quantization attributes to meta
            self.decode._save_logits_quant_attrs()

            # save output KV cache's quantization attributes to meta for attention sink
            self.decode._save_output_kv_cache_quant_attrs()

            # 상태 슬롯(uint16 태깅 시) 의 zero_point 를 러너에 전달하기 위해 저장
            self.decode._save_output_state_quant_attrs()

            if self.apply_embedding:
                self._encoding_override(
                    quantized_model=self.calibration_prefill.tok_embedding,
                    unquantized_model=self.decode.tok_embedding,
                    override_kv_cache=False,
                )

            # Saving Decode QDQ Model EP for SQNR evaluation.
            # 4B 급에서는 이 EP 하나가 20GB 를 넘는다. 저장 뒤에도 참조가 남아 있으면
            # 뒤이어 오는 QNN 컴파일(메모리 정점)과 겹쳐 OOM 으로 죽는다.
            # 저장 즉시 해제하고, QNN_SKIP_QDQ_EP=1 이면 아예 만들지 않는다.
            if os.environ.get("QNN_SKIP_QDQ_EP") == "1":
                logging.info("QNN_SKIP_QDQ_EP=1: skip saving decode QDQ EP")
            else:
                qdq_ep = torch.export.export(
                    self.calibration_prefill.decoder,
                    self.calibration_prefill.export_input,
                    strict=True,
                )
                qdq_ep_path = (
                    f"{self.decode.control_args.artifact}/{DECODE_QDQ_FILENAME}"
                )
                torch.export.save(qdq_ep, qdq_ep_path)
                logging.info(f"QDQ EP saved to {qdq_ep_path}")
                del qdq_ep
                gc.collect()

            if self.apply_embedding:
                self._encoding_override(
                    quantized_model=self.calibration_prefill.tok_embedding,
                    unquantized_model=self.decode.tok_embedding,
                    override_kv_cache=False,
                )

            # For hybrid mode, override encoding of prefill model.
            if (
                self.prefill.decoder is not None
                and self.prefill.meta["get_use_kv_cache"]
            ):
                if getattr(self.config, "calibrate_ar1", False):
                    # Unroll counts differ (DeltaNet): the prefill graph was
                    # self-calibrated; only unify its IO encodings with decode
                    # so the shared KV/state buffers and logits use one scale.
                    self._override_io_encodings(
                        src_model=self.decode.decoder,
                        dst_model=self.prefill.decoder,
                    )
                    if os.environ.get("QNN_DELTA_MUTBUF", "0") == "1":
                        self._override_mutbuf_encodings(
                            src_model=self.decode.decoder,
                            dst_model=self.prefill.decoder,
                        )
                    if getattr(self.prefill, "prefill_valid", False) or getattr(self.prefill, "defer_state", False):
                        n = gather_explicit_requant(self.prefill.decoder)
                        logging.info(f"prefill_valid/defer_state: {n} conv-state gathers requantized explicitly after the Gather")
                    from executorch.examples.qualcomm.oss_scripts.llama.model import delta_rule_op

                    mode = delta_rule_op.shared_state_encoding_mode()
                    if mode:
                        for name, gm in (("decode", self.decode.decoder), ("prefill", self.prefill.decoder)):
                            n, already_io = delta_rule_op.force_shared_state_encoding(gm, mode)
                            logging.info(f"shared state encoding mode {mode} ({name}): {n} delta-op state outputs forced; "
                                         f"calibrated state_in already on the IO encoding for {already_io}")
                        if os.environ.get("QNN_STATE_IO_FROM_Q", "0") == "1":
                            logging.info(f"QNN_STATE_IO_FROM_Q: {len(delta_rule_op.STATE_IO_FIXED)} state input dequantizes "
                                         f"moved onto their quantize's encoding: {sorted(delta_rule_op.STATE_IO_FIXED)[:12]}")
                    if os.environ.get("QNN_DEFER_EXACT", "0") == "1" and getattr(self.prefill, "defer_state", False):
                        counts = delta_rule_op.force_exact_ingredient_io(self.prefill.decoder)
                        logging.info(f"QNN_DEFER_EXACT: ingredient inputs put on their output encodings: {counts}")
                else:
                    self._encoding_override(
                        quantized_model=self.decode.decoder,
                        unquantized_model=self.prefill.decoder,
                        override_kv_cache=True,
                    )

                if self.apply_embedding:
                    self._encoding_override(
                        quantized_model=self.decode.tok_embedding,
                        unquantized_model=self.prefill.tok_embedding,
                        override_kv_cache=False,
                    )

        if os.environ.get("QNN_SNAPSHOT_GROUP_OUTPUTS") == "1":
            from executorch.examples.qualcomm.oss_scripts.llama.model.snapshot_outputs import split_history_outputs
            import json
            from pathlib import Path
            manifest = split_history_outputs(self.prefill.decoder, self.prefill.model_args.layer_types)
            Path(self.control_args.artifact, "prefill_output_roles.json").write_text(json.dumps(manifest, indent=2))
            logging.info("Snapshot V2: exposed %d history groups without concatenation", sum(manifest["groups"].values()))

        # 2026-10-11 KST QNN_QDQ_DUMP=<file>: dump the final quantized prefill graph (every node, its args with node
        # names, users, value shape, decoder layer) for the snapshot-vs-recompute encoding comparison;
        # QNN_QDQ_DUMP_EXIT=1 stops before lowering.
        if os.environ.get("QNN_QDQ_DUMP"):
            import json as _json
            def _arg(a):
                if isinstance(a, torch.fx.Node):
                    return {"node": a.name}
                if isinstance(a, (list, tuple)):
                    return [_arg(x) for x in a]
                if isinstance(a, (int, float, bool, str)) or a is None:
                    return a
                return str(a)
            with open(os.environ["QNN_QDQ_DUMP"], "w") as _f:
                for _n in self.prefill.decoder.graph.nodes:
                    _v = _n.meta.get("val")
                    _f.write(_json.dumps({
                        "name": _n.name, "op": _n.op, "target": str(_n.target), "args": _arg(_n.args),
                        "users": [u.name for u in _n.users],
                        "shape": list(_v.shape) if torch.is_tensor(_v) else None,
                        "layer": node_layer_idx(_n)}) + "\n")
            logging.info(f"QNN_QDQ_DUMP: prefill graph written to {os.environ['QNN_QDQ_DUMP']}")
            if os.environ.get("QNN_QDQ_DUMP_EXIT") == "1":
                logging.info("QNN_QDQ_DUMP_EXIT=1: stopping before lowering")
                os._exit(0)

        # calibration_prefill is only used for encoding override
        # free it once encoding override is complete.
        del self.calibration_prefill
        gc.collect()

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TOK_EMBEDDING]
            models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in models if m is not None
            ]  # tokens
            tok_embedding_graph_names = TOK_EMBEDDING_GRAPH_NAMES[: len(models)]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
        example_inputs = [m.export_input for m in models if m is not None]
        graph_names = DECODER_GRAPH_NAMES[: len(models)]

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding for model in models],
                    )
                ),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_dep_table for model in models],
                    )
                ),
                passes_job=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_passes_job for model in models],
                    )
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
                passes=[BuildQuantIo()],
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TOK_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)

        # decoder lowering
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={
                **self.decode.meta,
                **({"get_prefill_valid": 1} if getattr(self.prefill, "prefill_valid", False) else {}),
                **({"get_prefill_last_row": 1} if getattr(self.prefill, "prefill_last_row", False) else {}),
                **(
                    {"get_defer_state": 1, "get_defer_masked": 1,
                     "get_n_delta_layers": self.prefill.defer_n_delta}
                    if getattr(self.prefill, "defer_state", False) else {}
                ),
            },
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            for graph_name in graph_names:
                update_spill_fill_size(edge_prog_mgr.exported_program(graph_name))

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id

        if config := getattr(config, modality, None):
            if modality == AUDIO_ENCODER:
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
                self.num_layers = auto_model.config.encoder_config.num_layers
                self.ctx_size = auto_model.config.encoder_config.context_size
            elif modality == TEXT_ENCODER:
                raise NotImplementedError(f"{modality} is under development")
            elif modality == VISION_ENCODER:
                auto_model = AutoModel.from_pretrained(
                    repo_id, _attn_implementation="eager"
                )
                self.num_layers = auto_model.config.vision_config.num_hidden_layers
            else:
                raise NotImplementedError(f"Find no {modality}")

            auto_model = auto_model.to(torch.float32).eval()
            self.model = config().create_encoder(auto_model.config).eval()
            self.model.load_state_dict(
                auto_model.state_dict(), strict=False
            )  # set strict to false to simplify parameter loading for non-text models
            self.example_input = self.model.get_example_inputs()

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

            # metadata
            self.config = config

            self._encoder_inference = EncoderInference()

        self.pass_manager_cls = get_qnn_pass_manager_cls()
        self.passes_job = self.pass_manager_cls.get_capture_program_passes()
        self.dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
        )

    def _tag_ios(self, node, fixed_point_type):
        if not torch.is_tensor(node.meta.get("val")):
            # 다중 출력(튜플) 노드는 자체 val 이 텐서가 아니다. getitem 사용자가 태깅된다.
            return None
        quant_io_type = None

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # GraniteSpeech: tag _to_copy op as quantized tensors for attn dist. It is caused by sharding
        if (
            issubclass(self.config, GraniteSpeechEncoder)
            and node.target == exir_ops.edge.aten._to_copy.default
            and node.meta["val"].size() == (self.ctx_size, self.ctx_size)
        ):
            quant_io_type = torch.int32

        return quant_io_type

    def _get_sharding_get_pattern(self):
        prefixes = [
            "encoder.layers",
            "vision_tower.encoder.layer",
            "vision_model.encoder.layers",
        ]
        prefix_alt = "|".join(re.escape(p) for p in prefixes)
        return rf"^(?:{prefix_alt})\.(\d+)"

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        # check if sharding required
        if self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.num_layers,
                shares=self.config.num_sharding,
                pattern=self._get_sharding_get_pattern(),
                shard_layers=getattr(self.config, "shard_layers", None),
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

            if not request_data.skip_quantize:
                fixed_point_type = {"io_type": torch.float32}

                # setup quantized IO
                self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
                self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                    "get_quant_io_dtype_fn"
                ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
            dep_table=self.dep_table,
            passes_job=self.passes_job,
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            update_spill_fill_size(edge_prog_mgr.exported_program())
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(
            ExecutorchBackendConfig(passes=[BuildQuantIo()])
        )
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)

    def _calibrate(self, model, calibration_datasets):
        outputs = []
        for batch in calibration_datasets:
            outputs_each_batch = [
                self._encoder_inference.predict_step(model, data)
                for data in batch["inputs"]
            ]
            outputs.append(outputs_each_batch)
        return DataLoader(
            ModalityEncoderDataset(outputs),
            batch_size=self.control_args.batch_size,
            shuffle=False,
            drop_last=self.control_args.batch_size > 1,
        )

    def quantize(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        calibration_datasets = request_data.quantization_data.calib_loader

        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            if request_data.skip_quantize:
                logging.info(f"skipping encoder quantization for {self.modality}")
                intermediate_outputs = self._calibrate(self.model, calibration_datasets)
                request_data.quantization_data.intermediate_outputs = (
                    intermediate_outputs
                )
                return

            quantizer = make_quantizer(
                backend=request_data.backend, soc_model=request_data.soc_model
            )
            quantizer.set_recipe(self.quant_recipe.recipe)
            self.model = prepare_pt2e(self.model, quantizer)

            # start calibration
            intermediate_outputs = self._calibrate(self.model, calibration_datasets)
            request_data.quantization_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            # QDQ inference
            if self.control_args.verbose:
                qdq_intermediate_outputs = self._calibrate(
                    self.model, calibration_datasets
                )
                request_data.quantization_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.control_args = control_args
        self.config = config
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=self.audio_encoder.model or self.vision_encoder.model,
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TOK_EMBEDDING,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
        skip_quantize: Dict[str, bool],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                    skip_quantize=skip_quantize[m] if m in skip_quantize else False,
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        skip_quantize: Dict[str, bool],
        backend,
        soc_model,
    ):
        # Local fix (2026-09-01): when every modality skips quantization
        # (--use_fp16), do not build calibration dataloaders at all. The lm_eval
        # token collection otherwise runs (and crashes on tokenizers without a
        # BOS id, e.g. Qwen3.5) even though the data would never be used.
        if skip_quantize and all(skip_quantize.values()):
            quantize_request = Request(
                inspect.currentframe().f_code.co_name,
                {
                    m: Request.Data(
                        skip_quantize=True,
                        tokenizer=tokenizer_wrapper.tokenizer,
                    )
                    for m in self._modalities
                },
            )
            self.process(quantize_request)
            return

        data_config = DataConfig.from_args(self.control_args)
        dataset_builder = DatasetBuilder(
            data_config=data_config,
            llm_config=self.config,
            tokenizer_wrapper=tokenizer_wrapper,
            attn_mask=self.text_decoder.calibration_prefill.attn_mask,
        )
        if self.control_args.qat:
            calib_loader, train_loader, val_loader = (
                dataset_builder.build_qat_dataloaders()
            )
        else:
            calib_loader = dataset_builder.build_calib_dataloaders()
            train_loader = dict.fromkeys(calib_loader)
            val_loader = dict.fromkeys(calib_loader)

        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    quantization_data=Request.QuantizationData(
                        calib_loader=calib_loader[m],
                        train_loader=train_loader[m],
                        val_loader=val_loader[m],
                    ),
                    skip_quantize=skip_quantize.get(m, False),
                    tokenizer=tokenizer_wrapper.tokenizer,
                    backend=backend,
                    soc_model=soc_model,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
