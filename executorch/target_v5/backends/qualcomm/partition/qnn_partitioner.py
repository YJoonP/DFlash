# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import os
import operator
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.export.graph_signature import OutputKind
from executorch.backends.qualcomm.builders import node_visitor_manager
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_FALLBACK_NODE,
)

from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import (
    get_current_qnn_manager,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from .common_defs import (
    allow_list_operator,
    constant_operator,
    not_supported_operator,
    to_be_implemented_operator,
)
from .utils import filter_fn, generate_qnn_executorch_option, get_skip_decomp_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _localize_delta_state_outputs(partitions, graph):
    """Keep output-only grouped state concatenation beside its producers."""
    mode = os.environ.get("QNN_DELTA_LOCAL_STATE", "decode")
    if mode not in ("off", "decode", "all"):
        raise ValueError("QNN_DELTA_LOCAL_STATE must be off, decode, or all")
    if mode == "off":
        return partitions
    owners = {node: partition for partition in partitions for node in partition.nodes}
    for node in graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None or schema.name != "aten::cat":
            continue
        if not node.users or any(user.op != "output" for user in node.users):
            continue
        inputs = node.args[0]
        if not inputs or any(
            value.target != operator.getitem or value.args[1] != 1
            or getattr(getattr(value.args[0].target, "_schema", None), "name", None)
            not in {"qwen35::delta_rule_state16", "qwen35::delta_rule_decode16",
                    "qwen35::delta_rule_state16_masked"}
            for value in inputs
        ):
            continue
        if mode == "decode":
            # Full32 measurements favor this placement for AR=1, but regress
            # prefill. Preserve multi-token scheduling until measured separately.
            queries = [value.args[0].args[0] for value in inputs]
            shapes = [getattr(query.meta.get("val"), "shape", ()) for query in queries]
            if any(len(shape) != 4 or not isinstance(shape[1], int) or shape[1] != 1
                   for shape in shapes):
                continue
        source = owners.get(inputs[0])
        destination = owners.get(node)
        if source is None or destination is None or source is destination:
            continue
        if any(owners.get(value) is not source for value in inputs):
            continue
        # Only a graph-output sink moves; this cannot introduce a dependency
        # back into another partition or cross an unsupported operation.
        destination.remove_node(node)
        source.add_node(node)
        owners[node] = source
    if os.environ.get("QNN_SNAPSHOT_GROUP_OUTPUTS") == "1":
        for node in graph.nodes:
            if node.target is not operator.getitem or node.args[1] != 1:
                continue
            producer = node.args[0]
            if getattr(getattr(producer.target, "_schema", None), "name", None) != "qwen35::delta_rule_snapshots16":
                continue
            if any(user.op != "output" for user in node.users):
                raise ValueError(f"Snapshot group {node.name} has a non-output consumer")
            source, destination = owners.get(producer), owners.get(node)
            if source is None or destination is None:
                raise ValueError(f"Snapshot group {node.name} is not delegated")
            if source is not destination:
                destination.remove_node(node); source.add_node(node); owners[node] = source
    return [partition for partition in partitions if partition.nodes]


# mutable buffer 상태의 갱신값(BUFFER_MUTATION 출력)을 생산 샤드 안에 둔다 (2026-09-17 KST).
# split pass 는 층 이름이 없는 통과 노드(getitem/cat/slice/copy)를 마지막 샤드에 배치한다.
# 그러면 층 0 의 새 상태가 샤드 경계를 넘어 마지막 샤드에서 출력되고, 버퍼를 먹은 샤드와
# 갱신값을 내는 샤드가 달라져 _get_new_signature 가 BUFFER_MUTATION 을 델리게이트로 못 옮긴다
# (고아 출력 -> SpecViolationError, 또는 상위 그래프에 토큰당 상태 전체 copy_).
_MUTBUF_PASS_THROUGH = {
    "aten::copy", "aten::cat", "aten::slice_copy", "aten::slice", "aten::clone",
    "aten::alias_copy", "aten::_to_copy", "aten::index_select",
    "quantized_decomposed::dequantize_per_tensor",
    "quantized_decomposed::quantize_per_tensor",
}


def _schema_name(node):
    return getattr(getattr(node.target, "_schema", None), "name", None)


def _localize_mutation_outputs(partitions, graph, edge_program):
    if os.environ.get("QNN_MUTBUF_LOCALIZE", "0") != "1":
        return partitions
    signature = edge_program.graph_signature
    mutation_names = {
        spec.arg.name
        for spec in signature.output_specs
        if spec.kind == OutputKind.BUFFER_MUTATION
    }
    if not mutation_names:
        return partitions
    debug = os.environ.get("QNN_MUTBUF_DEBUG", "0") == "1"
    owners = {node: partition for partition in partitions for node in partition.nodes}
    ids = {id(p): i for i, p in enumerate(partitions)}
    by_name = {node.name: node for node in graph.nodes}
    moved = 0
    for name in sorted(mutation_names):
        node = by_name.get(name)
        if node is None:
            continue
        chain, cur = [], node
        while (
            cur is not None
            and cur.op == "call_function"
            and (cur.target is operator.getitem or _schema_name(cur) in _MUTBUF_PASS_THROUGH)
        ):
            chain.append(cur)
            if cur.target is operator.getitem:
                nxt = cur.args[0]
            elif _schema_name(cur) == "aten::copy":
                nxt = cur.args[1]
            elif _schema_name(cur) == "aten::cat":
                nxt = cur.args[0][0] if cur.args[0] else None
            else:
                nxt = cur.args[0]
            cur = nxt if isinstance(nxt, torch.fx.Node) else None
        producer = cur
        source = owners.get(producer)
        if source is None:
            # conv 상태처럼 사슬이 버퍼 placeholder 에서 끝나면(cat(placeholder, x) -> slice),
            # 사슬 안에서 가장 깊은 소유 노드(cat)의 파티션이 생산 샤드다. prefill 은 cat 이
            # (1,8192,35) 라 기존 conv 지역화의 (1,8192,4) 조건에 안 걸려 여기서 처리한다.
            for item in reversed(chain):
                if owners.get(item) is not None:
                    source = owners[item]
                    break
        if debug:
            logging.warning(
                "[MUTBUF_LOCALIZE] %s producer=%s(part %s) chain=%s",
                name,
                getattr(producer, "name", None),
                ids.get(id(source)),
                [(c.name, _schema_name(c) or "getitem", ids.get(id(owners.get(c)))) for c in chain],
            )
        if source is None:
            continue
        for item in reversed(chain):
            dest = owners.get(item)
            if dest is source:
                continue
            if dest is None:
                # QNN 이 못 받는 op 이 사슬에 있다. 여기서 멈춘다 (부분 이동은 무의미).
                if debug:
                    logging.warning("[MUTBUF_LOCALIZE] stop at %s: unsupported op", item.name)
                break
            side = [
                u for u in item.users
                if u.op != "output" and u not in chain and owners.get(u) is not source
            ]
            if side:
                if debug:
                    logging.warning("[MUTBUF_LOCALIZE] stop at %s: side users %s", item.name, [u.name for u in side])
                break
            dest.remove_node(item)
            source.add_node(item)
            owners[item] = source
            moved += 1
    if debug:
        logging.warning("[MUTBUF_LOCALIZE] moved %d nodes", moved)
    return [partition for partition in partitions if partition.nodes]


def _localize_delta_conv_outputs(partitions, graph):
    """Keep Qwen3.5 decode convolution-state slices in their producer shard."""
    mode = os.environ.get("QNN_DELTA_LOCAL_CONV_STATE", "off")
    if mode not in ("off", "decode"):
        raise ValueError("QNN_DELTA_LOCAL_CONV_STATE must be off or decode")
    if mode == "off":
        return partitions
    owners = {node: partition for partition in partitions for node in partition.nodes}
    for node in graph.nodes:
        if not node.users or any(user.op != "output" for user in node.users):
            continue
        # (1, 8192, 3) channel-major or (1, 3, 8192) time-major (QNN_CONV_TMAJOR=1)
        if tuple(getattr(node.meta.get("val"), "shape", ())) not in ((1, 8192, 3), (1, 3, 8192)):
            continue
        chain, value = [], node
        while getattr(getattr(value.target, "_schema", None), "name", None) == "aten::slice_copy":
            # A side consumer would need its own cross-shard dependency.
            if len(value.users) != 1:
                break
            chain.append(value)
            value = value.args[0]
        if not chain or getattr(getattr(value.target, "_schema", None), "name", None) != "aten::cat":
            continue
        if tuple(getattr(value.meta.get("val"), "shape", ())) not in ((1, 8192, 4), (1, 4, 8192)):
            continue
        source, destination = owners.get(value), owners.get(node)
        if source is None or destination is None or source is destination:
            continue
        if any(owners.get(item) is not destination for item in chain):
            continue
        for item in reversed(chain):
            destination.remove_node(item)
            source.add_node(item)
            owners[item] = source
    return [partition for partition in partitions if partition.nodes]


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        is_qnn_partitioner=True,
        phase="QnnPartitioner",
    ):
        option = generate_qnn_executorch_option(compiler_specs)
        python_options = flatbuffer_to_option(option)
        self.node_visitors = node_visitor_manager.get_node_visitors(
            edge_program,
            op_package_infos=python_options.op_package_options.op_package_infos,
            is_qnn_partitioner=is_qnn_partitioner,
        )

        self.skip_node_op_set = skip_node_op_set
        self.skip_node_id_set = skip_node_id_set
        self.is_qnn_partitioner = is_qnn_partitioner
        # Label prefixed to op-support log lines so callers that reuse this
        # checker (e.g. the LPAI fallback pass) are distinguishable in the logs.
        self.phase = phase
        self.nodes_to_wrappers = defaultdict(dict)
        self.qnn_manager = get_current_qnn_manager(
            python_options.backend_options.backend_type, compiler_specs
        )

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if node.target in to_be_implemented_operator:
            logger.info(
                f"[{self.phase}] {node.target.__name__} | Skipped, this op can be supported, please report an issue in https://github.com/pytorch/executorch/issues"
            )
            return False

        if node.meta.get(QCOM_FALLBACK_NODE, False):
            logger.info(f"[{self.phase}] {node.target.__name__} | Forced Fallback")
            return False

        if (
            node.target in allow_list_operator
            # bypass if custom op appears
            or OpContextLoader.namespace == node.target.namespace
            # bypass dequantize op for parameters & buffers
            or node.meta.get(QCOM_BYPASS_NODE, False)
        ):
            logger.info(f"[{self.phase}] {node.target.__name__} | Forced Passed")
            return True

        if (
            node.name in self.skip_node_id_set
            or node.target.__name__ in self.skip_node_op_set
        ):
            logger.info(f"[{self.phase}] {node.target.__name__} | Skipped")
            return False

        supported = False
        if node.target.__name__ not in self.node_visitors:
            logger.info(
                f"[{self.phase}] {node.target.__name__} | No node visitor, unsupported"
            )
            return False
        op_wrapper = self.node_visitors[node.target.__name__].define_node(
            node, self.nodes_to_wrappers
        )
        if node.target in constant_operator:
            return True

        op_wrapper_list = []
        if isinstance(op_wrapper, List):
            op_wrapper_list.extend(op_wrapper)
        else:
            op_wrapper_list.append(op_wrapper)

        if op_wrapper is not None:
            supported = self.qnn_manager.IsNodeSupportedByBackend(
                [op_wrapper.GetOpWrapper() for op_wrapper in op_wrapper_list]
            )

        self.nodes_to_wrappers.clear()
        logger.info(f"[{self.phase}] {node.target.__name__} | {supported}")
        # 진단용: DeltaNet 상태 경계의 mul/cat 이 층에 따라 다르게 판정되는 원인 추적.
        # QNN 에 실제로 넘어가는 dtype 을 통과/거부 양쪽에서 찍어 비교한다.
        if os.environ.get("QNN_PART_DEBUG") and node.target.__name__ in (
            "aten.mul.Tensor",
            "aten.cat.default",
        ):
            try:
                from executorch.backends.qualcomm.builders.utils import deduce_dtype
                from executorch.backends.qualcomm.utils.constants import (
                    QCOM_QUANT_ATTRS,
                    QCOM_QUANTIZED_IO,
                    QCOM_REQUANTIZE,
                )

                val = node.meta.get("val")
                osz = tuple(val.size()) if hasattr(val, "size") else None
                if osz in ((1, 32, 128, 128), (1, 8192, 4)):
                    def describe(n):
                        if not isinstance(n, torch.fx.Node):
                            return f"const:{type(n).__name__}"
                        qa = n.meta.get(QCOM_QUANT_ATTRS)
                        rq = n.meta.get(QCOM_REQUANTIZE)
                        io = n.meta.get(QCOM_QUANTIZED_IO)
                        dt = None
                        if qa is not None:
                            try:
                                dt = deduce_dtype(n.meta["val"], qa)
                            except Exception:
                                dt = "?"
                        rng = (
                            (qa.get("quant_min"), qa.get("quant_max")) if qa else None
                        )
                        return (
                            f"{n.name}[{n.op}] range={rng} dtype={dt} io={io} "
                            f"requant={sorted(rq) if rq else None}"
                        )

                    flat = []
                    for a in node.args:
                        flat.extend(a if isinstance(a, (list, tuple)) else [a])
                    logger.info(
                        f"[PARTDBG] {node.name} out={osz} supported={supported} "
                        f"| OUT {describe(node)} | "
                        + " | ".join(f"IN {describe(a)}" for a in flat)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[PARTDBG] failed on {node.name}: {exc}")
        return supported


class QnnPartitioner(Partitioner):
    """
    QnnPartitioner identifies subgraphs that can be lowered to QNN backend, by tagging nodes for delegation,
    and manages special cases such as mutable buffers and consumed constants.
    """

    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
    ):
        """
        Args:
            compiler_specs (List[CompileSpec]): Backend compiler specifications.
            skip_node_id_set (set, optional): Set of node IDs to exclude from partitioning.
            skip_node_op_set (set, optional): Set of OpOverload to exclude from partitioning.
            skip_mutable_buffer (bool, optional): If True, mutable buffers are not delegated to QNN.
        """
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)
        self.backend = flatbuffer_to_option(
            generate_qnn_executorch_option(self.compiler_specs_snapshot)
        ).backend_options.backend_type

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = set() if skip_node_id_set is None else skip_node_id_set
        self.skip_node_op_set = set() if skip_node_op_set is None else skip_node_op_set
        self.skip_mutable_buffer = skip_mutable_buffer

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            edge_program,
            self.compiler_specs_snapshot,
            self.skip_node_id_set,
            self.skip_node_op_set,
        )
        partitions = generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )
        graph = edge_program.graph_module.graph
        partitions = _localize_delta_state_outputs(partitions, graph)
        partitions = _localize_delta_conv_outputs(partitions, graph)
        return _localize_mutation_outputs(partitions, graph, edge_program)

    def tag_nodes(
        self, partitions: List[Partition], edge_program: torch.export.ExportedProgram
    ) -> None:
        """
        Tags nodes in the given partitions and the edge program's graph with delegation tags for QNN partitioning.
        """
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"qnn_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

        # need to take care of consumed constants
        consumed_constants = (
            *edge_program.graph_signature.inputs_to_buffers,
            *edge_program.graph_signature.inputs_to_parameters,
        )
        for node in edge_program.graph_module.graph.nodes:
            # find placeholders as lifted_constants
            if node.op != "placeholder" or len(node.users) != 0:
                continue

            if node.name in consumed_constants:
                # does no harm to merge them into last partition,
                # since they will all be removed in following stage
                node.meta["delegation_tag"] = delegation_tag

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        # Generate partitions by QNN op_support checker
        partitions = self.generate_partitions(edge_program)
        del self.op_support_checker
        # If partitions are found, handle tagging of nodes, constant data, and mutated buffers for delegation
        self.tag_nodes(partitions, edge_program)
        if os.environ.get("QNN_MUTBUF_DEBUG", "0") == "1":
            for i, part in enumerate(partitions):
                nodes = list(part.nodes)
                tags = {n.meta.get("delegation_tag") for n in nodes}
                sample = [n.name for n in nodes if "delta_rule" in n.name or "conv_state" in n.name][:3]
                logging.warning("[MUTBUF_PART] part %d tag=%s nodes=%d sample=%s", i, tags, len(nodes), sample)
        tag_constant_data(edge_program)
        if not self.skip_mutable_buffer:
            logger.info(
                "Qnn partitioner will delegate torch mutable buffer with the same I/O address during the runtime, "
                "so if your model contains mutable buffer, "
                "then you can get the better performance with skip_mutable_buffer=False. "
                "If you encounter accuracy issue during the runtime, "
                "then please set `skip_mutable_buffer=True` and try again."
            )
            tag_mutated_buffer(edge_program)

        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

    # override
    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Determines which op should not be decomposed during partitioning.
        The list of operators is obtained from `get_skip_decomp_table()`.
        The filter function (`filter_fn`) can be used to further refine which nodes are not decomposed. (advanced use case)
        """
        do_not_decompose = get_skip_decomp_table()
        return (do_not_decompose, filter_fn)
