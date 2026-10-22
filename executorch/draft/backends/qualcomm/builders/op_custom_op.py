# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict, Iterable

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import operator
import os

import torch

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)

from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_UNQUANTIZED_OUTPUTS,
)

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP


def _resolve_qnn_data_type(
    py_type: type, arg_name: str, target: str, *, of_element: bool = False
):
    """QNN data type for a custom-op arg type, or a ValueError naming the arg."""
    what = "element type" if of_element else "type"
    if py_type is str:
        raise ValueError(
            f"String {what} for argument '{arg_name}' of {target} is unsupported: "
            "QNN_DATATYPE_STRING is not plumbed through AddScalarParam in "
            "aot/python/PyQnnManagerAdaptor.h. Encode it as an int in the op "
            "schema, or add the missing case."
        )
    if py_type not in QNN_TENSOR_TYPE_MAP:
        raise ValueError(
            f"Argument '{arg_name}' of {target} has unsupported {what} "
            f"{py_type.__name__}: QNN_TENSOR_TYPE_MAP has no entry for it."
        )
    return QNN_TENSOR_TYPE_MAP[py_type]


class CustomOp(NodeVisitor):
    target = ""
    op_package_info = QnnExecuTorchOpPackageInfo()

    def __init__(self, op_package_info: QnnExecuTorchOpPackageInfo, *args) -> None:
        super().__init__(*args)
        self.target = op_package_info.custom_op_name
        self.op_package_info = op_package_info

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        custom_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            self.op_package_info.op_package_name,
            self.op_package_info.qnn_op_type_name,
        )

        custom_input_tensors = []
        custom_attr_keys = [arg.name for arg in node.target._schema.arguments]
        if len(custom_attr_keys) != len(node.args):
            warnings.warn(
                f"Number of inputs ({len(node.args)}) mismatch the number of args ({len(custom_attr_keys)}) in schema for the custom node ({self.target}).",
                stacklevel=1,
            )
            return
        for arg, arg_name in zip(node.args, custom_attr_keys):
            if arg is None:
                continue
            if isinstance(arg, torch.fx.Node):
                input_tensor = self.get_tensor(arg, node)
                input_tensor_wrapper = self.define_tensor(
                    arg,
                    node,
                    input_tensor,
                    PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    nodes_to_wrappers,
                )
                custom_input_tensors.append(input_tensor_wrapper)
            elif isinstance(arg, str):
                # str is Iterable, so it would otherwise be taken for a tensor param.
                _resolve_qnn_data_type(str, arg_name, self.target)
            elif isinstance(arg, Iterable):
                values = list(arg)
                if not values:
                    raise ValueError(
                        f"Argument '{arg_name}' of {self.target} is empty: a QNN "
                        "tensor param needs at least one element."
                    )
                tensor_parm_shape = [len(values)]
                custom_op.AddTensorParam(
                    arg_name,
                    _resolve_qnn_data_type(
                        type(values[0]), arg_name, self.target, of_element=True
                    ),
                    len(tensor_parm_shape),
                    tensor_parm_shape,
                    np.array(values),
                    True,
                )
            else:
                custom_op.AddScalarParam(
                    arg_name,
                    _resolve_qnn_data_type(type(arg), arg_name, self.target),
                    {QCOM_DATA: arg},
                )

        if isinstance(node.meta["val"], tuple):
            # 출력마다 양자화가 다를 수 있다. 출력별 스펙은 getitem 노드에 붙으므로
            # 인코딩은 거기서 읽는다. node 를 그대로 쓰면 모든 출력이 첫 출력의
            # 스케일을 물려받아, float 이어야 할 출력이 엉뚱한 스케일로 정수화된다
            # (2026-09-10 실측: DeltaRule 의 state_out 이 out 의 스케일을 받아 되먹임이 파괴됨).
            getitem_by_idx = {
                u.args[1]: u
                for u in node.users
                if u.op == "call_function"
                and u.target == operator.getitem
                and len(u.args) > 1
                and isinstance(u.args[1], int)
            }
            custom_output_tensors = []
            # Export can propagate custom metadata from an argument; a marker
            # for another op must not change this op's output encoding.
            float_outputs = (
                node.meta.get("custom", {})
                .get(QCOM_UNQUANTIZED_OUTPUTS, {})
                .get(node.target._schema.name, ())
            )
            for i in range(len(node.meta["val"])):
                output_tensor = self.get_tensor(node, node, i)
                output_tensor_wrapper = self.define_tensor(
                    node,
                    node,
                    output_tensor,
                    PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    nodes_to_wrappers,
                    wrapper_idx=i,
                    quant_source_node=getitem_by_idx.get(i),
                    skip_quantization=i not in getitem_by_idx and i in float_outputs,
                )
                custom_output_tensors.append(output_tensor_wrapper)
        else:
            output_tensor = self.get_tensor(node, node)
            output_tensor_wrapper = self.define_tensor(
                node,
                node,
                output_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            custom_output_tensors = [output_tensor_wrapper]

        custom_op.AddInputTensors(custom_input_tensors)
        custom_op.AddOutputTensors(custom_output_tensors)
        if os.environ.get("QNN_CUSTOM_IO_DEBUG") == "1":
            for label, tensors in (("input", custom_input_tensors), ("output", custom_output_tensors)):
                for index, tensor in enumerate(tensors):
                    tensor = PyQnnManager.PyQnnTensorWrapper(tensor)
                    print(f"CUSTOM_IO {node.name} {label}[{index}] {tensor.GetDataType()} {tensor.GetDims()}", flush=True)
        return custom_op
