"""Expose individual history groups after calibration, without a history concat."""
import operator
import json
import logging
import torch
from torch.utils import _pytree


def split_history_outputs(gm, layer_types):
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    snapshot = torch.ops.qwen35.delta_rule_snapshots16.default
    output = next(n for n in gm.graph.nodes if n.op == 'output')
    values = list(output.args[0])
    n = len(layer_types)
    if len(values) != 1 + 2*n:
        raise ValueError(f'Unexpected snapshot output contract: {len(values)}')

    def unwrap(node):
        while node.target in (q, dq):
            node = node.args[0]
        return node

    # The old output Q/DQ supplies the calibrated graph-IO encoding. The
    # generic IO override pairs users by order and can leave a stale Q/DQ on
    # one input branch. Repair only the state transport before dropping cat.
    # Do not derive scales from newly observed group histories.
    shape_ops = {torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor,
                 torch.ops.aten.clone.default, torch.ops.aten.alias.default}
    repairs = []
    transport = {}
    for layer, kind in enumerate(layer_types):
        if kind != 'linear_attention':
            continue
        value = values[1+n+layer]
        if value.target is not dq or value.args[0].target is not q:
            raise ValueError(f'Layer {layer} lacks calibrated state output Q/DQ')
        canonical = value.args[1:]
        if len(canonical) != 5 or canonical[2:4] != (0, 65535) or canonical[4] is not torch.int32:
            raise ValueError(f'Layer {layer} has unexpected uint16 state encoding: {canonical}')
        cat = unwrap(value)
        if cat.target is not torch.ops.aten.cat.default or cat.args[1] != 1:
            raise ValueError(f'Layer {layer} lacks grouped history cat')
        roots = set()
        for group, part in enumerate(cat.args[0]):
            raw = unwrap(part)
            if raw.target is not operator.getitem or raw.args[1] != 1 or raw.args[0].target is not snapshot:
                raise ValueError(f'Layer {layer} group {group} lacks a snapshot producer')
            kernel = raw.args[0]
            chain = []
            node = kernel.args[5]
            while node.op != 'placeholder':
                if node.target in (q, dq):
                    if node.args[3:6] != canonical[2:]:
                        raise ValueError(f'Layer {layer} has incompatible state transport dtype/range')
                    chain.append(node)
                elif node.target not in shape_ops:
                    raise ValueError(f'Layer {layer} has non-transport state ancestor {node}')
                node = node.args[0]
            roots.add(node)
            node = part
            while node is not raw:
                if node.target not in (q, dq) or node.args[3:6] != canonical[2:]:
                    raise ValueError(f'Layer {layer} has incompatible history output chain')
                chain.append(node); node = node.args[0]
            for node in chain:
                previous = transport.get(node)
                if previous is not None and previous != canonical:
                    raise ValueError(f'State transport {node} is shared across incompatible layers')
                transport[node] = canonical
                if node.args[1:] != canonical:
                    repairs.append({'layer': layer, 'group': group, 'node': node.name,
                                    'before': [str(x) for x in node.args[1:]],
                                    'after': [str(x) for x in canonical]})
        if len(roots) != 1:
            raise ValueError(f'Layer {layer} groups do not read one state input')
    # Apply only after every path has been verified. No partial graph rewrite
    # escapes if an unsupported path or incompatible encoding is encountered.
    for node, encoding in transport.items():
        node.args = (node.args[0], *encoding)
    if repairs:
        logging.warning('[SnapshotStateEncodingRepair] %s', json.dumps(repairs))

    flat, roles, groups = [], [], {}
    for i, value in enumerate(values):
        if i == 0:
            flat.append(value); roles.append('logits'); continue
        layer = (i-1) % n
        linear = layer_types[layer] == 'linear_attention'
        if i <= n:
            flat.append(value)
            roles.append(f'{"state_conv" if linear else "kv_k"}:{layer}')
        elif not linear:
            flat.append(value); roles.append(f'kv_v:{layer}')
        else:
            cat = unwrap(value)
            if cat.target is not torch.ops.aten.cat.default or cat.args[1] != 1:
                raise ValueError(f'Expected history head concat for layer {layer}: {cat}')
            parts = list(cat.args[0])
            for group, part in enumerate(parts):
                raw = unwrap(part)
                if raw.target is not operator.getitem or raw.args[1] != 1 or raw.args[0].target is not snapshot:
                    raise ValueError(f'History group {layer}:{group} not from snapshot kernel')
                if value.target is not dq or part.target is not dq or value.args[1:] != part.args[1:]:
                    raise ValueError(f'History group {layer}:{group} would change IO quantization')
                flat.append(part); roles.append(f'state_rec_group:{layer}:{group}')
            groups[str(layer)] = len(parts)
    output.args = (tuple(flat),)
    # Converted PT2E modules reconstruct the old nested output with this tree spec.
    # The new export deliberately uses a flat tuple and an explicit role manifest.
    spec = _pytree.tree_flatten(tuple(flat))[1]
    codegen = gm.graph._codegen
    if hasattr(codegen, "pytree_info"):
        codegen.pytree_info = codegen.pytree_info._replace(out_spec=spec)
    gm._out_spec = spec
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return {'version': 2, 'roles': roles, 'groups': groups,
            'history_concat_removed': True, 'quantization_unchanged_by_split': True,
            'state_transport_repairs': repairs, 'canonical_encoding_source': 'calibrated graph state output',
            'non_state_quantization_unchanged': True}
