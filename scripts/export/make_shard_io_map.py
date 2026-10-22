#!/usr/bin/env python3
"""컨텍스트 디렉터리에 shard_io_map.json 을 만든다 (Qwen3.5 하이브리드 + 지연 커밋 타깃).

런타임(MobiSpec.cpp)이 샤드별 QNN 텐서를 **이름으로** 역할에 연결할 수 있게 한다. 이름만으로는
안 되는 이유가 두 가지다.
  1. 재료(ingredient) 출력은 모양이 겹친다(beta/decay 가 같은 모양). 순서 규칙에만 기대면 위험하다.
  2. 출력 전용 후처리(k 의 stride-2 slice, gate 의 unsqueeze)는 파티셔너가 **마지막 샤드**로 옮긴다.
     그래서 층 L 의 재료 k 는 L 이 속한 샤드가 아니라 마지막 샤드의 출력으로 나온다(2026-09-16 실측).

매핑 근거: QNN 텐서 이름의 `input_<k>_` 와 `output_order_<n>_` 은 그 샤드 델리게이트의 인자 순서다.
델리게이트 인자는 [입력 n_in 개][출력 n_out 개] 이고, .pte 의 program 입출력 값 인덱스와 맞춰
역할을 확정한다. 역할 이름:
  tokens, pos, mask, hidden_in, hidden_out, logits, valid_prev, valid_cur, conv_idx_prev,
  state_conv:L, state_rec:L, kv_k:L, kv_v:L, ing_x:L, ing_k:L, ing_v:L, ing_beta:L, ing_decay:L
"""
import argparse, json, re, sys
from pathlib import Path

# executorch = the patched checkout on PYTHONPATH (scripts/env.sh)
from executorch.exir._serialize._program import deserialize_pte_binary  # noqa: E402
from executorch.exir.schema import DelegateCall, Tensor  # noqa: E402

FIELDS = ["x", "k", "v", "beta", "decay"]


def roles_for_program(params, method="prefill_forward", ckpt_groups=0):
    """program 입력/출력 인덱스 -> 역할 문자열. export 규약(static_llama.forward)을 따른다."""
    n_layers = params["n_layers"]
    layer_types = params["layer_types"]
    delta = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    defer = bool(params.get("defer_state", 0))

    inp = {0: "tokens", 1: "mask", 2: "pos"}
    for i in range(n_layers):
        inp[3 + i] = ("state_conv:%d" if layer_types[i] == "linear_attention" else "kv_k:%d") % i
        inp[3 + n_layers + i] = ("state_rec:%d" if layer_types[i] == "linear_attention" else "kv_v:%d") % i
    if defer:
        base = 3 + 2 * n_layers
        inp[base] = "valid_prev"
        for j, L in enumerate(delta):
            for f, name in enumerate(FIELDS):
                inp[base + 1 + j * len(FIELDS) + f] = f"ing_{name}:{L}"
        if params.get("defer_masked", 0):
            # 2026-10-01 masked-op deferral: int32 inputs after the ingredients (static_llama.forward)
            inp[base + 1 + len(FIELDS) * len(delta)] = "valid_cur"
            inp[base + 2 + len(FIELDS) * len(delta)] = "conv_idx_prev"

    out = {0: "logits"}
    for i in range(n_layers):
        out[1 + i] = ("state_conv:%d" if layer_types[i] == "linear_attention" else "kv_k:%d") % i
        out[1 + n_layers + i] = ("state_rec:%d" if layer_types[i] == "linear_attention" else "kv_v:%d") % i
    if defer and not params.get("state_snapshots", False):
        base = 1 + 2 * n_layers
        for j, L in enumerate(delta):
            for f, name in enumerate(FIELDS):
                out[base + j * len(FIELDS) + f] = f"ing_{name}:{L}"
    # 2026-10-12 23:5x KST (ckpt10): --ckpt-groups G -> state_ckpt_group:L:g roles after the ingredient outputs
    if defer and ckpt_groups:
        ck = 1 + 2 * n_layers + len(FIELDS) * len(delta)
        for j, L in enumerate(delta):
            for g in range(ckpt_groups):
                out[ck + j * ckpt_groups + g] = f"state_ckpt_group:{L}:{g}"
    if params.get("snapshot_output_roles"):
        out = dict(enumerate(params["snapshot_output_roles"]))
    return inp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--ckpt-groups", type=int, default=0)
    ap.add_argument("--ctx", required=True, help="context dir holding forward_<i>_json.json and params.json")
    ap.add_argument("--method", default="prefill_forward")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = Path(a.ctx)
    params = json.load(open(ctx / "params.json"))
    prog = deserialize_pte_binary(open(a.pte, "rb").read()).program
    plan = [p for p in prog.execution_plan if p.name == a.method][0]
    in_roles, out_roles = roles_for_program(params, a.method, a.ckpt_groups)
    value_in = {v: k for k, v in enumerate(plan.inputs)}
    value_out = {}
    for k, v in enumerate(plan.outputs):
        value_out.setdefault(v, []).append(k)

    def sizes(idx):
        v = plan.values[idx].val
        return list(v.sizes) if isinstance(v, Tensor) else None

    dcalls = []
    for chain in plan.chains:
        for instr in chain.instructions:
            if isinstance(instr.instr_args, DelegateCall):
                dcalls.append(list(instr.instr_args.args))
    shards = sorted(ctx.glob("forward_*_json.json"), key=lambda p: int(p.name.split("_")[1]))
    assert len(shards) == len(dcalls), (len(shards), len(dcalls))

    # 샤드 사이를 잇는 hidden: 한 델리게이트의 출력이 다음 델리게이트의 입력인 값
    produced = {}
    for d, args in enumerate(dcalls):
        for pos, val in enumerate(args):
            produced.setdefault(val, []).append((d, pos))

    out_map = {"snapshot_format": params.get("snapshot_format", 1), "snapshot_groups": params.get("snapshot_groups", {}), "method": a.method, "state_snapshots": bool(params.get("state_snapshots", False)), "verify_ar_len": params.get("verify_ar_len", 0), "shards": []}
    for d, (jf, args) in enumerate(zip(shards, dcalls)):
        j = json.load(open(jf))
        g = [x for x in j["info"]["graphs"] if x["info"]["graphName"] == a.method][0]["info"]
        qin = sorted(g["graphInputs"], key=lambda t: int(re.search(r"input_(\d+)_", t["info"]["name"]).group(1)))
        qout = sorted(g["graphOutputs"], key=lambda t: int(re.search(r"output_order_(\d+)_", t["info"]["name"]).group(1)))
        n_in, n_out = len(qin), len(qout)
        assert len(args) == n_in + n_out, (jf.name, len(args), n_in, n_out)
        entry = {"shard": d, "inputs": {}, "outputs": {}}
        hidden_dims = [1, params.get("verify_ar_len", 0), params["dim"]]
        for k, t in enumerate(qin):
            val = args[k]
            name = t["info"]["name"]
            dims = t["info"]["dimensions"]
            if val in value_in:
                role = in_roles.get(value_in[val], "unknown_in_%d" % value_in[val])
            elif list(dims) == hidden_dims:
                role = "hidden_in"          # residual stream from the previous shard
            else:
                # Any other tensor produced by one delegate and consumed by another. The export has
                # these because output-only tail ops (the ingredient key's stride-2 slice, the
                # gates' unsqueeze) are assigned to the last shard, so the un-sliced key travels
                # there. The runtime has to copy them across; calling them hidden would memcpy a
                # 256 KiB tensor into the 80 KiB hidden buffer.
                role = "link:%d" % val
            entry["inputs"][name] = {"role": role, "dims": dims, "dtype": t["info"]["dataType"]}
        for k, t in enumerate(qout):
            val = args[n_in + k]
            name = t["info"]["name"]
            dims = t["info"]["dimensions"]
            roles = [out_roles.get(p, "unknown_out_%d" % p) for p in value_out.get(val, [])]
            if roles:
                role = roles[0]
            elif list(dims) == hidden_dims:
                role = "hidden_out"
            else:
                role = "link:%d" % val
            entry["outputs"][name] = {"role": role, "dims": dims, "dtype": t["info"]["dataType"]}
        out_map["shards"].append(entry)

    # 검증: 모든 역할이 정확히 한 번씩 나오는지
    counts = {}
    for s in out_map["shards"]:
        for side in ("inputs", "outputs"):
            for v in s[side].values():
                counts[(side, v["role"])] = counts.get((side, v["role"]), 0) + 1
    # every link must have exactly one producer and at least one consumer
    links_out = {r for (side, r) in counts if side == "outputs" and r.startswith("link:")}
    links_in  = {r for (side, r) in counts if side == "inputs" and r.startswith("link:")}
    if links_out != links_in:
        print("WARNING: unmatched inter-shard links:",
              sorted(links_out ^ links_in)[:8], file=sys.stderr)
    missing = [r for r in out_roles.values() if ("outputs", r) not in counts]
    if missing:
        print("WARNING: outputs missing roles:", missing[:8], file=sys.stderr)
    dst = Path(a.out) if a.out else ctx / "shard_io_map.json"
    json.dump(out_map, open(dst, "w"), indent=1)
    n_ing = sum(1 for (side, r), c in counts.items() if side == "outputs" and r.startswith("ing_"))
    print(f"wrote {dst}: {len(out_map['shards'])} shards, "
          f"{sum(len(s['outputs']) for s in out_map['shards'])} outputs, ingredient roles {n_ing}")
    for s in out_map["shards"][:1] + out_map["shards"][-1:]:
        print(f"  shard {s['shard']}: in roles {sorted({v['role'].split(':')[0] for v in s['inputs'].values()})}, "
              f"out roles {sorted({v['role'].split(':')[0] for v in s['outputs'].values()})}")


if __name__ == "__main__":
    main()
