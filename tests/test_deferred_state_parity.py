"""지연 커밋(QNN_DEFER_STATE=1, ar_len=T) 그래프 vs 즉시 커밋(ar_len=1) 스텝의 동등성.

추측 디코딩 검증을 흉내 낸다: 매 스텝 T 토큰 청크를 넣되 앞 n_i 개만 수락된 것으로 친다(나머지는
무작위 '드래프트 쓰레기'). 지연 그래프는 다음 호출에서 valid_prev 로 직전 청크의 앞 n_{i-1} 개만
커밋한다. 수락 위치의 로짓은 즉시 커밋 모델이 수락 토큰만 하나씩 처리한 로짓과 같아야 하고,
마지막 flush 뒤의 상태도 같아야 한다.

환경변수로 경로를 고른다: QNN_DELTA_KERNEL=0(펼친 재귀) / 1(커스텀 op CPU 참조 구현).
"""
import os, sys, json, random
os.environ["QNN_DEFER_STATE"] = "1"
import torch
sys.path.insert(0, os.path.dirname(os.environ["ET_DRAFT"]))  # parent of the patched checkout (scripts/env.sh)
torch.manual_seed(0); random.seed(0)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import LlamaModel
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask, CausalAttentionMask

T, CL, V = 8, 64, 100
layer_types = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
def make_cfg(ar_len):
    c = ModelArgs(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16, hidden_dim=128,
                  linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                  linear_value_head_dim=16, linear_conv_kernel_dim=4, norm_eps=1e-6, max_batch_size=1,
                  max_seq_len=CL, max_context_len=CL, use_kv_cache=True, vocab_size=V, use_hf_rope=True,
                  use_qk_norm=True, qk_norm_before_rope=True, layer_types=layer_types)
    c.ar_len = ar_len; c.enable_masked_softmax = True
    return c
A = LlamaModel(make_cfg(1), ar_len=1, output_new_cache_only=True, output_cache=True).eval()
B = LlamaModel(make_cfg(T), ar_len=T, output_new_cache_only=True, output_cache=True).eval()
B.load_state_dict(A.state_dict(), strict=True)
assert not A.deferred_state and B.deferred_state, (A.deferred_state, B.deferred_state)
print("delta layers", B.delta_layer_ids, "conv select", B.defer_conv_select, "kernel", os.environ.get("QNN_DELTA_KERNEL"))

# accepted token stream and per-step acceptance counts
n_steps = 5
accepts = [3, 8, 1, 5, 4]
seq = [random.randrange(V) for _ in range(sum(accepts))]

# ---- A: immediate, token by token over accepted tokens ----
exA = A.get_example_inputs()
maskA = AttentionMask(CausalAttentionMask(1, 1, CL)); kA = [t.clone() for t in exA[3]]; vA = [t.clone() for t in exA[4]]
logitsA = []
with torch.no_grad():
    for p, tok in enumerate(seq):
        out = A(torch.tensor([[tok]], dtype=torch.int32), list(maskA)[0], torch.tensor([[p]], dtype=torch.int32), *kA, *vA)
        logitsA.append(out[0][0, 0].clone())
        for i, lt in enumerate(layer_types):
            if lt == "linear_attention": kA[i], vA[i] = out[1][i], out[2][i]
            else: kA[i][:, :, :, p:p+1] = out[1][i]; vA[i][:, :, p:p+1, :] = out[2][i]
        maskA.smart_mask_update(p, 1)

# ---- B: deferred chunks with partial acceptance ----
exB = B.get_example_inputs()
assert len(exB) == 7
maskB = AttentionMask(CausalAttentionMask(1, T, CL)); kB = [t.clone() for t in exB[3]]; vB = [t.clone() for t in exB[4]]
valid_prev = torch.zeros_like(exB[5]); ing = [t.clone() for t in exB[6]]
pos = 0; max_err = 0.0
with torch.no_grad():
    for step, n_acc in enumerate(accepts):
        chunk = seq[pos:pos+n_acc] + [random.randrange(V) for _ in range(T - n_acc)]  # tail = rejected drafts
        tokens = torch.tensor([chunk], dtype=torch.int32)
        pos_t = torch.arange(pos, pos+T, dtype=torch.int32).unsqueeze(0)
        out = B(tokens, list(maskB)[0], pos_t, *kB, *vB, valid_prev, *ing)
        logits, newK, newV, new_ing = out
        for j in range(n_acc):
            err = (logits[0, j] - logitsA[pos + j]).abs().max().item(); max_err = max(max_err, err)
        for i, lt in enumerate(layer_types):
            if lt == "linear_attention": kB[i], vB[i] = newK[i], newV[i]   # committed (previous chunk) state
            else: kB[i][:, :, :, pos:pos+n_acc] = newK[i][:, :, :, :n_acc]; vB[i][:, :, pos:pos+n_acc, :] = newV[i][:, :, :n_acc, :]
        ing = [t.clone() for t in new_ing]
        valid_prev = torch.zeros_like(valid_prev); valid_prev[0, :n_acc] = 1.0
        maskB.smart_mask_update(pos, n_acc); pos += n_acc
        print(f"step {step}: accepted {n_acc}, max|dlogit| so far {max_err:.2e}")
    # flush: commit the last chunk, compare states with A
    out = B(torch.zeros(1, T, dtype=torch.int32), list(maskB)[0], torch.arange(pos, pos+T, dtype=torch.int32).unsqueeze(0), *kB, *vB, valid_prev, *ing)
    errS = max((out[2][i] - vA[i]).abs().max().item() for i in B.delta_layer_ids)
    errC = max((out[1][i] - kA[i]).abs().max().item() for i in B.delta_layer_ids)
print(f"logits max err {max_err:.2e}  state err {errS:.2e}  conv err {errC:.2e}")
ok = max_err < 1e-4 and errS < 1e-4 and errC < 1e-4
print("PASS" if ok else "FAIL")
