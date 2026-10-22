#!/usr/bin/env python3
"""드래프트 블록 임베딩용 fp16 테이블 (QNN_Exporter/scripts/dump_draft_embd.py 와 같은 규약).

Qwen3.5-4B 는 임베딩과 lm_head 가 tied 이므로 타깃 메타 체크포인트의 tok_embeddings.weight 를 쓴다.
행 우선 [vocab, dim] little-endian fp16, 헤더 없음 (248320 x 2560 x 2B = 1.27 GB). 런타임이 mmap.
  dump_draft_embd.py --out draft_q35_embd_f16.bin
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="target meta checkpoint (qwen3_5_4b_meta.pth)")
ap.add_argument("--key", default="tok_embeddings.weight")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sd = torch.load(a.ckpt, weights_only=True, map_location="cpu", mmap=True)
w = sd[a.key]
w.to(torch.float16).contiguous().numpy().tofile(a.out)
print(f"{a.out}: {tuple(w.shape)} fp16, {w.numel()*2/2**20:.0f} MiB")
