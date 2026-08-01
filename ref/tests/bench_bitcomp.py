#!/usr/bin/env python3
"""2/3/4-bit 비교 벤치마크 — MSE, Recall, 저장공간, 속도"""

import math, os, shutil, time, sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.turboquant.eden import EdenConfig, EdenQuantizer, compute_beta_codebook, pack_bits, unpack_bits, pack_signs, unpack_signs
from src.turboquant.memory import TurboDiskStore, TurboQuantizer_V2

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cpu"

def unit_vectors(n, dim):
    x = torch.randn(n, dim)
    return x / torch.norm(x, dim=-1, keepdim=True)

def mse(x, x_hat):
    return torch.mean((x - x_hat) ** 2).item()

def recall_at_k(store, query, vectors, k):
    """TurboDiskStore 검색 recall vs brute-force inner product."""
    results = store.search(query, top_k=k)
    result_ids = set(r[0] for r in results)
    bf_scores = torch.matmul(vectors, query)
    bf_top = set(torch.topk(bf_scores, k).indices.tolist())
    overlap = result_ids & bf_top
    return len(overlap) / k

# ═══════════════════════════════════════════════════════════
# 1. MSE 비교 (d=64, 128, 256, 512, 1024)
# ═══════════════════════════════════════════════════════════
print("=" * 72)
print("1. MSE 비교 (합성 단위 벡터, N=500)")
print("=" * 72)
print(f"{'bits':>4} {'dim':>5} {'V2 MSE':>12} {'EDEN-biased':>14} {'EDEN-unbiased':>16} {'V3-vs-V2':>10} {'biased-vs-unbiased':>18}")
print("-" * 72)

mse_dims = [64, 128, 256, 512, 1024]
# 조금만: 64, 1024만 full run, 나머지는 빠르게
for dim in mse_dims:
    for bits in [2, 3, 4]:
        eq_b = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="biased", residual_bits=1, seed=42))
        eq_u = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="unbiased", residual_bits=1, seed=42))
        v2 = TurboQuantizer_V2(dim, bits)
        
        x = unit_vectors(500, dim)
        
        # EDEN biased
        idx_b, rs_b, rsc_b, S_b = eq_b.quantize(x)
        x_b = eq_b.decode(idx_b, rs_b, rsc_b, S_b)
        m_b = mse(x, x_b)
        
        # EDEN unbiased
        idx_u, rs_u, rsc_u, S_u = eq_u.quantize(x)
        x_u = eq_u.decode(idx_u, rs_u, rsc_u, S_u)
        m_u = mse(x, x_u)
        
        # V2
        x_v2_list = []
        for i in range(x.shape[0]):
            idxx, sgn, scl = v2.quantize(x[i])
            y_hat = v2.levels[idxx]
            residual = sgn.float() * scl
            y_hat = y_hat + residual
            x_v2_list.append(y_hat @ v2.rotation.T)
        x_v2 = torch.stack(x_v2_list)
        m_v2 = mse(x, x_v2)
        
        impr = (1 - m_b / m_v2) * 100
        ratio_bu = m_b / m_u if m_u > 0 else 0
        print(f"{bits:>4d} {dim:>5d} {m_v2:>12.3e} {m_b:>14.3e} {m_u:>16.3e} "
              f"{impr:>+9.2f}% {ratio_bu:>8.4f}")
    print()

# ═══════════════════════════════════════════════════════════
# 2. Storage 비교
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("2. Storage 비교 (d=1024, N=1M 기준)")
print("=" * 60)
print(f"{'bits':>4} {'idx_bytes':>10} {'sign_bytes':>11} {'scale_bytes':>12} {'S_bytes':>8} {'total':>8} {'uint8_equiv':>12} {'savings':>9}")
print("-" * 60)

d = 1024
n = 1_000_000
for bits in [2, 3, 4]:
    idx_bytes = n * d * bits // 8
    sign_bytes = n * d // 8
    scale_bytes = n * 4  # float32
    S_bytes = n * 4       # float32
    total = idx_bytes + sign_bytes + scale_bytes + S_bytes
    uint8_equiv = n * d * (1 + 1)  # V2: index uint8 + sign uint8
    savings = (1 - total / uint8_equiv) * 100
    print(f"{bits:>4d} {idx_bytes:>10,d} {sign_bytes:>11,d} {scale_bytes:>12,d} "
          f"{S_bytes:>8,d} {total:>8,d} {uint8_equiv:>12,d} {savings:>+8.1f}%")
print()

# ═══════════════════════════════════════════════════════════
# 3. Recall 비교 (d=128, N=1000 — 빠른 합성 데이터)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("3. Recall 비교 (d=128, N=1000, 합성 단위벡터)")
print("=" * 60)
print(f"{'bits':>4} {'k':>4} {'V2 recall':>10} {'EDEN recall':>12} {'Δ':>6}")
print("-" * 60)

dim_r, n_r = 128, 1000
vectors = unit_vectors(n_r, dim_r)

for bits in [2, 3, 4]:
    eq = EdenQuantizer(EdenConfig(dim=dim_r, bits=bits, mode="biased", residual_bits=1, seed=42))
    store = TurboDiskStore(dim_r, bits, f"/tmp/bench_recall_b{bits}", quantizer=eq)
    
    for v in vectors:
        store.add(v)
    
    # V2 store for baseline
    store_v2 = TurboDiskStore(dim_r, bits, f"/tmp/bench_recall_v2_b{bits}")
    for v in vectors:
        store_v2.add(v)
    
    queries = unit_vectors(50, dim_r)
    for k in [1, 5, 10, 64]:
        r_eden = np.mean([recall_at_k(store, q, vectors, k) for q in queries])
        r_v2 = np.mean([recall_at_k(store_v2, q, vectors, k) for q in queries])
        delta = r_eden - r_v2
        print(f"{bits:>4d} {k:>4d} {r_v2:>10.4f} {r_eden:>12.4f} {delta:>+6.2%}")
    
    shutil.rmtree(f"/tmp/bench_recall_b{bits}", ignore_errors=True)
    shutil.rmtree(f"/tmp/bench_recall_v2_b{bits}", ignore_errors=True)
    print()

# ═══════════════════════════════════════════════════════════
# 4. 속도 비교 (양자화 + 디코딩 + pack/unpack)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("4. 속도 비교 (d=1024, N=1000)")
print("=" * 60)
print(f"{'bits':>4} {'quantize':>10} {'decode':>10} {'pack_bits':>10} {'unpack_bits':>10}")
print("-" * 60)

d_speed = 1024
n_speed = 1000
x_speed = unit_vectors(n_speed, d_speed)

for bits in [2, 3, 4]:
    eq = EdenQuantizer(EdenConfig(dim=d_speed, bits=bits, mode="biased", residual_bits=1, seed=42))
    
    # warmup
    eq.quantize(x_speed[:10])
    
    t0 = time.perf_counter()
    idx, rs, rsc, S = eq.quantize(x_speed)
    t_q = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    x_hat = eq.decode(idx, rs, rsc, S)
    t_d = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    packed = pack_bits(idx, bits)
    t_p = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    unpacked = unpack_bits(packed, bits, d_speed)
    t_up = time.perf_counter() - t0
    
    print(f"{bits:>4d} {t_q:>10.4f}s {t_d:>10.4f}s {t_p:>10.4f}s {t_up:>10.4f}s")
print()

# ═══════════════════════════════════════════════════════════
# 5. Packing 효율
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("5. Packing 효율 (d=1024, N=1)")
print("=" * 60)
print(f"{'bits':>4} {'unpacked':>10} {'packed':>10} {'ratio':>8}")
print("-" * 60)

for bits in [2, 3, 4]:
    x1 = unit_vectors(1, d_speed)
    eq = EdenQuantizer(EdenConfig(dim=d_speed, bits=bits, mode="biased", seed=42))
    idx, rs, rsc, S = eq.quantize(x1)
    idx_bytes = idx.numel() * idx.element_size()  # original tensor
    packed = pack_bits(idx, bits)
    packed_bytes = packed.numel() * packed.element_size()
    ratio = packed_bytes / idx_bytes
    print(f"{bits:>4d} {idx_bytes:>10d} {packed_bytes:>10d} {ratio:>7.3f}")

print()
print("=" * 60)
print("✓ 2/3/4-bit comparison complete")
print("=" * 60)
