"""TurboQuant 알고리즘 검증 — algo_a (Lloyd-Max+QJL) vs algo_b (V2 levels+residual) vs FP32

측정 항목:
  1. Inner Product 추정 오차율 (알고리즘 정확도)
  2. 압축률 (저장 효율)
  3. 검색 Recall@k (FP32 대비排名 보존율)
  4. 검색 속도
"""

import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_store import MemoryStore
from turbo_quant import build_state, compress, prepare_query, estimate as estimate_a
from turbo_quant_v2 import build_state_v2, compress_v2, prepare_query_v2, estimate_v2

DIM = 384
NUM_TEST = 500
NUM_SEARCH = 2000
TOP_K = 10

np.random.seed(42)


def benchmark_compression_accuracy():
    """1. 압축 알고리즘 정확도: 실제 inner product vs 추정 inner product"""
    print("=" * 60)
    print("1. Inner Product 추정 정확도")
    print("=" * 60)

    state_a = build_state(dim=DIM, bits=3, seed=42)
    state_b = build_state_v2(dim=DIM, bits=3, seed=42)

    # Generate test data
    vectors = np.random.randn(NUM_TEST, DIM).astype(np.float32)
    queries = np.random.randn(NUM_TEST // 5, DIM).astype(np.float32)

    # Normalize (cosine similarity 기준)
    for v in vectors:
        n = np.linalg.norm(v)
        if n > 1e-12:
            v /= n
    for q in queries:
        n = np.linalg.norm(q)
        if n > 1e-12:
            q /= n

    # Pre-compress all vectors
    compressed_a = [compress(v, state_a) for v in vectors]
    compressed_b = [compress_v2(v, state_b) for v in vectors]

    actuals = []
    ests_a = []
    ests_b = []

    print(f"총 {len(queries)}개 쿼리 × {len(vectors)}개 벡터 = {len(queries) * len(vectors)}쌍 테스트")
    print()

    for q_idx, q in enumerate(queries):
        q_a, q_qjl = prepare_query(q, state_a)
        q_b = prepare_query_v2(q, state_b)

        for v_idx in range(len(vectors)):
            actual = float(np.dot(q, vectors[v_idx]))
            idx_a, norm_a, qjl_a, r_norm_a = compressed_a[v_idx]
            est_a = estimate_a(q_a, q_qjl, state_a, idx_a, norm_a, qjl_a, r_norm_a)
            indices_b, signs_b, scale_b = compressed_b[v_idx]
            est_b = estimate_v2(q_b, state_b, indices_b, signs_b, scale_b, query_norm=1.0)
            actuals.append(actual)
            ests_a.append(est_a)
            ests_b.append(est_b)

    actuals = np.array(actuals)
    ests_a = np.array(ests_a)
    ests_b = np.array(ests_b)

    abs_err_a = np.abs(actuals - ests_a)
    abs_err_b = np.abs(actuals - ests_b)
    rel_err_a = abs_err_a / (np.abs(actuals) + 1e-9)
    rel_err_b = abs_err_b / (np.abs(actuals) + 1e-9)

    print(f"Algo A (Lloyd-Max+QJL):")
    print(f"  실제 dot:      mean={np.mean(actuals):+.4f} std={np.std(actuals):.4f}")
    print(f"  추정 dot:      mean={np.mean(ests_a):+.4f} std={np.std(ests_a):.4f}")
    print(f"  MAE (절대):    {np.mean(abs_err_a):.4f}")
    print(f"  RMSE:          {np.sqrt(np.mean(abs_err_a**2)):.4f}")
    print(f"  중간 상대오차: {np.median(rel_err_a):.1%}")
    print(f"  <10% 상대오차: {np.mean(rel_err_a < 0.10):.1%}")
    print(f"  <50% 상대오차: {np.mean(rel_err_a < 0.50):.1%}")
    print()

    print(f"Algo B (Levels+Residual):")
    print(f"  추정 dot:      mean={np.mean(ests_b):+.4f} std={np.std(ests_b):.4f}")
    print(f"  MAE (절대):    {np.mean(abs_err_b):.4f}")
    print(f"  RMSE:          {np.sqrt(np.mean(abs_err_b**2)):.4f}")
    print(f"  중간 상대오차: {np.median(rel_err_b):.1%}")
    print(f"  <10% 상대오차: {np.mean(rel_err_b < 0.10):.1%}")
    print(f"  <50% 상대오차: {np.mean(rel_err_b < 0.50):.1%}")
    print()

    return vectors, queries, compressed_a, compressed_b, state_a, state_b


def benchmark_compression_ratio():
    """2. 압축률 비교"""
    print("=" * 60)
    print("2. 압축률 비교")
    print("=" * 60)

    fp32_bytes = DIM * 4
    # Actual storage (byte-aligned, with dim prefix)
    algo_a_actual = 4 + DIM * 4 + 4 + DIM * 1 + 4  # 1932 bytes (int32 per idx is wasteful)
    algo_b_actual = 4 + DIM * 1 + DIM * 1 + 2      # 774 bytes
    # Theoretical (bit-packed, no dim prefix, idx packed to 2-bit)
    algo_a_packed = DIM * 2 // 8 + DIM * 1 // 8 + 4 + 4  # ~168 bytes
    algo_b_packed = DIM * 3 // 8 + DIM * 1 // 8 + 2      # ~194 bytes
    fp32_bits = DIM * 32

    print(f"{'항목':<30} {'FP32':>12} {'Algo A':>12} {'Algo B':>12}")
    print("-" * 66)
    print(f"{'실제 Bytes/vector':<30} {fp32_bytes:>12} {algo_a_actual:>12} {algo_b_actual:>12}")
    print(f"{'압축률 (실제)':<30} {'1.00x':>12} {fp32_bytes/algo_a_actual:>10.2f}x {fp32_bytes/algo_b_actual:>10.2f}x")
    print(f"{'압축률 (bit-packed)':<30} {'1.00x':>12} {fp32_bytes/algo_a_packed:>10.1f}x {fp32_bytes/algo_b_packed:>10.1f}x")
    print(f"{'압축률 (이론, bits)':<30} {'1.00x':>12} {fp32_bits/(DIM*3):>10.1f}x {fp32_bits/(DIM*4+16):>10.1f}x")
    print()


def benchmark_recall(vectors, queries, compressed_a, compressed_b, state_a, state_b):
    """3. Recall@k: FP32 상위 결과 대비 압축 알고리즘의排名 보존율"""
    print("=" * 60)
    print("3. Recall@k (FP32 대비 순위 보존율)")
    print("=" * 60)

    n_search = len(vectors)
    n_queries = min(20, len(queries))

    for top_k in [1, 3, 5, 10, 20]:
        recalls_a = []
        recalls_b = []

        for q_idx in range(n_queries):
            q = queries[q_idx]
            q_a, q_qjl = prepare_query(q, state_a)
            q_b = prepare_query_v2(q, state_b)

            # FP32 exact scores
            fp32_scores = [float(np.dot(q, vectors[i])) for i in range(n_search)]
            fp32_top = set(np.argsort(fp32_scores)[-top_k:][::-1])

            # Algo A estimated scores
            scores_a = []
            for v_idx in range(n_search):
                idx_a, norm_a, qjl_a, r_norm_a = compressed_a[v_idx]
                est = estimate_a(q_a, q_qjl, state_a, idx_a, norm_a, qjl_a, r_norm_a)
                scores_a.append(est)
            algo_a_top = set(np.argsort(scores_a)[-top_k:][::-1])

            # Algo B estimated scores
            scores_b = []
            for v_idx in range(n_search):
                indices_b, signs_b, scale_b = compressed_b[v_idx]
                est = estimate_v2(q_b, state_b, indices_b, signs_b, scale_b, query_norm=1.0)
                scores_b.append(est)
            algo_b_top = set(np.argsort(scores_b)[-top_k:][::-1])

            recall_a = len(fp32_top & algo_a_top) / top_k
            recall_b = len(fp32_top & algo_b_top) / top_k
            recalls_a.append(recall_a)
            recalls_b.append(recall_b)

        print(f"Top-{top_k:<2}  | Algo A: {np.mean(recalls_a):.2%}  | Algo B: {np.mean(recalls_b):.2%}")

    print()


def benchmark_speed(vectors, queries, compressed_a, compressed_b, state_a, state_b):
    """4. 검색 속도 비교"""
    print("=" * 60)
    print("4. 검색 속도 (1000 vectors × 10 queries)")
    print("=" * 60)

    n_search = len(vectors)
    n_queries = min(10, len(queries))
    n_warmup = 3

    # Warm up
    for _ in range(n_warmup):
        q = queries[0]
        prepare_query(q, state_a)
        prepare_query_v2(q, state_b)

    # FP32 search
    start = time.perf_counter()
    for qi in range(n_queries):
        q = queries[qi]
        scores = [float(np.dot(q, vectors[i])) for i in range(n_search)]
        np.argsort(scores)[-TOP_K:]
    fp32_time = (time.perf_counter() - start) / n_queries

    # Algo A search
    start = time.perf_counter()
    for qi in range(n_queries):
        q = queries[qi]
        q_a, q_qjl = prepare_query(q, state_a)
        scores = []
        for v_idx in range(n_search):
            idx_a, norm_a, qjl_a, r_norm_a = compressed_a[v_idx]
            est = estimate_a(q_a, q_qjl, state_a, idx_a, norm_a, qjl_a, r_norm_a)
            scores.append(est)
        np.argsort(scores)[-TOP_K:]
    algo_a_search_time = (time.perf_counter() - start) / n_queries

    # Algo B search
    start = time.perf_counter()
    for qi in range(n_queries):
        q = queries[qi]
        q_b = prepare_query_v2(q, state_b)
        scores = []
        for v_idx in range(n_search):
            indices_b, signs_b, scale_b = compressed_b[v_idx]
            est = estimate_v2(q_b, state_b, indices_b, signs_b, scale_b, query_norm=1.0)
            scores.append(est)
        np.argsort(scores)[-TOP_K:]
    algo_b_search_time = (time.perf_counter() - start) / n_queries

    print(f"{'Method':<15} {'평균 검색 시간':>20} {'QPS':>10}")
    print("-" * 45)
    print(f"{'FP32':<15} {fp32_time*1000:>17.2f}ms {n_search/fp32_time:>10.0f}")
    print(f"{'Algo A':<15} {algo_a_search_time*1000:>17.2f}ms {n_search/algo_a_search_time:>10.0f}")
    print(f"{'Algo B':<15} {algo_b_search_time*1000:>17.2f}ms {n_search/algo_b_search_time:>10.0f}")

    rel_a = fp32_time / algo_a_search_time if algo_a_search_time > 0 else float('inf')
    rel_b = fp32_time / algo_b_search_time if algo_b_search_time > 0 else float('inf')
    print(f"{'Algo A vs FP32':<15} {'':>20} {rel_a:>9.2f}x")
    print(f"{'Algo B vs FP32':<15} {'':>20} {rel_b:>9.2f}x")
    print()


def benchmark_memory_store(vectors):
    """5. MemoryStore 통합 검증"""
    print("=" * 60)
    print("5. MemoryStore 통합 검증")
    print("=" * 60)

    results = {}
    for mode, label in [(None, 'FP32'), ('algo_a', 'Algo A'), ('algo_b', 'Algo B')]:
        db_path = f"_verify_{mode or 'fp32'}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        store = MemoryStore(db_path, compression=mode)
        n_store = min(500, len(vectors))

        start = time.time()
        for i in range(n_store):
            store.add(f"vector_{i}", vectors[i])
        store_time = time.time() - start

        status = store.stats()
        results[label] = {
            'store_time': store_time,
            'stats': status,
        }

        search_time = 0
        if mode is None:
            q = vectors[0]
            start = time.time()
            _ = store.search("vector", q, top_k=5)
            search_time = time.time() - start

        print(f"  {label}: 저장 {n_store}건 {store_time:.2f}s | "
              f"압축률: {status['compression_ratio']}x | "
              f"모드: {dict(status['compression_modes'])}")

        store.close()
        if os.path.exists(db_path):
            os.remove(db_path)

    print()


if __name__ == "__main__":
    print()
    print("▓" * 60)
    print("  TurboQuant 알고리즘 검증 리포트")
    print("  Algorithm A: Lloyd-Max 2-bit + QJL 1-bit unbiased correction")
    print("  Algorithm B: 3-bit normal quantile levels + 1-bit sign/scale residual")
    print("▓" * 60)
    print()

    try:
        vectors, queries, comp_a, comp_b, state_a, state_b = benchmark_compression_accuracy()
    except Exception as e:
        print(f"ERROR in accuracy benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    benchmark_compression_ratio()

    try:
        benchmark_recall(vectors, queries, comp_a, comp_b, state_a, state_b)
    except Exception as e:
        print(f"ERROR in recall benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_speed(vectors, queries, comp_a, comp_b, state_a, state_b)
    except Exception as e:
        print(f"ERROR in speed benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_memory_store(vectors)
    except Exception as e:
        print(f"ERROR in store integration: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    for f in os.listdir('.'):
        if f.startswith('_verify_') and f.endswith('.db'):
            os.remove(f)
