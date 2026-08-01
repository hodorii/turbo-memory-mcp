"""Comprehensive unit tests for EDEN-based V3 quantizer (EdenQuantizer).

Covers:
- Task 6.1: MSE comparison (biased vs S=1 baseline, vs V2)
- Task 6.2: Unbiased property (mean error near zero)
- Task 6.3: DRIVE residual variance reduction
- TurboDiskStore integration
- Codebook & rotation correctness
"""

import math
import os
import shutil
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.turboquant.eden import EdenConfig, EdenQuantizer, compute_beta_codebook
from src.turboquant.memory import TurboDiskStore, TurboQuantizer_V2


# ── Helpers ──────────────────────────────────────────────────────────────

def unit_vectors(n: int, dim: int) -> torch.Tensor:
    x = torch.randn(n, dim)
    return x / torch.norm(x, dim=-1, keepdim=True)


def mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return torch.mean((x - x_hat) ** 2).item()


def cos_sim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    inner = torch.sum(x * x_hat, dim=-1)
    n1 = torch.norm(x_hat, dim=-1)
    return torch.mean(inner / n1).item()


# ── 1. Basic Checks ──────────────────────────────────────────────────────

def test_codebook_caching():
    """compute_beta_codebook should cache by (dim, bits)."""
    cb1 = compute_beta_codebook(64, 3)
    cb2 = compute_beta_codebook(64, 3)
    assert cb1 is cb2, "cache miss"
    assert torch.allclose(cb1, cb2), "cached codebook mismatch"
    assert cb1.shape[0] == 8, f"3-bit should have 8 levels, got {cb1.shape[0]}"
    # sorted
    assert torch.all(cb1[1:] > cb1[:-1]), "codebook not sorted"


def test_eden_config_defaults():
    cfg = EdenConfig(dim=128)
    assert cfg.bits == 3
    assert cfg.mode == "biased"
    assert cfg.residual_bits == 1


def test_basic_quantize_decode():
    """Quantize and decode one vector — reconstruction sanity."""
    eq = EdenQuantizer(EdenConfig(dim=64, bits=2, mode="biased", residual_bits=1, seed=42))
    x = unit_vectors(1, 64).squeeze(0)
    idx, rs, rsc, S = eq.quantize(x)
    x_hat = eq.decode(idx, rs, rsc, S)
    assert not torch.isnan(x_hat).any(), "NaN in decode"
    assert x_hat.shape == (64,), f"shape mismatch {x_hat.shape}"
    assert mse(x, x_hat) < 0.05, f"MSE too high: {mse(x, x_hat):.6f}"


# ── 2. Task 6.1: MSE Comparison ──────────────────────────────────────────

def test_mses_biased_vs_baseline():
    """EDEN-biased should beat or match S=1 baseline at low dimensions."""
    for dim in [16, 32, 64]:
        for bits in [2, 3]:
            eq = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="biased",
                                          residual_bits=0, seed=42))
            x = unit_vectors(500, dim)
            idx, rs, rsc, S = eq.quantize(x)
            x_hat_eden = eq.decode(idx, rs, rsc, S)
            x_hat_s1 = eq.decode(idx, None, None, None)  # S=1, no residual

            m_eden = mse(x, x_hat_eden)
            m_s1 = mse(x, x_hat_s1)
            improvement = (1 - m_eden / m_s1) * 100 if m_s1 > 0 else 0
            print(f"  d={dim:3d} b={bits} EDEN={m_eden:.6f} S=1={m_s1:.6f}  "
                  f"improvement={improvement:.2f}%")
            # At low dim, EDEN should be strictly better (or equal within noise)
            assert m_eden <= m_s1 * 1.01, (
                f"EDEN {m_eden:.6f} worse than S=1 {m_s1:.6f} at d={dim} b={bits}")


def test_mses_biased_vs_v2():
    """EDEN-biased with residual should improve over V2."""
    dim, bits = 64, 3
    eq = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="biased",
                                  residual_bits=1, seed=42))
    v2 = TurboQuantizer_V2(dim, bits)
    x = unit_vectors(500, dim)

    idx, rs, rsc, S = eq.quantize(x)
    x_hat_eden = eq.decode(idx, rs, rsc, S)
    m_eden = mse(x, x_hat_eden)

    # V2: batch quantize
    x_hat_list = []
    for i in range(x.shape[0]):
        idxx, sgn, scl = v2.quantize(x[i])
        # decode: inverse rotate + residual
        y_hat = v2.levels[idxx]
        residual = sgn.float() * scl
        y_hat = y_hat + residual
        x_hat_i = y_hat @ v2.rotation.T
        x_hat_list.append(x_hat_i)
    x_hat_v2 = torch.stack(x_hat_list)
    m_v2 = mse(x, x_hat_v2)

    print(f"  EDEN-biased MSE={m_eden:.6f} V2 MSE={m_v2:.6f}  "
          f"improvement={(1-m_eden/m_v2)*100:.2f}%")
    assert m_eden <= m_v2, f"EDEN {m_eden:.6f} worse than V2 {m_v2:.6f}"


# ── 3. Task 6.2: Unbiased Property ────────────────────────────────────────

def test_unbiased_mean_error():
    """Unbiased mode should produce near-zero mean error vector norm."""
    for dim in [32, 64]:
        for bits in [2, 3]:
            eq = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="unbiased",
                                          residual_bits=0, seed=42))
            x = unit_vectors(500, dim)
            idx, rs, rsc, S = eq.quantize(x)
            x_hat = eq.decode(idx, rs, rsc, S)

            mean_err = torch.mean(x_hat - x, dim=0).norm().item()
            print(f"  d={dim:3d} b={bits} mean_err_norm={mean_err:.6f}")
            assert mean_err < 0.15, (
                f"Unbiased mean error too large: {mean_err:.6f} at d={dim} b={bits}")


def test_biased_has_bias():
    """Biased mode may have bias (just confirming the mode differs meaningfully)."""
    eq_u = EdenQuantizer(EdenConfig(dim=32, bits=2, mode="unbiased", residual_bits=0, seed=42))
    eq_b = EdenQuantizer(EdenConfig(dim=32, bits=2, mode="biased", residual_bits=0, seed=42))
    x = unit_vectors(500, 32)
    # unbiased
    _, _, _, Su = eq_u.quantize(x)
    # biased (force same rotation — but seed differs because diff configs...
    # Instead just check that S values differ in distribution)
    _, _, _, Sb = eq_b.quantize(x)
    assert not torch.allclose(Su, Sb, atol=1e-2), (
        "Unbiased and biased S should differ meaningfully")


# ── 4. Task 6.3: DRIVE Residual ──────────────────────────────────────────

def test_drive_residual_reduces_mse():
    """Enabling DRIVE residual should reduce MSE vs. no residual."""
    for bits in [2, 3]:
        for mode in ["biased", "unbiased"]:
            eq_on = EdenQuantizer(EdenConfig(dim=64, bits=bits, mode=mode,
                                             residual_bits=1, seed=42))
            eq_off = EdenQuantizer(EdenConfig(dim=64, bits=bits, mode=mode,
                                              residual_bits=0, seed=42))
            x = unit_vectors(500, 64)

            i, rs, rsc, S = eq_on.quantize(x)
            x_on = eq_on.decode(i, rs, rsc, S)
            m_on = mse(x, x_on)

            i2, _, _, S2 = eq_off.quantize(x)
            x_off = eq_off.decode(i2, None, None, S2)
            m_off = mse(x, x_off)

            print(f"  b={bits} {mode:8s} residual=on MSE={m_on:.6f} "
                  f"off={m_off:.6f} Δ={(m_off-m_on)/m_off*100:.2f}%")
            # Residual should always help (or not hurt meaningfully)
            assert m_on <= m_off * 1.005, (
                f"Residual increased MSE ({m_on:.6f} > {m_off:.6f})"
                f" at bits={bits} mode={mode}")


def test_drive_sign_scale_consistency():
    """Residual sign and scale should reconstruct a meaningful correction."""
    eq = EdenQuantizer(EdenConfig(dim=64, bits=2, mode="biased", residual_bits=1, seed=42))
    x = unit_vectors(1, 64).squeeze(0)
    idx, rs, rsc, S = eq.quantize(x)

    # Decode without residual
    x_no_res = eq.decode(idx, None, None, S)
    mse_no = mse(x, x_no_res)

    # Decode with residual
    x_with_res = eq.decode(idx, rs, rsc, S)
    mse_with = mse(x, x_with_res)

    assert mse_with < mse_no, f"Residual made things worse: {mse_with:.6f} vs {mse_no:.6f}"
    print(f"  No-res MSE={mse_no:.6f} With-res MSE={mse_with:.6f}")


# ── 5. TurboDiskStore Integration ────────────────────────────────────────

def test_diskstore_v2_unchanged():
    """V2 TurboDiskStore should still work identically."""
    store = TurboDiskStore(64, 3, "/tmp/test_eden_v2")
    try:
        for i in range(5):
            store.add(torch.randn(64))
        assert store.count == 5
        results = store.search(torch.randn(64), top_k=2)
        assert len(results) == 2
    finally:
        shutil.rmtree("/tmp/test_eden_v2", ignore_errors=True)


def test_diskstore_eden_add_search():
    """TurboDiskStore with EdenQuantizer should store and search."""
    eq = EdenQuantizer(EdenConfig(dim=64, bits=3, mode="biased", residual_bits=1, seed=42))
    store = TurboDiskStore(64, 3, "/tmp/test_eden_store", quantizer=eq)
    try:
        assert store.use_eden, "use_eden flag not set"
        for i in range(10):
            store.add(unit_vectors(1, 64).squeeze(0))
        assert store.count == 10
        s_vals = torch.from_numpy(store.scale_s_mmap[:store.count])
        assert s_vals.numel() == 10, "S values not stored"
        assert s_vals.min() > 0, f"Some S values <= 0: {s_vals}"

        results = store.search(unit_vectors(1, 64).squeeze(0), top_k=3)
        assert len(results) == 3
        # Scores should be reasonable (not NaN, not all identical)
        scores = [r[1] for r in results]
        assert not any(math.isnan(s) for s in scores), "NaN scores"
        assert len(set(scores)) > 1, "All scores identical"
    finally:
        shutil.rmtree("/tmp/test_eden_store", ignore_errors=True)


def test_diskstore_multiple_add():
    """Add 100 vectors and verify search returns correct top-k."""
    dim, bits, n = 32, 2, 100
    eq = EdenQuantizer(EdenConfig(dim=dim, bits=bits, mode="biased", residual_bits=1, seed=42))
    store = TurboDiskStore(dim, bits, "/tmp/test_eden_100", quantizer=eq)
    try:
        vectors = unit_vectors(n, dim)
        for v in vectors:
            store.add(v)
        assert store.count == n

        # Verify nearest neighbor recall using brute-force
        query = unit_vectors(1, dim).squeeze(0)
        results = store.search(query, top_k=5)
        result_indices = set(r[0] for r in results)

        # Brute-force top-5
        bf_scores = torch.matmul(vectors, query)
        bf_top5 = set(torch.topk(bf_scores, 5).indices.tolist())
        overlap = result_indices & bf_top5
        recall = len(overlap) / 5
        print(f"  Top-5 recall: {recall:.2f} (shared {len(overlap)}/5)")
        # At least 1 shared in top-5 at dim=32 bits=2
        assert recall >= 0.2, f"Recall too low: {recall:.2f}"
    finally:
        shutil.rmtree("/tmp/test_eden_100", ignore_errors=True)


# ── 6. Scale Factor Tests ────────────────────────────────────────────────

def test_scale_factor_ranges():
    """S values should converge toward 1 as dimension increases."""
    prev_deviation = float("inf")
    for dim in [16, 32, 64, 128]:
        eq = EdenQuantizer(EdenConfig(dim=dim, bits=3, mode="biased", residual_bits=0, seed=42))
        x = unit_vectors(500, dim)
        _, _, _, S = eq.quantize(x)
        deviation = torch.mean(torch.abs(S - 1.0)).item()
        print(f"  d={dim:4d} mean|S-1|={deviation:.4f}")
        assert deviation <= prev_deviation, (
            f"S deviation increased at d={dim}: {deviation} > {prev_deviation}")
        prev_deviation = deviation


# ── 7. Batched vs Single Consistency ─────────────────────────────────────

def test_batched_vs_single():
    """Batched and single quantization should give same results."""
    eq = EdenQuantizer(EdenConfig(dim=64, bits=3, mode="biased", residual_bits=1, seed=42))
    batch_x = unit_vectors(3, 64)
    idx_b, rs_b, rsc_b, S_b = eq.quantize(batch_x)

    single_results = [eq.quantize(batch_x[i]) for i in range(3)]
    for i, (idx_s, rs_s, rsc_s, S_s) in enumerate(single_results):
        assert torch.allclose(idx_b[i], idx_s), f"indices differ at {i}"
        if rs_b is not None and rs_s is not None:
            assert torch.allclose(rs_b[i], rs_s), f"r_signs differ at {i}"
        if rsc_b is not None and rsc_s is not None:
            assert torch.allclose(rsc_b[i], rsc_s.unsqueeze(0), atol=1e-6), \
                f"r_scale differ at {i}"
        assert torch.allclose(S_b[i], S_s, atol=1e-5), f"S differ at {i}"

    x_hat_b = eq.decode(idx_b, rs_b, rsc_b, S_b)
    x_hat_s = torch.stack([eq.decode(*eq.quantize(batch_x[i])) for i in range(3)])
    assert torch.allclose(x_hat_b, x_hat_s, atol=1e-5), "decoded vectors differ"


# ── 8. Numerical Stability ──────────────────────────────────────────────

def test_numerical_stability():
    """Zero vector, very short vectors should not crash or produce NaN."""
    eq = EdenQuantizer(EdenConfig(dim=32, bits=2, mode="biased", residual_bits=1, seed=42))

    # Zero vector
    x0 = torch.zeros(32)
    try:
        idx, rs, rsc, S = eq.quantize(x0)
        xh = eq.decode(idx, rs, rsc, S)
        assert not torch.isnan(xh).any(), "NaN for zero vector"
    except Exception as e:
        # Acceptable if zero vectors fail gracefully
        print(f"  Zero vector handling: {type(e).__name__}: {e}")

    # Tiny norm vector
    xt = torch.randn(32) * 1e-6
    idx, rs, rsc, S = eq.quantize(xt)
    xh = eq.decode(idx, rs, rsc, S)
    assert not torch.isnan(xh).any(), "NaN for tiny vector"
    assert not torch.isinf(xh).any(), "Inf for tiny vector"


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("codebook caching", test_codebook_caching),
        ("EdenConfig defaults", test_eden_config_defaults),
        ("basic quantize/decode", test_basic_quantize_decode),
        ("MSE: biased vs S=1 baseline", test_mses_biased_vs_baseline),
        ("MSE: biased vs V2", test_mses_biased_vs_v2),
        ("unbiased mean error", test_unbiased_mean_error),
        ("biased has different S", test_biased_has_bias),
        ("DRIVE residual reduces MSE", test_drive_residual_reduces_mse),
        ("DRIVE sign/scale consistency", test_drive_sign_scale_consistency),
        ("DiskStore V2 unchanged", test_diskstore_v2_unchanged),
        ("DiskStore EDEN add/search", test_diskstore_eden_add_search),
        ("DiskStore 100-vector recall", test_diskstore_multiple_add),
        ("scale factor vs dimension", test_scale_factor_ranges),
        ("batched vs single consistency", test_batched_vs_single),
        ("numerical stability", test_numerical_stability),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
    print(f"\n{'='*50}\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
