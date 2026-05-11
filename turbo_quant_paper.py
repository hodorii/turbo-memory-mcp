import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


def _lloyd_max(dist_sampler, n_levels, n_iters=50, n_samples=200_000):
    samples = dist_sampler(n_samples)
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(n_iters):
        idx = np.clip(np.digitize(samples, (centroids[:-1] + centroids[1:]) / 2), 0, n_levels - 1)
        new = np.array([
            samples[idx == i].mean() if (idx == i).any() else c
            for i, c in enumerate(centroids)
        ])
        if np.allclose(centroids, new, atol=1e-6):
            break
        centroids = new
    return np.sort(centroids)


def _sphere_coord(d, n):
    z = np.random.beta((d - 1) / 2.0, (d - 1) / 2.0, n)
    return 2 * z - 1


_CODEBOOKS = {}


def _beta_centroids(d, b):
    key = (d, b)
    if key not in _CODEBOOKS:
        _CODEBOOKS[key] = _lloyd_max(lambda n: _sphere_coord(d, n), 2 ** b)
    return _CODEBOOKS[key]


@dataclass
class TurboQuantState:
    dim: int
    b: int
    rotation: np.ndarray
    centroids: np.ndarray
    qjl_matrix: np.ndarray

    @classmethod
    def build(cls, dim, b=3, seed=42):
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(R)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return cls(
            dim=dim, b=b,
            rotation=Q.astype(np.float32),
            centroids=_beta_centroids(dim, b).astype(np.float32),
            qjl_matrix=rng.standard_normal((dim, dim)).astype(np.float32),
        )


def _unpack_numpy(packed, dim):
    """Pure numpy 3-bit extraction — no Python loops.
    Processes 48 groups of 3 bytes in parallel.
    Returns uint8[dim] centroid indices.
    """
    packed_3 = packed.reshape(-1, 3)
    b0, b1, b2 = packed_3[:, 0], packed_3[:, 1], packed_3[:, 2]
    i0 = b0 & 7
    i1 = (b0 >> 3) & 7
    i2 = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
    i3 = (b1 >> 1) & 7
    i4 = (b1 >> 4) & 7
    i5 = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
    i6 = (b2 >> 2) & 7
    i7 = b2 >> 5
    return np.stack([i0, i1, i2, i3, i4, i5, i6, i7], axis=1).ravel()


def compress(x, state):
    x = x.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(x))
    if norm < 1e-12:
        n_bytes = math.ceil(state.dim * state.b / 8)
        return (np.zeros(n_bytes, dtype=np.uint8), 0.0,
                np.ones(state.dim, dtype=np.int8), 0.0)
    x_unit = x / norm
    rotated = state.rotation @ x_unit
    thresholds = (state.centroids[:-1] + state.centroids[1:]) / 2
    indices = np.digitize(rotated, thresholds).astype(np.uint8)
    np.clip(indices, 0, (2 ** state.b) - 1, out=indices)
    packed = _pack_bits(indices, state.b)
    x_hat = (state.rotation.T @ state.centroids[indices]) * norm
    residual = x - x_hat
    r_norm = float(np.linalg.norm(residual))
    if r_norm > 1e-12:
        qjl = np.sign(state.qjl_matrix @ residual).astype(np.int8)
        qjl[qjl == 0] = 1
    else:
        qjl = np.ones(state.dim, dtype=np.int8)
    return packed, norm, qjl, r_norm


def decompress(packed, norm, state):
    if norm < 1e-12:
        return np.zeros(state.dim, dtype=np.float32)
    indices = _unpack_numpy(packed, state.dim)
    return (state.rotation.T @ state.centroids[indices]) * norm


def prepare_query(query, state):
    query = query.astype(np.float32, copy=False)
    return state.rotation @ query, state.qjl_matrix @ query


def estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm):
    if norm < 1e-12:
        return 0.0
    indices = _unpack_numpy(packed, state.dim)
    s1 = float(np.dot(state.centroids[indices], q_rot)) * norm
    c = math.sqrt(math.pi / 2.0) / state.dim
    s2 = c * r_norm * float(np.dot(q_qjl.astype(np.float32), qjl.astype(np.float32)))
    return s1 + s2


def estimate_preunpacked(indices, norm, qjl, r_norm, state, q_rot, q_qjl):
    """estimate() variant taking pre-unpacked uint8 indices (avoids repeated unpack)."""
    if norm < 1e-12:
        return 0.0
    s1 = float(np.dot(state.centroids[indices], q_rot)) * norm
    c = math.sqrt(math.pi / 2.0) / state.dim
    s2 = c * r_norm * float(np.dot(q_qjl.astype(np.float32), qjl.astype(np.float32)))
    return s1 + s2


def batch_search(q_rot, q_qjl, state, paper_rows: list) -> list:
    """Score ALL stored paper vectors against a query using pure numpy BLAS.

    paper_rows: list of (entry_id, text, indices_uint8, norm, qjl, r_norm)
    Returns: list of (entry_id, text, score) sorted descending.
    """
    if not paper_rows:
        return []
    n = len(paper_rows)
    dim = state.dim

    centroids = state.centroids
    c = math.sqrt(math.pi / 2.0) / dim

    idx_arr = np.empty((n, dim), dtype=np.uint8)
    norm_arr = np.empty(n, dtype=np.float32)
    qjl_arr = np.empty((n, dim), dtype=np.int8)
    r_norm_arr = np.empty(n, dtype=np.float32)
    ids, texts = [], []
    for i, (eid, text, idx, norm, qjl, r_norm) in enumerate(paper_rows):
        idx_arr[i] = idx
        norm_arr[i] = norm
        qjl_arr[i] = qjl
        r_norm_arr[i] = r_norm
        ids.append(eid)
        texts.append(text)

    s1 = np.dot(centroids[idx_arr], q_rot) * norm_arr
    s2 = c * r_norm_arr * np.dot(qjl_arr.astype(np.float32), q_qjl)
    scores = s1 + s2

    order = np.argsort(scores)[::-1]
    return [(ids[i], texts[i], float(scores[i])) for i in order]


def _pack_bits(indices, b):
    dim = len(indices)
    n_bytes = math.ceil(dim * b / 8)
    out = np.zeros(n_bytes, dtype=np.uint8)
    byte_i, bit_pos, mask = 0, 0, (1 << b) - 1
    for idx in indices:
        for bit in range(b):
            if (idx >> bit) & 1:
                out[byte_i] |= (1 << bit_pos)
            bit_pos += 1
            if bit_pos == 8:
                byte_i += 1
                bit_pos = 0
    return out


def _unpack_bits(packed, b, dim):
    mask = (1 << b) - 1
    indices = np.zeros(dim, dtype=np.uint8)
    byte_i, bit_offset, consumed = 0, 0, 0
    while consumed < dim * b:
        avail = 8 - bit_offset
        chunk = packed[byte_i] >> bit_offset
        if avail >= b:
            indices[consumed // b] = chunk & mask
            bit_offset += b
            consumed += b
            if bit_offset == 8:
                byte_i += 1
                bit_offset = 0
        else:
            first = avail
            second = b - first
            indices[consumed // b] = (chunk & ((1 << first) - 1)) | ((packed[byte_i + 1] & ((1 << second) - 1)) << first)
            byte_i += 1
            bit_offset = second
            consumed += b
    return indices


def compress_blob(packed, qjl, qjl_packed, norm, r_norm):
    import struct
    qjl_bytes = qjl_packed.tobytes()
    packed_bytes = packed.tobytes()
    header = struct.pack('<ii', len(qjl_bytes), len(packed_bytes))
    return header + struct.pack('<ffi', norm, r_norm, len(qjl_bytes)) + qjl_bytes + packed_bytes


def decompress_blob(blob, dim, b):
    import struct
    n_qjl, n_packed = struct.unpack('<ii', blob[:8])
    norm, r_norm, _ = struct.unpack('<ffi', blob[8:20])
    qjl = np.frombuffer(blob[20:20+n_qjl], dtype=np.int8).copy()
    packed = np.frombuffer(blob[20+n_qjl:20+n_qjl+n_packed], dtype=np.uint8).copy()
    return packed, norm, qjl, r_norm


if __name__ == "__main__":
    import time
    DIM, N = 384, 1000
    state = TurboQuantState.build(DIM, b=3, seed=0)
    print(f"d={DIM}, b={state.b}, c[0]={state.centroids[0]:.4f}")

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    compressed = [compress(v, state) for v in vecs]
    queries = vecs[:50]

    # Accuracy
    est_dots, true_dots = [], []
    for q in queries:
        q_rot, q_qjl = prepare_query(q, state)
        for i, v in enumerate(vecs):
            packed, norm, qjl, r_norm = compressed[i]
            est_dots.append(estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm))
            true_dots.append(float(np.dot(q, v)))

    est, tru = np.array(est_dots), np.array(true_dots)
    mae = np.mean(np.abs(tru - est))
    rmse = np.sqrt(np.mean((tru - est)**2))
    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}")

    fp32 = DIM * 4
    algo = math.ceil(DIM * state.b / 8) + 4 + 4 + DIM + 4
    print(f"FP32={fp32}B  Compressed={algo}B  Ratio={fp32/algo:.1f}x")
    print()

    # Speed: batch_search vs per-vector estimate
    paper_rows = []
    for i, (packed, norm, qjl, r_norm) in enumerate(compressed):
        idx = _unpack_numpy(packed, DIM)
        paper_rows.append((str(i), f"vec_{i}", idx, norm, qjl, r_norm))

    Q = 50
    q_rot, q_qjl = prepare_query(queries[0], state)

    t0 = time.time()
    for _ in range(Q):
        results = batch_search(q_rot, q_qjl, state, paper_rows)
    t1 = time.time()
    print(f"batch_search:  {(t1-t0)*1e6/Q:.1f} us/search  ({(t1-t0)/(Q*N)*1e6:.3f} us/vec)")
    print(f"  Top-3: {[float(f'{s:.4f}') for _,_,s in results[:3]]}")

    # Per-vector estimate
    t0 = time.time()
    for _ in range(Q):
        for packed, norm, qjl, r_norm in compressed:
            estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm)
    t1 = time.time()
    print(f"per-vec estimate: {(t1-t0)*1e6/Q:.1f} us/search  ({(t1-t0)/(Q*N)*1e6:.3f} us/vec)")

    # Pre-unpacked per-vector
    t0 = time.time()
    for _ in range(Q):
        for idx, norm, qjl, r_norm in [(paper_rows[i][2], paper_rows[i][3], paper_rows[i][4], paper_rows[i][5]) for i in range(N)]:
            estimate_preunpacked(idx, norm, qjl, r_norm, state, q_rot, q_qjl)
    t1 = time.time()
    print(f"preunpacked per-vec: {(t1-t0)*1e6/Q:.1f} us/search  ({(t1-t0)/(Q*N)*1e6:.3f} us/vec)")

    # FP32 reference
    fp32_vecs = vecs.copy()
    t0 = time.time()
    for _ in range(Q):
        np.dot(fp32_vecs, q_rot)
    t1 = time.time()
    print(f"FP32 batch:          {(t1-t0)*1e6/Q:.1f} us/search  ({(t1-t0)/(Q*N)*1e6:.3f} us/vec)")