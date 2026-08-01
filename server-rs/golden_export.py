#!/usr/bin/env python3
"""
Export golden EDEN V3 test vectors for Rust verification.

Generates a reproducible state + query + compressed vector using
the Python reference implementation and writes binary files that
the Rust tests can load for integration verification.

Usage:
    cd /Users/hodorii/dev/turbo-memory-mcp
    python3 server-rs/golden_export.py

Outputs in server-rs/testdata/:
    state.bin        - EdenState (dim, bits, rotation[dim*dim], centroids[8])
    golden_packed.bin - Packed 3-bit indices (144 bytes for dim=384)
    golden_q_rot.bin  - q_rot vector (384 float32)
    golden_q_qjl.bin  - q_qjl vector (384 float32)
    golden_stored_qjl.bin - stored QJL vector (384 int8)
    golden_values.json - norm, r_norm, expected_score
"""

import json
import numpy as np
import struct
import math
import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from turbo_quant_paper import (
    TurboQuantState,
    compress,
    prepare_query,
    estimate,
    _unpack_numpy,
    _pack_bits,
)

DIM = 384
SEED = 42

def main():
    # 1. Build deterministic state
    state = TurboQuantState.build(DIM, b=3, seed=SEED)
    print(f"state built: dim={state.dim}, b={state.b}")
    print(f"centroids: {state.centroids.tolist()}")
    print(f"rotation shape: {state.rotation.shape}")
    print(f"qjl_matrix shape: {state.qjl_matrix.shape}")

    # 2. Generate a deterministic random query vector
    rng = np.random.default_rng(SEED)
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    print(f"query[:8]: {query[:8].tolist()}")

    # 3. Compress the query to create a "stored" vector
    #    (treat query as a stored memory entry for the golden test)
    packed, norm, qjl, r_norm = compress(query, state)
    print(f"norm: {norm}")
    print(f"r_norm: {r_norm}")
    print(f"packed len: {len(packed)}")
    print(f"packed[:6]: {packed[:6].tolist()}")
    print(f"qjl[:16]: {qjl[:16].tolist()}")

    # 4. Unpack to verify
    indices = _unpack_numpy(packed, DIM)
    print(f"indices[:16]: {indices[:16].tolist()}")

    # 5. Prepare a *different* query vector (for scoring)
    rng2 = np.random.default_rng(SEED + 1)
    q2 = rng2.standard_normal(DIM).astype(np.float32)
    q2 /= np.linalg.norm(q2)
    q_rot, q_qjl = prepare_query(q2, state)
    print(f"q_rot[:16]: {q_rot[:16].tolist()}")
    print(f"q_qjl[:16]: {q_qjl[:16].tolist()}")

    # 6. Compute expected score
    expected_score = estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm)
    print(f"expected_score (estimate): {expected_score:.10f}")

    # 7. Verify with pre-unpacked version
    s2 = (math.sqrt(math.pi / 2.0) / DIM) * r_norm * float(np.dot(q_qjl.astype(np.float32), qjl.astype(np.float32)))
    s1 = float(np.dot(state.centroids[indices], q_rot)) * norm
    print(f"s1: {s1:.10f}, s2: {s2:.10f}, total: {s1+s2:.10f}")

    # 8. Write binary files
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
    os.makedirs(outdir, exist_ok=True)

    # state.bin: u32 dim + u32 bits + rotation [dim*dim] f32 + qjl_matrix [dim*dim] f32 + centroids [8] f32
    with open(os.path.join(outdir, 'state.bin'), 'wb') as f:
        f.write(struct.pack('<II', state.dim, state.b))
        f.write(state.rotation.astype(np.float32).tobytes())
        f.write(state.qjl_matrix.astype(np.float32).tobytes())
        f.write(state.centroids.astype(np.float32).tobytes())
    print(f"Wrote state.bin ({os.path.getsize(os.path.join(outdir, 'state.bin'))} bytes)")

    # golden_packed.bin: raw packed bytes
    with open(os.path.join(outdir, 'golden_packed.bin'), 'wb') as f:
        f.write(packed.tobytes())
    print(f"Wrote golden_packed.bin ({os.path.getsize(os.path.join(outdir, 'golden_packed.bin'))} bytes)")

    # golden_q_rot.bin: f32 array
    with open(os.path.join(outdir, 'golden_q_rot.bin'), 'wb') as f:
        f.write(q_rot.astype(np.float32).tobytes())
    print(f"Wrote golden_q_rot.bin ({os.path.getsize(os.path.join(outdir, 'golden_q_rot.bin'))} bytes)")

    # golden_q_qjl.bin: f32 array
    with open(os.path.join(outdir, 'golden_q_qjl.bin'), 'wb') as f:
        f.write(q_qjl.astype(np.float32).tobytes())
    print(f"Wrote golden_q_qjl.bin ({os.path.getsize(os.path.join(outdir, 'golden_q_qjl.bin'))} bytes)")

    # golden_stored_qjl.bin: i8 array
    with open(os.path.join(outdir, 'golden_stored_qjl.bin'), 'wb') as f:
        f.write(qjl.astype(np.int8).tobytes())
    print(f"Wrote golden_stored_qjl.bin ({os.path.getsize(os.path.join(outdir, 'golden_stored_qjl.bin'))} bytes)")

    # golden_values.json: norm, r_norm, expected_score
    values = {
        "norm": float(norm),
        "r_norm": float(r_norm),
        "expected_score": float(expected_score),
        "dim": DIM,
    }
    with open(os.path.join(outdir, 'golden_values.json'), 'w') as f:
        json.dump(values, f, indent=2)
    print(f"Wrote golden_values.json: {values}")

    print("\nDone! Fresh golden data written to testdata/")
    print(f"\nTo update Rust tests, embed these values:")
    print(f"  # Centroids: {state.centroids.tolist()}")
    print(f"  # NORM: {norm}")
    print(f"  # R_NORM: {r_norm}")
    print(f"  # EXPECTED_SCORE: {expected_score}")
    print(f"  # Q_ROT_16: {q_rot[:16].tolist()}")
    print(f"  # Q_QJL_16: {q_qjl[:16].tolist()}")
    print(f"  # STORED_QJL_16: {qjl[:16].tolist()}")
    print(f"  # INDICES_16: {indices[:16].tolist()}")
    print(f"  # Packed[:6]: {packed[:6].tolist()}")


if __name__ == "__main__":
    main()
