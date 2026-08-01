#!/usr/bin/env python3
"""
Benchmark: Rust turbo-memory-rs vs Python reference

Usage:
    cd /Users/hodorii/dev/turbo-memory-mcp/server-rs
    python3 benches/bench_runner.py

Output:
    Startup time (< 1s target)
    Search throughput (> 10x vs Python target)
    Per-operation latencies
"""

import json
import os
import subprocess
import sys
import time
import math
import struct

BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "turbo-memory-rs")
TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
STATE_BIN = os.path.join(TESTDATA, "state.bin")
INDEX_PATH = "/tmp/bench_rs.tmd"
DB_PATH = "/tmp/bench_rs.db"

# Add project root for Python reference import
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

DIM = 384
N_ENTRIES = 1000
N_SEARCHES = 100
TOP_K = 5


def make_embedding(seed: int):
    """Deterministic random embedding."""
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def clean_state():
    for p in [INDEX_PATH, DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def send_rpc(proc: subprocess.Popen, method: str, params: dict = None) -> dict:
    """Send JSON-RPC to stdio server and read response."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
    }
    if params is not None:
        req["params"] = params
    
    line = json.dumps(req) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()
    
    resp_line = proc.stdout.readline()
    if not resp_line:
        raise RuntimeError("Empty response from server (crashed?)")
    
    return json.loads(resp_line)


def bench_startup() -> float:
    """Measure startup time: binary launch to initialize response."""
    clean_state()
    
    start = time.perf_counter()
    proc = subprocess.Popen(
        [BINARY, "--index", INDEX_PATH, "--db", DB_PATH, "--state", STATE_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    
    # Send initialize
    try:
        resp = send_rpc(proc, "initialize")
        elapsed = time.perf_counter() - start
        assert resp.get("result", {}).get("serverInfo", {}).get("name") == "turbo-memory-rs", \
            f"Unexpected initialize response: {resp}"
        print(f"  Startup time: {elapsed*1000:.1f} ms")
        print(f"  Server info: {resp['result']['serverInfo']}")
        return elapsed, proc
    except Exception as e:
        proc.kill()
        print(f"  ERROR: {e}")
        raise


def bench_add(proc, n_entries: int) -> float:
    """Add N entries via MCP add tool."""
    print(f"\nAdding {n_entries} entries...")
    
    # Ping first to warm up
    send_rpc(proc, "ping")
    
    start = time.perf_counter()
    for i in range(n_entries):
        emb = make_embedding(i)
        params = {
            "name": "add",
            "arguments": {
                "text": f"Benchmark entry {i}",
                "embedding": emb,
                "importance": 0.5,
            }
        }
        resp = send_rpc(proc, "tools/call", params)
        if resp.get("error"):
            raise RuntimeError(f"Add error at {i}: {resp['error']}")
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i+1}/{n_entries} ({elapsed/(i+1)*1000:.2f} ms/add)")
    
    elapsed = time.perf_counter() - start
    avg = elapsed / n_entries * 1000
    print(f"  Added {n_entries} entries in {elapsed:.2f}s (avg {avg:.2f} ms/add)")
    return elapsed


def bench_search(proc, n_searches: int) -> dict:
    """Measure search throughput."""
    print(f"\nRunning {n_searches} searches...")
    
    latencies = []
    for i in range(n_searches):
        emb = make_embedding(i + 10000)
        params = {
            "name": "search",
            "arguments": {
                "query_vec": emb,
                "top_k": TOP_K,
            }
        }
        t0 = time.perf_counter()
        resp = send_rpc(proc, "tools/call", params)
        t = time.perf_counter() - t0
        latencies.append(t * 1000)
        
        if resp.get("error"):
            print(f"  Search error at {i}: {resp['error']}")
    
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    print(f"  Avg latency: {avg:.3f} ms")
    print(f"  P50 latency: {p50:.3f} ms")
    print(f"  P99 latency: {p99:.3f} ms")
    print(f"  Throughput:  {1000/avg:.0f} searches/sec")
    
    return {"avg_ms": avg, "p50_ms": p50, "p99_ms": p99, "n": n_searches}


def bench_search_python(n_entries: int, n_searches: int) -> dict:
    """Benchmark Python reference search (eden estimate) for comparison."""
    try:
        from turbo_quant_paper import TurboQuantState, compress, prepare_query, estimate
    except ImportError:
        print("  SKIP: turbo_quant_paper not importable (check PYTHONPATH)")
        return None
    
    import numpy as np
    
    print(f"\nPython reference: {n_searches} searches over {n_entries} entries...")
    
    state = TurboQuantState.build(DIM, b=3, seed=42)
    
    # Create entries
    entries = []
    for i in range(n_entries):
        v = make_embedding(i)
        v_np = np.array(v, dtype=np.float32)
        packed, norm, qjl, r_norm = compress(v_np, state)
        entries.append((packed, norm, qjl, r_norm))
    
    print(f"  Generated {len(entries)} compressed entries")
    
    latencies = []
    for i in range(n_searches):
        q = np.array(make_embedding(i + 10000), dtype=np.float32)
        q_rot, q_qjl = prepare_query(q, state)
        
        t0 = time.perf_counter()
        results = []
        for packed, norm, qjl, r_norm in entries:
            score = estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm)
            results.append(score)
        results.sort(reverse=True)
        top = results[:TOP_K]
        t = time.perf_counter() - t0
        latencies.append(t * 1000)
    
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    
    print(f"  Avg latency: {avg:.3f} ms")
    print(f"  P50 latency: {p50:.3f} ms")
    print(f"  Throughput:  {1000/avg:.0f} searches/sec")
    
    return {"avg_ms": avg, "p50_ms": p50, "n": n_searches}


def bench_stats(proc):
    """Query memory_stats."""
    resp = send_rpc(proc, "tools/call", {
        "name": "memory_stats",
        "arguments": {}
    })
    print(f"\nMemory stats: {resp['result']['content'][0]['text']}")
    return json.loads(resp['result']['content'][0]['text'])


def main():
    print("=" * 60)
    print("TurboMemory RS Benchmarks")
    print("=" * 60)
    print(f"Binary: {BINARY}")
    print(f"State:  {STATE_BIN}")
    print(f"Dim:    {DIM}")
    print(f"Entries:{N_ENTRIES}")
    print(f"Searches:{N_SEARCHES}")
    print()
    
    # 1. Startup time (cold)
    print("--- 1. Startup Time (Cold) ---")
    startup_time, proc = bench_startup()
    assert startup_time < 1.0, f"FAIL: Startup {startup_time*1000:.1f}ms > 1s target"
    print(f"  ✅ Startup < 1s: {startup_time*1000:.1f}ms")
    
    # 2. Add entries
    print("\n--- 2. Add Entries ---")
    bench_add(proc, N_ENTRIES)
    
    # 3. Memory stats
    stats = bench_stats(proc)
    assert stats["total_entries"] == N_ENTRIES, \
        f"Expected {N_ENTRIES} entries, got {stats['total_entries']}"
    assert stats["mmap_entries"] == N_ENTRIES, \
        f"Expected {N_ENTRIES} mmap entries, got {stats['mmap_entries']}"
    print(f"  ✅ {stats['total_entries']} entries in store ({stats['mmap_entries']} in mmap)")
    
    # 4. Search throughput (Rust)
    print("\n--- 3. Search Throughput (Rust) ---")
    rust_results = bench_search(proc, N_SEARCHES)
    
    # 5. Search throughput (Python reference)
    print("\n--- 4. Search Throughput (Python Reference) ---")
    py_results = bench_search_python(N_ENTRIES, N_SEARCHES)
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Startup time:  {startup_time*1000:.1f} ms {'✅' if startup_time < 1.0 else '❌'}")
    print(f"  Rust search:   {rust_results['avg_ms']:.3f} ms avg ({rust_results['n']} searches)")
    
    if py_results:
        speedup = py_results['avg_ms'] / rust_results['avg_ms']
        print(f"  Python search: {py_results['avg_ms']:.3f} ms avg ({py_results['n']} searches)")
        print(f"  Speedup:       {speedup:.1f}x {'✅' if speedup > 10 else '⚠️'}")
        
        if speedup > 10:
            print(f"\n  ✅ ALL BENCHMARKS PASS")
        else:
            print(f"\n  ⚠️  Speedup {speedup:.1f}x < 10x target")
    else:
        print(f"  Python ref:    SKIPPED (not importable)")
    
    # Cleanup
    print("\nShutting down server...")
    # Stats
    stats = bench_stats(proc)
    print(f"  Final stats: {stats['total_entries']} entries")
    
    # Send shutdown (not defined, just close stdin)
    proc.stdin.close()
    proc.wait(timeout=5)
    print("Done.")
    
    return rust_results, py_results, startup_time


if __name__ == "__main__":
    main()
