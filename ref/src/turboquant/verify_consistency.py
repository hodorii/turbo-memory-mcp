import json
import numpy as np
import torch
from src.turboquant.quantizers import QuantizerRegistry

def generate_test_vectors(filename="test_vectors.json", num_vectors=10, dim=1024):
    print(f"Generating {num_vectors} test vectors...")
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    
    with open(filename, "w") as f:
        json.dump(vectors.tolist(), f)
    print(f"Saved to {filename}")

def save_quantizer_state(quantizer, filename="quantizer_state.json"):
    state = {
        "algo_id": quantizer.__class__.__name__,
        "dim": quantizer.dim,
        "bits": quantizer.bits,
    }
    
    # Handle rotation (direct or internal)
    if hasattr(quantizer, 'rotation'):
        state["rotation"] = quantizer.rotation.flatten().tolist()
    elif hasattr(quantizer, '_internal') and hasattr(quantizer._internal, 'rotation'):
        state["rotation"] = quantizer._internal.rotation.flatten().tolist()
        
    # Handle codebook (direct or internal)
    if hasattr(quantizer, 'codebook'):
        state["codebook"] = quantizer.codebook.flatten().tolist()
    elif hasattr(quantizer, '_internal') and hasattr(quantizer._internal, 'codebook'):
        state["codebook"] = quantizer._internal.codebook.flatten().tolist()
        
    with open(filename, "w") as f:
        json.dump(state, f)
    print(f"State saved to {filename}")

def run_python_reference(algo_id="EDEN", vectors_file="test_vectors.json", dim=1024, bits=3, output_file="python_results.json"):
    with open(vectors_file, "r") as f:
        vectors = np.array(json.load(f), dtype=np.float32)
    
    quantizer = QuantizerRegistry.get_quantizer(algo_id, dim=dim, bits=bits, seed=42)
    
    results = []
    for v in vectors:
        q_res = quantizer.quantize(v)
        x_hat = quantizer.decode(q_res)
        score = quantizer.calculate_score(v, q_res)
        
        results.append({
            "values": q_res.values.tolist(),
            "signs": q_res.signs.tolist() if q_res.signs is not None else None,
            "scale": q_res.scale,
            "x_hat": x_hat.tolist(),
            "score": score
        })
    
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Python results saved to {output_file}")
    return results

def verify_consistency(python_results_file="python_results.json", rust_results_file="rust_results.json"):
    try:
        with open(python_results_file, "r") as f:
            python_results = json.load(f)
        with open(rust_results_file, "r") as f:
            rust_results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if len(python_results) != len(rust_results):
        print(f"Error: Result count mismatch! Python: {len(python_results)}, Rust: {len(rust_results)}")
        return
    
    epsilon = 1e-4
    all_pass = True
    
    for i, (p, r) in enumerate(zip(python_results, rust_results)):
        if p["values"] != r["values"]:
            print(f"Vector {i}: Values mismatch! P: {p['values'][:5]}... R: {r['values'][:5]}...")
            all_pass = False
        
        if p["signs"] is not None and r["signs"] is not None:
            if p["signs"] != r["signs"]:
                print(f"Vector {i}: Signs mismatch!")
                all_pass = False
            
        if p["scale"] is not None and r["scale"] is not None:
            if abs(p["scale"] - r["scale"]) > epsilon:
                print(f"Vector {i}: Scale mismatch! P: {p['scale']}, R: {r['scale']}")
                all_pass = False
            
        p_hat = np.array(p["x_hat"])
        r_hat = np.array(r["x_hat"])
        diff = np.linalg.norm(p_hat - r_hat)
        if diff > epsilon:
            print(f"Vector {i}: x_hat reconstruction mismatch! Norm: {diff}")
            all_pass = False
            
        if abs(p["score"] - r["score"]) > epsilon:
            print(f"Vector {i}: Score mismatch! P: {p['score']}, R: {r['score']}")
            all_pass = False
    
    if all_pass:
        print("✅ ALL TESTS PASSED: Python and Rust are numerically consistent!")
    else:
        print("❌ CONSISTENCY CHECK FAILED: Discrepancies found.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python verify_consistency.py [setup|verify]")
        sys.exit(1)
        
    mode = sys.argv[1]
    if mode == "setup":
        generate_test_vectors()
        quantizer = QuantizerRegistry.get_quantizer("EDEN", dim=1024, bits=3, seed=42)
        save_quantizer_state(quantizer)
        run_python_reference()
    elif mode == "verify":
        verify_consistency()
    else:
        print("Invalid mode. Use 'setup' or 'verify'.")


