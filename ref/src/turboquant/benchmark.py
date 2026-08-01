import time
import numpy as np
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from .registry import QuantizerRegistry
from .quantizers import DriveV3Quantizer, QJLQuantizer
from .interfaces import BaseQuantizer, QuantizedResult

@dataclass
class AlgoMetrics:
    algo_id: str
    bits: int
    vnmse: float
    recall_at_1: float
    recall_at_10: float
    latency_ns_per_op: float
    memory_bytes_per_vec: float
    throughput_vec_per_sec: float

class BenchmarkingHarness:
    def __init__(self, dim: int = 1024, n_vectors: int = 1000):
        self.dim = dim
        self.n_vectors = n_vectors
        # Use real data if available
        try:
            self.data = np.load('data/real_vectors.npy')[:n_vectors]
        except:
            self.data = np.random.randn(n_vectors, dim).astype(np.float32)
        
        # Normalize
        self.data /= np.linalg.norm(self.data, axis=1, keepdims=True)

    def _calculate_vnmse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        mse = np.mean(np.sum((original - reconstructed)**2, axis=1) / np.sum(original**2, axis=1))
        return float(mse)

    def _calculate_recall(self, original: np.ndarray, query: np.ndarray, 
                          quantizer: BaseQuantizer, quantized_data: List[QuantizedResult], k: int) -> float:
        gt_scores = np.dot(original, query)
        gt_top_k = np.argsort(gt_scores)[-k:]

        q_scores = np.array([quantizer.calculate_score(query, q_res) for q_res in quantized_data])
        q_top_k = np.argsort(q_scores)[-k:]

        intersection = np.intersect1d(gt_top_k, q_top_k)
        return float(len(intersection)) / k

    def run_benchmark(self, query_vec: np.ndarray, bits: int) -> List[AlgoMetrics]:
        results = []
        algos = QuantizerRegistry.list_available()
        
        print(f"🚀 Benchmarking: bits={bits}, dim={self.dim}, n_vectors={self.n_vectors}")
        
        for algo_id in algos:
            quantizer = QuantizerRegistry.get_quantizer(algo_id, dim=self.dim, bits=bits)
            
            start_q = time.perf_counter_ns()
            quantized_data = [quantizer.quantize(vec) for vec in self.data]
            end_q = time.perf_counter_ns()
            q_latency = (end_q - start_q) / self.n_vectors
            
            mem_bytes = 0
            if quantized_data:
                res = quantized_data[0]
                mem_bytes += res.values.nbytes
                if res.signs is not None: mem_bytes += res.signs.nbytes
                if res.scale is not None: mem_bytes += 4
            
            reconstructed = np.array([quantizer.decode(q) for q in quantized_data])
            vnmse = self._calculate_vnmse(self.data, reconstructed)

            start_s = time.perf_counter_ns()
            _ = [quantizer.calculate_score(query_vec, q) for q in quantized_data]
            end_s = time.perf_counter_ns()
            
            s_latency = (end_s - start_s) / self.n_vectors
            recall_1 = self._calculate_recall(self.data, query_vec, quantizer, quantized_data, 1)
            recall_10 = self._calculate_recall(self.data, query_vec, quantizer, quantized_data, 10)
            throughput = 1e9 / s_latency

            results.append(AlgoMetrics(
                algo_id=algo_id,
                bits=bits,
                vnmse=vnmse,
                recall_at_1=recall_1,
                recall_at_10=recall_10,
                latency_ns_per_op=s_latency,
                memory_bytes_per_vec=float(mem_bytes),
                throughput_vec_per_sec=throughput
            ))
            print(f"✅ {algo_id} ({bits}bit) completed: vNMSE={vnmse:.4f}, Recall@1={recall_1:.2%}")

        return results

    def print_report(self, all_metrics: List[AlgoMetrics]):
        print("\n" + "="*100)
        print(f"{'Algorithm':<15} | {'Bits':<6} | {'vNMSE':<10} | {'Recall@1':<10} | {'Latency':<12} | {'Mem(B)':<8}")
        print("-" * 100)
        for m in all_metrics:
            print(f"{m.algo_id:<15} | {m.bits:<6} | {m.vnmse:<10.4f} | {m.recall_at_1:<10.2%} | {m.latency_ns_per_op:<12.1f} | {m.memory_bytes_per_vec:<8.1f}")
        print("="*100)

if __name__ == "__main__":
    np.random.seed(42) # Fix seed for reproducibility
    harness = BenchmarkingHarness(dim=1024, n_vectors=1000)
    query = np.random.randn(1024).astype(np.float32)
    query /= np.linalg.norm(query)
    
    all_results = []
    for b in [2, 3, 4]:
        all_results.extend(harness.run_benchmark(query, b))
    
    harness.print_report(all_results)

