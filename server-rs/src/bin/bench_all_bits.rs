use std::time::Instant;
use std::fs::File;
use std::io::Read;
use npyz::NpyFile;
use turbo_memory_rs::quantizers::EdenQuantizer;
use turbo_memory_rs::traits::Quantizer;
use wide::f32x8;

// ── 2-bit score functions ──────────────────────────────────────────────

fn score_2bit_bytewise(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    for chunk in packed.chunks(2) {
        if i + 8 > dim || chunk.len() < 2 { break; }
        let b0 = chunk[0]; let b1 = chunk[1];
        let idx = [
            (b0 & 3) as usize, ((b0 >> 2) & 3) as usize,
            ((b0 >> 4) & 3) as usize, ((b0 >> 6) & 3) as usize,
            (b1 & 3) as usize, ((b1 >> 2) & 3) as usize,
            ((b1 >> 4) & 3) as usize, ((b1 >> 6) & 3) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 4; let shift = (i % 4) * 2;
        dot += centroids[((packed[byte_i] >> shift) & 3) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

fn score_2bit_uint(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    let mut off = 0;
    while i + 8 <= dim {
        let val = u16::from_le_bytes(packed[off..off + 2].try_into().unwrap()) as u32;
        let idx = [
            ((val >> 0) & 3) as usize, ((val >> 2) & 3) as usize,
            ((val >> 4) & 3) as usize, ((val >> 6) & 3) as usize,
            ((val >> 8) & 3) as usize, ((val >> 10) & 3) as usize,
            ((val >> 12) & 3) as usize, ((val >> 14) & 3) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8; off += 2;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 4; let shift = (i % 4) * 2;
        dot += centroids[((packed[byte_i] >> shift) & 3) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

// ── 3-bit score functions ──────────────────────────────────────────────

fn score_3bit_bytewise(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    for chunk in packed.chunks(3) {
        if i + 8 > dim || chunk.len() < 3 { break; }
        let b0 = chunk[0]; let b1 = chunk[1]; let b2 = chunk[2];
        let idx = [
            (b0 & 7) as usize, ((b0 >> 3) & 7) as usize,
            (((b0 >> 6) & 3) | ((b1 & 1) << 2)) as usize,
            ((b1 >> 1) & 7) as usize, ((b1 >> 4) & 7) as usize,
            (((b1 >> 7) & 1) | ((b2 & 3) << 1)) as usize,
            ((b2 >> 2) & 7) as usize, (b2 >> 5) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i * 3 / 8; let bit_offs = (i * 3) % 8;
        if byte_i >= packed.len() { break; }
        let val = if bit_offs <= 5 { (packed[byte_i] >> bit_offs) & 7 }
                  else { (packed[byte_i] >> bit_offs) | ((packed[byte_i + 1] & ((1 << (3 - (8 - bit_offs))) - 1)) << (8 - bit_offs)) };
        dot += centroids[(val & 7) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

fn score_3bit_uint(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    let mut off = 0;
    while i + 8 <= dim {
        let raw = u32::from_le_bytes([
            packed[off], packed[off + 1], packed[off + 2], 0,
        ]);
        let idx = [
            ((raw >> 0) & 7) as usize, ((raw >> 3) & 7) as usize,
            ((raw >> 6) & 7) as usize, ((raw >> 9) & 7) as usize,
            ((raw >> 12) & 7) as usize, ((raw >> 15) & 7) as usize,
            ((raw >> 18) & 7) as usize, ((raw >> 21) & 7) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8; off += 3;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i * 3 / 8; let bit_offs = (i * 3) % 8;
        if byte_i >= packed.len() { break; }
        let val = if bit_offs <= 5 { (packed[byte_i] >> bit_offs) & 7 }
                  else { (packed[byte_i] >> bit_offs) | ((packed[byte_i + 1] & ((1 << (3 - (8 - bit_offs))) - 1)) << (8 - bit_offs)) };
        dot += centroids[(val & 7) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

// ── 4-bit score functions ──────────────────────────────────────────────

fn score_4bit_bytewise(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    for chunk in packed.chunks(4) {
        if i + 8 > dim || chunk.len() < 4 { break; }
        let b0 = chunk[0]; let b1 = chunk[1]; let b2 = chunk[2]; let b3 = chunk[3];
        let idx = [
            (b0 & 15) as usize, (b0 >> 4) as usize,
            (b1 & 15) as usize, (b1 >> 4) as usize,
            (b2 & 15) as usize, (b2 >> 4) as usize,
            (b3 & 15) as usize, (b3 >> 4) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 2; let shift = (i % 2) * 4;
        dot += centroids[((packed[byte_i] >> shift) & 15) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

fn score_4bit_uint(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    let mut off = 0;
    while i + 8 <= dim {
        let val = u32::from_le_bytes(packed[off..off + 4].try_into().unwrap());
        let idx = [
            ((val >> 0) & 15) as usize, ((val >> 4) & 15) as usize,
            ((val >> 8) & 15) as usize, ((val >> 12) & 15) as usize,
            ((val >> 16) & 15) as usize, ((val >> 20) & 15) as usize,
            ((val >> 24) & 15) as usize, ((val >> 28) & 15) as usize,
        ];
        let c_vals = f32x8::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
            centroids[idx[4]], centroids[idx[5]], centroids[idx[6]], centroids[idx[7]],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        acc = acc + c_vals * f32x8::new(qr_arr);
        i += 8; off += 4;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 2; let shift = (i % 2) * 4;
        dot += centroids[((packed[byte_i] >> shift) & 15) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

// ── f32x4 (128-bit SIMD) variants ─────────────────────────────────────

fn score_2bit_x4(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    use wide::f32x4;
    let mut acc = f32x4::ZERO;
    let mut i = 0;
    for &b in packed.iter() {
        if i + 4 > dim { break; }
        let idx = [
            (b & 3) as usize, ((b >> 2) & 3) as usize,
            ((b >> 4) & 3) as usize, ((b >> 6) & 3) as usize,
        ];
        let c_vals = f32x4::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
        ]);
        let qr: [f32; 4] = q_rot[i..i + 4].try_into().unwrap();
        acc = acc + c_vals * f32x4::new(qr);
        i += 4;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 4; let shift = (i % 4) * 2;
        dot += centroids[((packed[byte_i] >> shift) & 3) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

fn score_3bit_x4(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    use wide::f32x4;
    let mut acc = f32x4::ZERO;
    let mut i = 0;
    let mut off = 0;
    while i + 4 <= dim && off + 2 <= packed.len() {
        let val = u16::from_le_bytes(packed[off..off + 2].try_into().unwrap());
        let idx = [
            (val & 7) as usize,
            ((val >> 3) & 7) as usize,
            ((val >> 6) & 7) as usize,
            ((val >> 9) & 7) as usize,
        ];
        let c_vals = f32x4::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
        ]);
        let qr: [f32; 4] = q_rot[i..i + 4].try_into().unwrap();
        acc = acc + c_vals * f32x4::new(qr);
        i += 4; off += 2;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i * 3 / 8; let bit_offs = (i * 3) % 8;
        if byte_i >= packed.len() { break; }
        let val = if bit_offs <= 5 { (packed[byte_i] >> bit_offs) & 7 }
                  else { (packed[byte_i] >> bit_offs) | ((packed[byte_i + 1] & ((1 << (3 - (8 - bit_offs))) - 1)) << (8 - bit_offs)) };
        dot += centroids[(val & 7) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

fn score_4bit_x4(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
    use wide::f32x4;
    let mut acc = f32x4::ZERO;
    let mut i = 0;
    for chunk in packed.chunks(2) {
        if i + 4 > dim || chunk.len() < 2 { break; }
        let idx = [
            (chunk[0] & 15) as usize, (chunk[0] >> 4) as usize,
            (chunk[1] & 15) as usize, (chunk[1] >> 4) as usize,
        ];
        let c_vals = f32x4::new([
            centroids[idx[0]], centroids[idx[1]], centroids[idx[2]], centroids[idx[3]],
        ]);
        let qr: [f32; 4] = q_rot[i..i + 4].try_into().unwrap();
        acc = acc + c_vals * f32x4::new(qr);
        i += 4;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        let byte_i = i / 2; let shift = (i % 2) * 4;
        dot += centroids[((packed[byte_i] >> shift) & 15) as usize] * q_rot[i];
        i += 1;
    }
    dot * s
}

// ── Unified scorer ─────────────────────────────────────────────────────
//
// Runtime bit-width dispatch: picks the fastest method per bit-width on M1.
//   - 2-bit: uint16 extraction
//   - 3-bit: bytewise extraction
//   - 4-bit: uint32 extraction

fn score_quantized(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, bits: u32, s: f32) -> f32 {
    match bits {
        2 => score_2bit_uint(centroids, q_rot, packed, dim, s),
        3 => score_3bit_bytewise(centroids, q_rot, packed, dim, s),
        4 => score_4bit_uint(centroids, q_rot, packed, dim, s),
        _ => panic!("score_quantized: unsupported bit-width {bits}"),
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn pack_indices(values: &[i32], dim: usize, bits: u32) -> Vec<u8> {
    let packed_sz = (dim * bits as usize + 7) / 8;
    let mut packed = vec![0u8; packed_sz];
    for i in 0..dim {
        let val = values[i] as u16;
        for b in 0..bits {
            if (val >> b) & 1 != 0 {
                let bit_pos = i * bits as usize + b as usize;
                packed[bit_pos >> 3] |= 1 << (bit_pos & 7);
            }
        }
    }
    packed
}

fn compute_ground_truth(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let scores: Vec<f32> = vectors.iter()
        .map(|v| query.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    indices[..k].to_vec()
}

fn recall_at_k(pred: &[usize], truth: &[usize]) -> f64 {
    pred.iter().filter(|x| truth.contains(x)).count() as f64 / truth.len() as f64
}

fn topk_from_scores(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    indices[..k].to_vec()
}

struct BenchResult {
    bits: u32,
    method: String,
    ns_per_vec: f64,
    ms_per_1k: f64,
    recall_5: f64,
    rank_corr: f64,
    packed_bytes: usize,
}

fn run_bench(
    bits: u32,
    packed_vecs: &[Vec<u8>],
    scales: &[f32],
    ground_truth: &[usize],
    method: &str,
    score_fn: impl Fn(&[u8], f32) -> f32,
) -> BenchResult {
    const ITERS: u32 = 100;
    let k = 5;

    // Warmup
    for _ in 0..10 {
        for (pv, &s) in packed_vecs.iter().zip(scales.iter()) {
            let _ = score_fn(pv, s);
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERS {
        for (pv, &s) in packed_vecs.iter().zip(scales.iter()) {
            let _ = score_fn(pv, s);
        }
    }
    let dur = start.elapsed();
    let ns_per_vec = dur.as_secs_f64() / (ITERS as f64 * packed_vecs.len() as f64) * 1e9;

    // Top-k
    let scores: Vec<f32> = packed_vecs.iter().zip(scales.iter())
        .map(|(pv, &s)| score_fn(pv, s))
        .collect();
    let topk = topk_from_scores(&scores, k);
    let recall = recall_at_k(&topk, ground_truth);
    let rank_corr = topk.iter().zip(ground_truth.iter())
        .filter(|(a, b)| a == b).count() as f64 / k as f64;

    let packed_bytes = if packed_vecs.is_empty() { 0 } else { packed_vecs[0].len() };

    BenchResult {
        bits,
        method: method.to_string(),
        ns_per_vec,
        ms_per_1k: ns_per_vec * packed_vecs.len() as f64 / 1_000_000.0,
        recall_5: recall,
        rank_corr,
        packed_bytes,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let npy_path = "/Users/hodorii/dev/turboquant/data/real_vectors.npy";

    // Load vectors once (shared across all bit widths)
    let mut npy_file = File::open(npy_path)?;
    let npy = NpyFile::new(&mut npy_file)?;
    let shape = npy.shape();
    let num_vecs = shape[1] as usize;
    drop(npy);
    let mut data = Vec::new();
    npy_file.read_to_end(&mut data)?;
    let f32_data: Vec<f32> = data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect();
    let all_vectors: Vec<Vec<f32>> = f32_data.chunks_exact(1024)
        .map(|chunk| chunk.to_vec()).collect();
    println!("Loaded {} vectors (1024-dim)\n", all_vectors.len());

    let query = &all_vectors[0];
    let ground_truth = compute_ground_truth(query, &all_vectors, 5);
    println!("Ground truth for query[0] (float32 dot, top-5): {:?}\n", ground_truth);

    // Pre-compute ground truth for ALL queries (needed for full recall eval)
    let all_ground_truths: Vec<Vec<usize>> = all_vectors.iter()
        .map(|q| compute_ground_truth(q, &all_vectors, 5))
        .collect();

    let bit_configs = [2u32, 3u32, 4u32];
    let mut all_results: Vec<BenchResult> = Vec::new();
    let mut full_recall_means: Vec<(u32, f64)> = Vec::new();

    for &bits in &bit_configs {
        println!("─── {}─bit ───", bits);
        let state_path = format!("/Users/hodorii/dev/turboquant/quantizer_state_{}bit.json", bits);
        let quantizer = EdenQuantizer::from_json(&state_path)?;

        let centroids: Vec<f32> = quantizer.codebook.clone();
        let n_centroids = 1 << bits;
        assert_eq!(centroids.len(), n_centroids);
        println!("  centroids ({}): {:?}", n_centroids,
            centroids.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>().join(", "));

        // Quantize all vectors
        let quant_start = Instant::now();
        let mut packed_vecs: Vec<Vec<u8>> = Vec::with_capacity(all_vectors.len());
        let mut scales: Vec<f32> = Vec::with_capacity(all_vectors.len());
        for v in &all_vectors {
            let q = quantizer.quantize(v);
            packed_vecs.push(pack_indices(&q.values, quantizer.dim, bits));
            scales.push(q.scale.unwrap_or(1.0));
        }
        let quant_ms = quant_start.elapsed().as_secs_f64() * 1000.0;
        println!("  quantize: {:.1}ms ({:.1}µs/vec)", quant_ms, quant_ms * 1000.0 / all_vectors.len() as f64);
        println!("  S range: [{:.4}, {:.4}]", scales.iter().fold(f32::MAX, |a, &b| a.min(b)), scales.iter().fold(f32::MIN, |a, &b| a.max(b)));

        // Pre-rotate query
        let mut y_query = vec![0.0f32; quantizer.dim];
        {
            let rotation = &quantizer.rotation;
            let dim = quantizer.dim;
            for i in 0..dim {
                let xi = f32x8::splat(query[i]);
                let row_offset = i * dim;
                let row = &rotation[row_offset..row_offset + dim];
                for j in (0..dim).step_by(8) {
                    let r_vec = f32x8::new(row[j..j + 8].try_into().unwrap());
                    let mut y_vec = f32x8::new(y_query[j..j + 8].try_into().unwrap());
                    y_vec += xi * r_vec;
                    let res: [f32; 8] = y_vec.into();
                    y_query[j..j + 8].copy_from_slice(&res);
                }
            }
        }

        // Benchmark individual methods for cross-method comparison
        let individual_methods: Vec<(&str, fn(&[f32], &[f32], &[u8], usize, f32) -> f32)> = match bits {
            2 => vec![("bytewise", score_2bit_bytewise), ("uint16", score_2bit_uint), ("x4", score_2bit_x4)],
            3 => vec![("bytewise", score_3bit_bytewise), ("uint32", score_3bit_uint), ("x4", score_3bit_x4)],
            4 => vec![("bytewise", score_4bit_bytewise), ("uint32", score_4bit_uint), ("x4", score_4bit_x4)],
            _ => unreachable!(),
        };
        for (method_name, score_fn) in &individual_methods {
            let result = run_bench(bits, &packed_vecs, &scales, &ground_truth, method_name,
                |pv, s| score_fn(&centroids, &y_query, pv, quantizer.dim, s));
            println!("  {:<9} | {:>7.1} ns/vec | {:>7.2} ms/1k | recall@5={:.4} | rank_corr={:.2}",
                result.method, result.ns_per_vec, result.ms_per_1k, result.recall_5, result.rank_corr);
            all_results.push(result);
        }

        // Benchmark unified scorer
        let unified_result = run_bench(bits, &packed_vecs, &scales, &ground_truth, "unified",
            |pv, s| score_quantized(&centroids, &y_query, pv, quantizer.dim, bits, s));
        println!("  {:<9} | {:>7.1} ns/vec | {:>7.2} ms/1k | recall@5={:.4} | rank_corr={:.2}",
            unified_result.method, unified_result.ns_per_vec, unified_result.ms_per_1k,
            unified_result.recall_5, unified_result.rank_corr);
        all_results.push(unified_result);

        // Full-dataset recall: evaluate over ALL queries using unified scorer
        let dim = quantizer.dim;
        let rotation = &quantizer.rotation;
        let mut total_recall = 0.0f64;
        let mut min_recall = 1.0f64;
        let mut max_recall = 0.0f64;
        let n_queries = all_vectors.len().min(1000);
        for qi in 0..n_queries {
            let q = &all_vectors[qi];
            let mut y_q = vec![0.0f32; dim];
            for i in 0..dim {
                let xi = f32x8::splat(q[i]);
                let row_offset = i * dim;
                let row = &rotation[row_offset..row_offset + dim];
                for j in (0..dim).step_by(8) {
                    let r_vec = f32x8::new(row[j..j + 8].try_into().unwrap());
                    let mut yv = f32x8::new(y_q[j..j + 8].try_into().unwrap());
                    yv += xi * r_vec;
                    let res: [f32; 8] = yv.into();
                    y_q[j..j + 8].copy_from_slice(&res);
                }
            }
            let scores: Vec<f32> = packed_vecs.iter().zip(scales.iter())
                .map(|(pv, &s)| score_quantized(&centroids, &y_q, pv, dim, bits, s))
                .collect();
            let topk = topk_from_scores(&scores, 5);
            let recall = recall_at_k(&topk, &all_ground_truths[qi]);
            total_recall += recall;
            if recall < min_recall { min_recall = recall; }
            if recall > max_recall { max_recall = recall; }
        }
        let mean_recall = total_recall / n_queries as f64;
        full_recall_means.push((bits, mean_recall));
        println!("  full recall@5 ({} queries, unified): mean={:.4}, min={:.4}, max={:.4}",
            n_queries, mean_recall, min_recall, max_recall);
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  FINAL COMPARISON (1024-dim, {} vectors, top-5)", all_vectors.len());
    println!("═══════════════════════════════════════════════════════════════════");
    println!("{:<6} {:<10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}", "Bits", "Method", "ns/vec", "ms/1k", "Recall@5", "RkCorr", "Size", "Full R@5");
    println!("{:<6} {:<10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}", "", "", "", "", "", "", "", "");
    for r in &all_results {
        println!("{:<6} {:<10} {:>8.1} {:>10.3} {:>9.4} {:>9.2} {:>6}B {:>10}",
            format!("{}-bit", r.bits), r.method, r.ns_per_vec, r.ms_per_1k, r.recall_5, r.rank_corr, r.packed_bytes, "—");
    }
    println!("{:<6} {:<10} {:>8.1} {:>10.3} {:>9} {:>9} {:>6}B {:>10}", "f32", "exact IP", 0.0, 0.0, "1.0000", "1.00", 4096, "1.0000");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("\nFull recall@5 (mean over all queries, unified scorer):");
    for (bits, mean) in &full_recall_means {
        println!("  {}-bit: mean={:.4}", bits, mean);
    }

    // Compare methods for each bit-width
    println!("\n⸻⸻  method comparison  ⸸⸻");
    for bits in &bit_configs {
        let bw = all_results.iter().find(|r| r.bits == *bits && r.method == "bytewise").unwrap();
        let ui = all_results.iter().find(|r| r.bits == *bits && r.method != "bytewise" && r.method != "x4" && r.method != "unified").unwrap();
        let unif = all_results.iter().find(|r| r.bits == *bits && r.method == "unified").unwrap();
        println!("  {}-bit: uint/bytewise = {:.2}x, unified = {:.1} ns/vec", bits, bw.ns_per_vec / ui.ns_per_vec, unif.ns_per_vec);
    }

    Ok(())
}
