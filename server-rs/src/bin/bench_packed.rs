use std::time::Instant;
use std::fs::File;
use std::io::Read;
use npyz::NpyFile;
use turbo_memory_rs::quantizers::EdenQuantizer;
use turbo_memory_rs::traits::Quantizer;
use wide::f32x8;

/// On-the-fly packed direct SIMD scoring.
/// Reads 3 bytes → extracts 8× 3-bit indices → gathers centroids → SIMD accumulate.
/// NO intermediate Vec<u8> allocation (unlike score_eden_simd which calls unpack_3bit).
fn score_eden_packed_direct(
    centroids: &[f32; 8],
    q_rot: &[f32],
    packed: &[u8],
    dim: usize,
) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;

    for chunk in packed.chunks(3) {
        if i + 8 > dim || chunk.len() < 3 {
            break;
        }
        let b0 = chunk[0];
        let b1 = chunk[1];
        let b2 = chunk[2];

        // 3 bytes → 8× 3-bit indices (LSB-first, matching unpack_3bit)
        let i0 = (b0 & 7) as usize;
        let i1 = ((b0 >> 3) & 7) as usize;
        let i2 = (((b0 >> 6) & 3) | ((b1 & 1) << 2)) as usize;
        let i3 = ((b1 >> 1) & 7) as usize;
        let i4 = ((b1 >> 4) & 7) as usize;
        let i5 = (((b1 >> 7) & 1) | ((b2 & 3) << 1)) as usize;
        let i6 = ((b2 >> 2) & 7) as usize;
        let i7 = (b2 >> 5) as usize;

        let c_vals = f32x8::new([
            centroids[i0], centroids[i1], centroids[i2], centroids[i3],
            centroids[i4], centroids[i5],             centroids[i6], centroids[i7],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        let qr = f32x8::new(qr_arr);
        acc = acc + c_vals * qr;

        i += 8;
    }

    // remainder (unlikely for dim=1024 which is 128×8 exactly)
    let mut s1 = acc.reduce_add();
    while i < dim {
        let byte_i = i * 3 / 8;
        let bit_offs = (i * 3) % 8;
        if byte_i >= packed.len() {
            break;
        }
        let val = if bit_offs <= 5 {
            (packed[byte_i] >> bit_offs) & 7
        } else {
            let low = packed[byte_i] >> bit_offs;
            let high = if byte_i + 1 < packed.len() {
                packed[byte_i + 1] & ((1 << (3 - (8 - bit_offs))) - 1)
            } else {
                0
            };
            low | (high << (8 - bit_offs))
        };
        s1 += centroids[val as usize] * q_rot[i];
        i += 1;
    }

    s1
}

/// Current approach: unpack_3bit first, then SIMD
fn score_eden_unpack_then_simd(
    centroids: &[f32; 8],
    q_rot: &[f32],
    packed: &[u8],
    dim: usize,
) -> f32 {
    let indices = turbo_memory_rs::eden::unpack_3bit(packed, dim);
    let mut acc = f32x8::ZERO;
    let mut i = 0;

    while i + 8 <= dim {
        let c_vals = f32x8::new([
            centroids[indices[i] as usize],
            centroids[indices[i + 1] as usize],
            centroids[indices[i + 2] as usize],
            centroids[indices[i + 3] as usize],
            centroids[indices[i + 4] as usize],
            centroids[indices[i + 5] as usize],
            centroids[indices[i + 6] as usize],
            centroids[indices[i + 7] as usize],
        ]);
        let qr_arr: [f32; 8] = q_rot[i..i + 8].try_into().unwrap();
        let qr = f32x8::new(qr_arr);
        acc = acc + c_vals * qr;
        i += 8;
    }

    let mut s1 = acc.reduce_add();
    for j in i..dim {
        s1 += centroids[indices[j] as usize] * q_rot[j];
    }

    s1
}

fn pack_quantized_indices(values: &[i32], dim: usize) -> Vec<u8> {
    let packed_sz = (dim * 3 + 7) / 8;
    let mut packed = vec![0u8; packed_sz];
    for i in 0..dim {
        let val = values[i] as u16;
        for b in 0..3 {
            if (val >> b) & 1 != 0 {
                let bit_pos = i * 3 + b;
                packed[bit_pos >> 3] |= 1 << (bit_pos & 7);
            }
        }
    }
    packed
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state_path = "/Users/hodorii/dev/turboquant/quantizer_state_full.json";
    let npy_path = "/Users/hodorii/dev/turboquant/data/real_vectors.npy";

    println!("Loading EdenQuantizer...");
    let quantizer = EdenQuantizer::from_json(state_path)?;
    let dim = quantizer.dim;
    println!("Dim: {}, Bits: {}", dim, quantizer.bits);

    let centroids: [f32; 8] = {
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(&quantizer.codebook[..8]);
        arr
    };

    // Load vectors
    let mut npy_file = File::open(npy_path)?;
    let npy = NpyFile::new(&mut npy_file)?;
    let shape = npy.shape();
    let num_vecs = shape[1] as usize;
    drop(npy);

    let mut data = Vec::new();
    npy_file.read_to_end(&mut data)?;
    let f32_data: Vec<f32> = data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let vectors: Vec<Vec<f32>> = f32_data.chunks_exact(dim)
        .map(|chunk| chunk.to_vec())
        .collect();
    println!("Loaded {} vectors.", vectors.len());

    // Quantize all vectors
    println!("Quantizing {} vectors...", vectors.len());
    let quant_start = Instant::now();
    let mut packed_vecs = Vec::with_capacity(vectors.len());
    for v in &vectors {
        let q = quantizer.quantize(v);
        packed_vecs.push(pack_quantized_indices(&q.values, dim));
    }
    let quant_time = quant_start.elapsed();
    println!("Quantize: {:.3} ms total, {:.1} µs/vec",
        quant_time.as_secs_f64() * 1000.0,
        quant_time.as_secs_f64() * 1_000_000.0 / vectors.len() as f64);

    // Pre-rotate query (using SIMD rotation matching rotate())
    let query = &vectors[0];
    let mut y_query = vec![0.0f32; dim];
    {
        let rotation = &quantizer.rotation;
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

    // Verify both approaches produce identical results
    println!("\nVerifying correctness on first 5 vectors...");
    for idx in 0..5.min(vectors.len()) {
        let s1 = score_eden_unpack_then_simd(&centroids, &y_query, &packed_vecs[idx], dim);
        let s2 = score_eden_packed_direct(&centroids, &y_query, &packed_vecs[idx], dim);
        let diff = (s1 - s2).abs();
        println!("  vec[{}]: unpack={:.8}, direct={:.8}, diff={:.2e}", idx, s1, s2, diff);
        assert!(diff < 1e-6, "Mismatch at vec[{}]: {:.8} vs {:.8}", idx, s1, s2);
    }
    println!("  ✓ All match");

    // Warmup
    for _ in 0..10 {
        for pv in &packed_vecs {
            let _ = score_eden_unpack_then_simd(&centroids, &y_query, pv, dim);
            let _ = score_eden_packed_direct(&centroids, &y_query, pv, dim);
        }
    }

    // ── Accuracy: compare top-k between unpack vs packed_direct ──
    let k = 5;
    // Scores from unpack approach (single pass, store all)
    let scores_unpack: Vec<f32> = packed_vecs.iter()
        .map(|pv| score_eden_unpack_then_simd(&centroids, &y_query, pv, dim))
        .collect();
    let scores_direct: Vec<f32> = packed_vecs.iter()
        .map(|pv| score_eden_packed_direct(&centroids, &y_query, pv, dim))
        .collect();

    let max_diff: f32 = scores_unpack.iter()
        .zip(scores_direct.iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(f32::max)
        .unwrap_or(0.0);
    println!("\nAccuracy: max score diff between unpack vs direct = {:.2e}", max_diff);

    // Top-k
    let mut indices_unpack: Vec<usize> = (0..scores_unpack.len()).collect();
    indices_unpack.sort_by(|&a, &b| scores_unpack[b].partial_cmp(&scores_unpack[a]).unwrap());
    let mut indices_direct: Vec<usize> = (0..scores_direct.len()).collect();
    indices_direct.sort_by(|&a, &b| scores_direct[b].partial_cmp(&scores_direct[a]).unwrap());

    let topk_unpack: Vec<usize> = indices_unpack[..k].to_vec();
    let topk_direct: Vec<usize> = indices_direct[..k].to_vec();

    let overlap: usize = topk_unpack.iter().filter(|x| topk_direct.contains(x)).count();
    let rank_corr: f64 = topk_unpack.iter().zip(topk_direct.iter())
        .filter(|(a, b)| a == b).count() as f64 / k as f64;

    println!("Top-{} comparison:", k);
    println!("  Overlap: {}/{}", overlap, k);
    println!("  Exact rank match: {}/{} ({:.2}%)", (rank_corr * k as f64) as usize, k, rank_corr * 100.0);
    println!("  Unpack top-{}: {:?}", k, topk_unpack);
    println!("  Direct top-{}: {:?}", k, topk_direct);

    // ── Benchmark: unpack_then_simd (current approach) ──
    const ITERS: u32 = 100;
    let start = Instant::now();
    for _ in 0..ITERS {
        for pv in &packed_vecs {
            let _ = score_eden_unpack_then_simd(&centroids, &y_query, pv, dim);
        }
    }
    let dur_unpack = start.elapsed();
    let per_vec_unpack = dur_unpack.as_secs_f64() / (ITERS as f64 * vectors.len() as f64);

    // ── Benchmark: packed_direct (new approach) ──
    let start = Instant::now();
    for _ in 0..ITERS {
        for pv in &packed_vecs {
            let _ = score_eden_packed_direct(&centroids, &y_query, pv, dim);
        }
    }
    let dur_direct = start.elapsed();
    let per_vec_direct = dur_direct.as_secs_f64() / (ITERS as f64 * vectors.len() as f64);

    // ── Results ──
    println!("\n========== PACKED DIRECT COMPARISON BENCHMARK ==========");
    println!("Vectors: {} | Dim: {} | Bits: 3 | Iterations: {}", vectors.len(), dim, ITERS);
    println!("---------------------------------------------------------");
    println!("Approach                    | per-vector  | vs FAISS (est)");
    println!("---------------------------------------------------------");
    println!("unpack_3bit + SIMD (current)| {:>8.3} ns | {:>8.3} ms/1kvec",
        per_vec_unpack * 1e9, per_vec_unpack * vectors.len() as f64 * 1000.0);
    println!("packed direct SIMD (new)    | {:>8.3} ns | {:>8.3} ms/1kvec",
        per_vec_direct * 1e9, per_vec_direct * vectors.len() as f64 * 1000.0);
    println!("---------------------------------------------------------");
    println!("Speedup: {:.2}x", per_vec_unpack / per_vec_direct);
    println!("FAISS est: ~0.25 µs/vec (0.25 ms / 1000 vec)");
    println!("=========================================================");

    // Storage comparison
    let packed_bytes_per = (dim * 3 + 7) / 8;
    let float32_bytes = dim * 4;
    println!("\nStorage per vector:");
    println!("  float32:       {} B", float32_bytes);
    println!("  packed 3-bit:  {} B ({:.1}x compression)", packed_bytes_per, float32_bytes as f64 / packed_bytes_per as f64);

    Ok(())
}
