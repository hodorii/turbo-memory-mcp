use std::fs::File;
use std::io::Read;
use npyz::NpyFile;
use turbo_memory_rs::quantizers::EdenQuantizer;
use turbo_memory_rs::traits::Quantizer;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bits = 3;
    let state_path = format!("/Users/hodorii/dev/turboquant/quantizer_state_{}bit.json", bits);
    let quantizer = EdenQuantizer::from_json(&state_path)?;

    println!("=== Rust {}‑bit Diagnostic ===", bits);
    println!("dim: {}", quantizer.dim);
    println!("codebook ({}): {:?}", quantizer.codebook.len(),
        quantizer.codebook.iter().map(|c| format!("{:.6}", c)).collect::<Vec<_>>());

    println!("rotation[0][0..5]: {:?}", &quantizer.rotation[0..5]);
    println!("rotation[1][0..5]: {:?}", &quantizer.rotation[1024..1029]);

    // Load vec[0] (same npy as benchmark)
    let npy_path = "/Users/hodorii/dev/turboquant/data/real_vectors.npy";
    let mut npy_file = File::open(npy_path)?;
    let npy = NpyFile::new(&mut npy_file)?;
    let mut data = Vec::new();
    npy_file.read_to_end(&mut data)?;
    let f32_data: Vec<f32> = data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect();
    let all_vectors: Vec<Vec<f32>> = f32_data.chunks_exact(1024)
        .map(|chunk| chunk.to_vec()).collect();
    let x0 = &all_vectors[0];
    println!("\nx0[0..5]: {:?}", &x0[0..5]);

    // Compute y = x @ R using naive double loop (same math as rotate())
    let dim = quantizer.dim;
    let rotation = &quantizer.rotation;
    let mut y = vec![0.0f32; dim];
    for i in 0..dim {
        for j in 0..dim {
            y[j] += x0[i] * rotation[i * dim + j];
        }
    }
    println!("\nrotated[0..20]: {:?}", &y[0..20]);
    println!("rotated[0..20] (rounded 6dp): {:?}",
        y[0..20].iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>());

    // Quantize
    let q = quantizer.quantize(x0);
    println!("\nindices[0..24]: {:?}", &q.values[0..24]);
    println!("scale (S): {:?}", q.scale);

    // Manual nearest-codebook check for first 10 rotated values
    println!("\nNearest‑codebook check (first 10):");
    for i in 0..10 {
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;
        for (idx, &val) in quantizer.codebook.iter().enumerate() {
            let dist = (y[i] - val).abs();
            if dist < min_dist {
                min_dist = dist;
                best_idx = idx;
            }
        }
        println!("  y[{}]={:.6} → codebook[{}]={:.6} (dist={:.6})", i, y[i], best_idx, quantizer.codebook[best_idx], min_dist);
    }

    Ok(())
}
