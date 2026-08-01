use std::time::Instant;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use rusqlite::{Connection, params};
use npyz::NpyFile;
use turbo_memory_rs::quantizers::EdenQuantizer;
use turbo_memory_rs::traits::{Quantizer, QuantizedResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state_path = "/Users/hodorii/dev/turboquant/quantizer_state_full.json";
    let npy_path = "/Users/hodorii/dev/turboquant/data/real_vectors.npy";
    let db_path = "/Users/hodorii/dev/turboquant/session_memory.db";

    println!("Loading EdenQuantizer from {}...", state_path);
    let quantizer = EdenQuantizer::from_json(state_path)?;
    println!("Quantizer loaded. Dim: {}, Bits: {}", quantizer.dim, quantizer.bits);

    let mut npy_file = File::open(npy_path)?;
    let npy = NpyFile::new(&mut npy_file)?;
    let shape = npy.shape();
    let dim = shape[0] as usize;
    let num_vecs = shape[1] as usize;
    
    let mut data = Vec::new();
    npy_file.read_to_end(&mut data)?;
    
    let f32_data: Vec<f32> = data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    
    let sillok_vectors: Vec<Vec<f32>> = f32_data.chunks_exact(quantizer.dim)
        .map(|chunk| chunk.to_vec())
        .collect();
    println!("Loaded {} Sillok vectors.", sillok_vectors.len());

    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare("SELECT embedding FROM entries")?;
    let session_vectors: Vec<Vec<f32>> = stmt.query_map([], |row| {
        let blob: Vec<u8> = row.get(0)?;
        let f32s: Vec<f32> = blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(f32s)
    })?.filter_map(|res| res.ok()).collect();
    println!("Loaded {} Session vectors.", session_vectors.len());

    let all_vectors = [("Sillok", sillok_vectors), ("Session", session_vectors)];

    for (name, vectors) in all_vectors {
        if vectors.is_empty() { continue; }
        
        let filtered_vectors: Vec<&Vec<f32>> = vectors.iter()
            .filter(|v| v.len() == quantizer.dim)
            .collect();
        
        if filtered_vectors.is_empty() {
            println!("\n--- Skipping {} dataset: no vectors match dimension {} ---", name, quantizer.dim);
            continue;
        }

        println!("\n--- Benchmarking {} dataset ({} vectors) ---", name, filtered_vectors.len());

        let start = Instant::now();
        let mut results = Vec::with_capacity(filtered_vectors.len());
        for v in &filtered_vectors {
            results.push(quantizer.quantize(v));
        }
        let duration = start.elapsed();
        let avg_q = duration.as_nanos() as f64 / filtered_vectors.len() as f64;
        println!("Quantization: Avg {:.2} ns/vec ({:.2} Mvecs/sec)", avg_q, 1e9 / avg_q);

        let start = Instant::now();
        for q in &results {
            let _ = quantizer.decode(q);
        }
        let duration = start.elapsed();
        let avg_d = duration.as_nanos() as f64 / filtered_vectors.len() as f64;
        println!("Decoding:     Avg {:.2} ns/vec ({:.2} Mvecs/sec)", avg_d, 1e9 / avg_d);

        let query = filtered_vectors[0];
        let mut y_query = vec![0.0f32; quantizer.dim];
        {
            let rotation = &quantizer.rotation;
            let dim = quantizer.dim;
            for i in 0..dim {
                use wide::f32x8;
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
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            for q in &results {
                let mut score_acc = 0.0f64;
                let scale = q.scale.unwrap_or(1.0) as f64;
                for i in 0..quantizer.dim {
                    let q_val = quantizer.codebook[q.values[i] as usize];
                    score_acc += (y_query[i] as f64) * (q_val as f64);
                }
                let _score = (score_acc * scale) as f32;
            }
        }
        let duration = start.elapsed();
        let avg_s = duration.as_nanos() as f64 / (filtered_vectors.len() as f64 * iterations as f64);
        println!("Scoring:      Avg {:.2} ns/op ({:.2} Mops/sec)", avg_s, 1e9 / avg_s);
    }

    Ok(())
}
