use std::fs::File;
use std::io::{BufReader, Write};
use serde::{Deserialize, Serialize};
use serde_json;

use turbo_memory_rs::quantizers::{DriveV3Quantizer, EdenQuantizer};
use turbo_memory_rs::traits::{Quantizer, QuantizedResult};

#[derive(Deserialize)]
struct QuantizerState {
    algo_id: String,
    dim: usize,
    bits: usize,
    mode: Option<String>,
    rotation: Option<Vec<f32>>,
    codebook: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct VectorResult {
    values: Vec<i32>,
    signs: Option<Vec<i8>>,
    scale: Option<f32>,
    x_hat: Vec<f32>,
    score: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: verify_rust <state_path> <vectors_path> <output_path>");
        std::process::exit(1);
    }
    let state_path = &args[1];
    let vectors_path = &args[2];
    let output_path = &args[3];

    println!("Loading state from {state_path}...");
    let state_file = File::open(state_path)?;
    let state: QuantizerState = serde_json::from_reader(BufReader::new(state_file))?;
    println!("Loaded state: algo_id={}, dim={}, bits={}, rotation_exists={}, codebook_exists={}", 
             state.algo_id, state.dim, state.bits, state.rotation.is_some(), state.codebook.is_some());
    
    let quantizer: Box<dyn Quantizer> = match state.algo_id.as_str() {
        "DriveV3Quantizer" | "DRIVE_V3" => Box::new(DriveV3Quantizer {
            dim: state.dim,
            rotation: state.rotation.expect("Rotation required for DriveV3"),
        }),
        "EdenQuantizer" | "EDEN" => Box::new(EdenQuantizer {
            dim: state.dim,
            bits: state.bits,
            mode: state.mode.unwrap_or_else(|| "unbiased".to_string()),
            rotation: state.rotation.expect("Rotation required for Eden"),
            codebook: state.codebook.expect("Codebook required for Eden"),
        }),
        _ => return Err(format!("Unsupported algo_id: {}", state.algo_id).into()),
    };

    println!("Loading test vectors from {vectors_path}...");
    let vectors_file = File::open(vectors_path)?;
    let vectors: Vec<Vec<f32>> = serde_json::from_reader(BufReader::new(vectors_file))?;

    let mut results = Vec::with_capacity(vectors.len());

    for (i, v) in vectors.iter().enumerate() {
        let q_res = quantizer.quantize(v);
        let x_hat = quantizer.decode(&q_res);
        let score = quantizer.score(v, &q_res);

        results.push(VectorResult {
            values: q_res.values,
            signs: q_res.signs,
            scale: q_res.scale,
            x_hat,
            score,
        });
        if (i + 1) % 10 == 0 {
            println!("Processed {}/{} vectors...", i + 1, vectors.len());
        }
    }

    let output_file = File::create(output_path)?;
    serde_json::to_writer_pretty(output_file, &results)?;
    println!("Successfully wrote results to {output_path}");

    Ok(())
}
