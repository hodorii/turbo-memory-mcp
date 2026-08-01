use crate::traits::{Quantizer, QuantizedResult};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use wide::f32x8;

#[derive(Deserialize, Serialize)]
pub struct EdenState {
    pub dim: usize,
    pub bits: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
    pub rotation: Vec<f32>,
    pub codebook: Vec<f32>,
    pub qjl_matrix: Vec<f32>,
}

fn default_mode() -> String {
    "unbiased".to_string()
}

/// EDEN 양자화 구현체 (Full b-bit Lloyd-Max + S-Scaling)
pub struct EdenQuantizer {
    pub dim: usize,
    pub bits: usize,
    pub mode: String,
    pub rotation: Vec<f32>,
    pub codebook: Vec<f32>,
}

impl EdenQuantizer {
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: EdenState = serde_json::from_reader(reader)?;
        
        Ok(Self {
            dim: state.dim,
            bits: state.bits,
            mode: state.mode,
            rotation: state.rotation,
            codebook: state.codebook,
        })
    }

    fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.dim];
        let rotation = &self.rotation;
        let dim = self.dim;

        for i in 0..dim {
            let xi = f32x8::splat(x[i]);
            let row_offset = i * dim;
            let row = &rotation[row_offset..row_offset + dim];
            
            for j in (0..dim).step_by(8) {
                let r_vec = f32x8::new(row[j..j + 8].try_into().unwrap());
                let mut y_vec = f32x8::new(y[j..j + 8].try_into().unwrap());
                y_vec += xi * r_vec;
                let res: [f32; 8] = y_vec.into();
                y[j..j + 8].copy_from_slice(&res);
            }



        }
        y
    }
}

impl Quantizer for EdenQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn quantize(&self, x: &[f32]) -> QuantizedResult {
        let y = self.rotate(x);
        let mut values = Vec::with_capacity(self.dim);
        
        for i in 0..self.dim {
            let mut min_dist = f32::MAX;
            let mut best_idx = 0;
            for (idx, &val) in self.codebook.iter().enumerate() {
                let dist = (y[i] - val).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }
            values.push(best_idx as i32);
        }
        
        let x_norm_sq: f32 = x.iter().map(|&val| val * val).sum();
        let norm = x_norm_sq.sqrt();
        let mut inner = 0.0f32;
        for i in 0..self.dim {
            let q_val = self.codebook[values[i] as usize];
            inner += y[i] * q_val;
        }
        
        let scale = if inner.abs() > 1e-10 {
            x_norm_sq / inner
        } else {
            1.0
        };

        // Calculate r_norm (residual norm)
        let mut x_hat = vec![0.0; self.dim];
        for i in 0..self.dim {
            let qv = self.codebook[values[i] as usize] * scale;
            let row_offset = i * self.dim;
            for j in (0..self.dim).step_by(8) {
                let r_vec = f32x8::new(self.rotation[row_offset + j..row_offset + j + 8].try_into().unwrap());
                let mut x_vec = f32x8::new(x_hat[j..j+8].try_into().unwrap());
                x_vec += f32x8::splat(qv) * r_vec;
                let x_arr: [f32; 8] = x_vec.into();
                x_hat[j..j + 8].copy_from_slice(&x_arr);
            }
        }
        let mut r_norm_sq = 0.0f32;
        for i in 0..self.dim {
            let diff = x[i] - x_hat[i];
            r_norm_sq += diff * diff;
        }
        let r_norm = r_norm_sq.sqrt();
        
        QuantizedResult {
            algo_id: "EDEN".to_string(),
            values,
            signs: None,
            scale: Some(scale),
            norm,
            r_norm,
        }
    }


    fn decode(&self, q: &QuantizedResult) -> Vec<f32> {
        let mut q_val = vec![0.0; self.dim];
        for i in 0..self.dim {
            q_val[i] = self.codebook[q.values[i] as usize];
        }
        
        let mut x_hat = vec![0.0; self.dim];
        let scale = q.scale.unwrap_or(1.0);
        let rotation = &self.rotation;
        let dim = self.dim;
        
        for i in 0..dim {
            let qv = f32x8::splat(q_val[i] * scale);
            let row_offset = i * dim;
            let row = &rotation[row_offset..row_offset + dim];
            for j in (0..dim).step_by(8) {
                let r_vec = f32x8::new(row[j..j + 8].try_into().unwrap());
                let mut x_vec = f32x8::new(x_hat[j..j + 8].try_into().unwrap());
                x_vec += qv * r_vec;
                let res: [f32; 8] = x_vec.into();
                x_hat[j..j + 8].copy_from_slice(&res);
            }



        }
        x_hat
    }

    fn score(&self, query: &[f32], q: &QuantizedResult) -> f32 {
        let y_query = self.rotate(query);
        let mut score_acc = 0.0f64;
        let scale = q.scale.unwrap_or(1.0) as f64;
        
        for i in 0..self.dim {
            let q_val = self.codebook[q.values[i] as usize];
            score_acc += (y_query[i] as f64) * (q_val as f64);
        }
        
        (score_acc * scale) as f32
    }

    fn score_packed(&self, query: &[f32], packed: &[u8], scale: f32) -> f32 {
        self.score_packed_impl(query, packed, scale)
    }
}

// ── Packed SIMD scoring (f32x8) ────────────────────────────────────────
// These skip the unpack step entirely, operating directly on packed bytes.
// Each bit-width uses a different extraction strategy:
//   2-bit → u16 (4 indices per u16, 2 bytes -> 8 indices in 2 u16 reads)
//   3-bit → bytewise (3 bytes → 8 indices, bit-twiddling)
//   4-bit → u32 (8 indices per u32, 4 bytes per u32 read)

fn score_2bit_packed(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
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

fn score_3bit_packed(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
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

fn score_4bit_packed(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, s: f32) -> f32 {
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

/// Number of packed bytes needed for `dim` coordinates at `bits` per coordinate.
pub fn packed_size(dim: usize, bits: usize) -> usize {
    (dim * bits + 7) / 8
}

/// Pack indices into bit-packed format.
/// Supports 2-bit (4 indices/byte), 3-bit (8 indices/3 bytes), 4-bit (2 indices/byte).
pub fn pack_indices(indices: &[u8], bits: usize, out: &mut [u8]) {
    let dim = indices.len();
    match bits {
        2 => {
            for i in (0..dim).step_by(4) {
                let byte_i = i / 4;
                let mut byte = 0u8;
                for j in 0..4 {
                    if i + j < dim {
                        byte |= (indices[i + j] & 3) << (j * 2);
                    }
                }
                out[byte_i] = byte;
            }
        }
        3 => {
            for i in 0..dim {
                let val = indices[i] as u16;
                for b in 0..3 {
                    if (val >> b) & 1 != 0 {
                        let bit_pos = i * 3 + b;
                        out[bit_pos >> 3] |= 1 << (bit_pos & 7);
                    }
                }
            }
        }
        4 => {
            for i in (0..dim).step_by(2) {
                let byte_i = i / 2;
                let mut byte = indices[i] & 15;
                if i + 1 < dim {
                    byte |= (indices[i + 1] & 15) << 4;
                }
                out[byte_i] = byte;
            }
        }
        _ => panic!("pack_indices: unsupported bit-width {bits}"),
    }
}

/// Runtime bit-width dispatch: picks the fastest packed method per bit-width.
pub fn score_quantized(centroids: &[f32], q_rot: &[f32], packed: &[u8], dim: usize, bits: u32, s: f32) -> f32 {
    match bits {
        2 => score_2bit_packed(centroids, q_rot, packed, dim, s),
        3 => score_3bit_packed(centroids, q_rot, packed, dim, s),
        4 => score_4bit_packed(centroids, q_rot, packed, dim, s),
        _ => panic!("score_quantized: unsupported bit-width {bits}"),
    }
}

impl EdenQuantizer {
    pub fn score_packed_impl(&self, query: &[f32], packed: &[u8], scale: f32) -> f32 {
        let y_query = self.rotate(query);
        score_quantized(&self.codebook, &y_query, packed, self.dim, self.bits as u32, scale)
    }
}

pub struct DriveV3Quantizer {
    pub dim: usize,
    pub rotation: Vec<f32>,
}

impl DriveV3Quantizer {
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: DriveV3State = serde_json::from_reader(reader)?;
        Ok(Self { dim: state.dim, rotation: state.rotation })
    }
}

#[derive(Deserialize, Serialize)]
struct DriveV3State {
    dim: usize,
    bits: usize,
    rotation: Vec<f32>,
    codebook: Vec<f32>,
}

impl Quantizer for DriveV3Quantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn quantize(&self, x: &[f32]) -> QuantizedResult {
        let mut y = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                y[j] += x[i] * self.rotation[i * self.dim + j];
            }
        }
        let signs: Vec<i8> = y.iter().map(|&v| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 }).collect();
        let scale = (y.iter().map(|&v| v.abs() as f64).sum::<f64>() / self.dim as f64) as f32;
        let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let mut x_hat = vec![0.0; self.dim];
        for j in 0..self.dim {
            let s_val = signs[j] as f32 * scale;
            for i in 0..self.dim {
                x_hat[i] += s_val * self.rotation[j * self.dim + i];
            }
        }
        let r_norm = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();

        QuantizedResult {
            algo_id: "DRIVE_V3".to_string(),
            values: vec![0; self.dim],
            signs: Some(signs),
            scale: Some(scale),
            norm,
            r_norm,
        }
    }

    fn decode(&self, q: &QuantizedResult) -> Vec<f32> {
        let signs = q.signs.as_ref().unwrap();
        let scale = q.scale.unwrap_or(0.1);
        let mut x_hat = vec![0.0; self.dim];
        for j in 0..self.dim {
            let s_val = signs[j] as f32 * scale;
            for i in 0..self.dim {
                x_hat[i] += s_val * self.rotation[j * self.dim + i];
            }
        }
        x_hat
    }

    fn score(&self, query: &[f32], q: &QuantizedResult) -> f32 {
        let mut y_query = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                y_query[j] += query[i] * self.rotation[i * self.dim + j];
            }
        }
        let signs = q.signs.as_ref().unwrap();
        let scale = q.scale.unwrap_or(0.1) as f64;
        let mut score = 0.0f64;
        for i in 0..self.dim {
            score += (signs[i] as f64) * (y_query[i] as f64);
        }
        (score * scale) as f32
    }
}

pub struct QJLQuantizer {
    pub dim: usize,
    pub rotation: Vec<f32>,
}

impl Quantizer for QJLQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn quantize(&self, x: &[f32]) -> QuantizedResult {
        let mut y = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                y[j] += x[i] * self.rotation[i * self.dim + j];
            }
        }
        let signs: Vec<i8> = y.iter().map(|&v| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 }).collect();
        let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let mut x_hat = vec![0.0; self.dim];
        for j in 0..self.dim {
            let s_val = signs[j] as f32;
            for i in 0..self.dim {
                x_hat[i] += s_val * self.rotation[j * self.dim + i];
            }
        }
        let r_norm = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();

        QuantizedResult {
            algo_id: "QJL".to_string(),
            values: vec![0; self.dim],
            signs: Some(signs),
            scale: None,
            norm,
            r_norm,
        }
    }

    fn decode(&self, q: &QuantizedResult) -> Vec<f32> {
        let signs = q.signs.as_ref().unwrap();
        let mut x_hat = vec![0.0; self.dim];
        for j in 0..self.dim {
            let s_val = signs[j] as f32;
            for i in 0..self.dim {
                x_hat[i] += s_val * self.rotation[j * self.dim + i];
            }
        }
        x_hat
    }

    fn score(&self, query: &[f32], q: &QuantizedResult) -> f32 {
        let mut y_query = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                y_query[j] += query[i] * self.rotation[i * self.dim + j];
            }
        }
        let signs = q.signs.as_ref().unwrap();
        let mut score = 0.0f64;
        for i in 0..self.dim {
            score += (signs[i] as f64) * (y_query[i] as f64);
        }
        score as f32
    }
}
