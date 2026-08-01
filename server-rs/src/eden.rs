use crate::quantizers::EdenState;

const SQRT_PI_2: f32 = 1.2533141373155003;

/// ---------------------------------------------------------------------------
/// 3-bit unpack: 3 bytes → 8 indices (matching Python _unpack_numpy exactly)
///
/// Byte layout (LSB-first):
///   b0[2:0]=i0, b0[5:3]=i1, b0[7:6]+b1[0]=i2,
///   b1[4:1]=i3, b1[7:5]=i4, b1[7]+b2[1:0]=i5,
///   b2[4:2]=i6, b2[7:5]=i7
/// ---------------------------------------------------------------------------
#[inline]
pub fn unpack_3bit(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut indices = vec![0u8; dim];
    let mut out = 0usize;
    for chunk in packed.chunks(3) {
        if out + 8 > dim || chunk.len() < 3 {
            break;
        }
        let b0 = chunk[0];
        let b1 = chunk[1];
        let b2 = chunk[2];
        indices[out] = b0 & 7;
        indices[out + 1] = (b0 >> 3) & 7;
        indices[out + 2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2);
        indices[out + 3] = (b1 >> 1) & 7;
        indices[out + 4] = (b1 >> 4) & 7;
        indices[out + 5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1);
        indices[out + 6] = (b2 >> 2) & 7;
        indices[out + 7] = b2 >> 5;
        out += 8;
    }
    // Handle trailing incomplete chunk
    while out < dim {
        let byte_i = out * 3 / 8;
        let bit_offs = (out * 3) % 8;
        if byte_i >= packed.len() {
            break;
        }
        let val;
        if bit_offs <= 5 {
            val = (packed[byte_i] >> bit_offs) & 7;
        } else {
            let low = packed[byte_i] >> bit_offs;
            let high = if byte_i + 1 < packed.len() {
                packed[byte_i + 1] & ((1 << (3 - (8 - bit_offs))) - 1)
            } else {
                0
            };
            val = low | (high << (8 - bit_offs));
        }
        indices[out] = val;
        out += 1;
    }
    indices
}

/// Bit-pack 3-bit indices (LSB-first). For testing round-trip.
pub fn pack_3bit(indices: &[u8], out: &mut [u8]) {
    let dim = indices.len();
    for i in 0..dim {
        let val = indices[i] as u16;
        for b in 0..3 {
            if (val >> b) & 1 != 0 {
                let bit_pos = i * 3 + b as usize;
                out[bit_pos >> 3] |= 1 << (bit_pos & 7);
            }
        }
    }
}

pub const fn packed_size_3bit(dim: usize) -> usize {
    (dim * 3 + 7) / 8
}

/// ---------------------------------------------------------------------------
/// Scalar EDEN V3 score for one compressed vector
///
/// score = <centroids[indices], q_rot> * norm  +  sqrt(π/2)/dim * r_norm * <q_qjl, stored_qjl>
/// ---------------------------------------------------------------------------
pub fn score_eden(
    centroids: &[f32; 8],
    q_rot: &[f32],
    q_qjl: &[f32],
    packed: &[u8],
    norm: f32,
    r_norm: f32,
    stored_qjl: &[i8],
    dim: usize,
) -> f32 {
    if norm < 1e-12 {
        return 0.0;
    }
    let indices = unpack_3bit(packed, dim);

    let mut s1 = 0.0f32;
    for i in 0..dim {
        s1 += centroids[indices[i] as usize] * q_rot[i];
    }
    s1 *= norm;

    let mut qjl_dot = 0.0f32;
    for i in 0..dim {
        qjl_dot += (stored_qjl[i] as f32) * q_qjl[i];
    }

    let c = SQRT_PI_2 / dim as f32;
    s1 + c * r_norm * qjl_dot
}

/// ---------------------------------------------------------------------------
/// SIMD-accelerated EDEN V3 scoring using wide::f32x16 (AVX-512) / f32x8 (AVX2)
///
/// Processes 16 coordinates per loop iteration.
/// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
pub fn score_eden_simd(
    centroids: &[f32; 8],
    q_rot: &[f32],
    q_qjl: &[f32],
    packed: &[u8],
    norm: f32,
    r_norm: f32,
    stored_qjl: &[i8],
    dim: usize,
) -> f32 {
    if norm < 1e-12 {
        return 0.0;
    }
    let indices = unpack_3bit(packed, dim);

    let mut s1_acc = wide::f32x8::ZERO;
    let mut qjl_acc = wide::f32x8::ZERO;
    let mut i = 0;

    // Process 8 at a time (AVX2, 256-bit)
    while i + 8 <= dim {
        let c0 = [
            centroids[indices[i] as usize],
            centroids[indices[i + 1] as usize],
            centroids[indices[i + 2] as usize],
            centroids[indices[i + 3] as usize],
            centroids[indices[i + 4] as usize],
            centroids[indices[i + 5] as usize],
            centroids[indices[i + 6] as usize],
            centroids[indices[i + 7] as usize],
        ];
        let c_vals = wide::f32x8::from(c0);
        let qr = wide::f32x8::from(<&[f32; 8]>::try_from(&q_rot[i..i + 8]).unwrap());
        s1_acc = s1_acc + c_vals * qr;

        let qj = wide::f32x8::from(<&[f32; 8]>::try_from(&q_qjl[i..i + 8]).unwrap());
        let st = [
            stored_qjl[i] as f32,
            stored_qjl[i + 1] as f32,
            stored_qjl[i + 2] as f32,
            stored_qjl[i + 3] as f32,
            stored_qjl[i + 4] as f32,
            stored_qjl[i + 5] as f32,
            stored_qjl[i + 6] as f32,
            stored_qjl[i + 7] as f32,
        ];
        let st_vals = wide::f32x8::from(st);
        qjl_acc = qjl_acc + qj * st_vals;

        i += 8;
    }

    let mut s1 = s1_acc.reduce_add();
    let mut qjl_dot = qjl_acc.reduce_add();

    // Remainder
    for j in i..dim {
        s1 += centroids[indices[j] as usize] * q_rot[j];
        qjl_dot += (stored_qjl[j] as f32) * q_qjl[j];
    }

    s1 *= norm;
    let c = SQRT_PI_2 / dim as f32;
    s1 + c * r_norm * qjl_dot
}

#[cfg(not(target_arch = "x86_64"))]
pub fn score_eden_simd(
    centroids: &[f32; 8],
    q_rot: &[f32],
    q_qjl: &[f32],
    packed: &[u8],
    norm: f32,
    r_norm: f32,
    stored_qjl: &[i8],
    dim: usize,
) -> f32 {
    // Fallback to scalar on non-x86
    score_eden(centroids, q_rot, q_qjl, packed, norm, r_norm, stored_qjl, dim)
}

/// Quantize a raw embedding vector using EDEN V3.
///
/// Returns (packed, norm, qjl, r_norm) matching Python `compress()`.
pub fn quantize(
    state: &EdenState,
    vec: &[f32],
) -> (Vec<u8>, f32, Vec<i8>, f32) {
    let dim = state.dim;
    if state.codebook.is_empty() || state.rotation.is_empty() {
        // No state: return zero vector
        let packed_sz = packed_size_3bit(dim);
        return (vec![0u8; packed_sz], 0.0, vec![1i8; dim], 0.0);
    }

    let norm = dot(vec, vec).sqrt();
    if norm < 1e-12 {
        let packed_sz = packed_size_3bit(dim);
        return (vec![0u8; packed_sz], 0.0, vec![1i8; dim], 0.0);
    }

    // Rotate unit vector: q_rot = rotation @ (vec / norm)
    let inv_norm = 1.0 / norm;
    let mut q_rot = vec![0.0f32; dim];
    for i in 0..dim {
        let row_start = i * dim;
        let row = &state.rotation[row_start..row_start + dim];
        let mut s = 0.0f32;
        for j in 0..dim {
            s += row[j] * vec[j] * inv_norm;
        }
        q_rot[i] = s;
    }

    // Find nearest centroid for each coordinate
    let centroids = &state.codebook;
    let n_centroids = centroids.len();
    let mut indices = vec![0u8; dim];
    let mut packed = vec![0u8; packed_size_3bit(dim)];
    for i in 0..dim {
        let val = q_rot[i];
        let mut best = 0u8;
        let mut best_dist = (val - centroids[0]).abs();
        for c in 1..n_centroids {
            let d = (val - centroids[c]).abs();
            if d < best_dist {
                best_dist = d;
                best = c as u8;
            }
        }
        indices[i] = best;
    }
    pack_3bit(&indices, &mut packed);

    // Reconstruct: x_hat = rotation^T @ centroids[indices] * norm
    let mut x_hat = vec![0.0f32; dim];
    for j in 0..dim {
        let mut s = 0.0f32;
        for i in 0..dim {
            // rotation^T[j][i] = rotation[i][j]
            let rot_ij = state.rotation[i * dim + j];
            s += rot_ij * centroids[indices[i] as usize];
        }
        x_hat[j] = s * norm;
    }

    // Residual = vec - x_hat
    let mut residual = vec![0.0f32; dim];
    for i in 0..dim {
        residual[i] = vec[i] - x_hat[i];
    }
    let r_norm = dot(&residual, &residual).sqrt();

    // QJL sign: qjl_matrix @ residual → sign → i8
    let mut qjl = vec![0i8; dim];
    if r_norm > 1e-12 {
        for i in 0..dim {
            let row_start = i * dim;
            let row = &state.qjl_matrix[row_start..row_start + dim];
            let mut s = 0.0f32;
            for j in 0..dim {
                s += row[j] * residual[j];
            }
            qjl[i] = if s >= 0.0 { 1i8 } else { -1i8 };
        }
    } else {
        qjl.fill(1i8);
    }

    (packed, norm, qjl, r_norm)
}

/// Prepare query: apply rotation and QJL matrix to query vector.
/// Returns (q_rot, q_qjl).
pub fn prepare_query(state: &EdenState, query: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let dim = state.dim;
    let q_rot = rotate_query(&state.rotation, query, dim);
    let q_qjl = apply_qjl(&state.qjl_matrix, query, dim);
    (q_rot, q_qjl)
}

/// Apply random orthogonal rotation: q_rot = rotation @ query
pub fn rotate_query(rotation: &[f32], query: &[f32], dim: usize) -> Vec<f32> {
    let mut q_rot = vec![0.0f32; dim];
    for i in 0..dim {
        let row_start = i * dim;
        let row = &rotation[row_start..row_start + dim];
        q_rot[i] = dot(row, query);
    }
    q_rot
}

/// Apply QJL matrix: q_qjl = qjl_matrix @ query
pub fn apply_qjl(qjl_matrix: &[f32], query: &[f32], dim: usize) -> Vec<f32> {
    let mut q_qjl = vec![0.0f32; dim];
    for i in 0..dim {
        let row_start = i * dim;
        let row = &qjl_matrix[row_start..row_start + dim];
        q_qjl[i] = dot(row, query);
    }
    q_qjl
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        sum += a[i] * b[i];
    }
    sum
}

/// ---------------------------------------------------------------------------
/// State I/O: binary export/import for EdenState
/// ---------------------------------------------------------------------------
pub fn save_state(path: &str, state: &EdenState) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(state.dim as u32).to_le_bytes())?;
    f.write_all(&(state.bits as u32).to_le_bytes())?;

    let rotation_bytes = bytemuck::cast_slice(&state.rotation);
    f.write_all(rotation_bytes)?;

    let qjl_bytes = bytemuck::cast_slice(&state.qjl_matrix);
    f.write_all(qjl_bytes)?;

    let codebook_bytes = bytemuck::cast_slice(&state.codebook);
    f.write_all(codebook_bytes)?;

    Ok(())
}

pub fn load_state(path: &str) -> std::io::Result<EdenState> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    if buf.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "state file too short (expected at least 8 bytes)",
        ));
    }
    let dim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let bits = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

    let rot_size = dim * dim * 4;
    let rotation_end = 8 + rot_size;
    let qjl_end = rotation_end + rot_size;

    let rotation: Vec<f32> = bytemuck::cast_slice(&buf[8..rotation_end]).to_vec();
    let qjl_matrix: Vec<f32> = bytemuck::cast_slice(&buf[rotation_end..qjl_end]).to_vec();
    let codebook: Vec<f32> = bytemuck::cast_slice(&buf[qjl_end..]).to_vec();

    Ok(EdenState {
        dim,
        bits,
        mode: "unbiased".to_string(),
        codebook,
        rotation,
        qjl_matrix,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Golden Vector test data (generated from Python with seed=42, dim=384)
    // See golden_export.py and testdata/ for the fresh data pipeline
    const CENTROIDS: [f32; 8] = [
        -0.107062735,
        -0.06569654,
        -0.035492413,
        -0.009296271,
        0.015659666,
        0.04153669,
        0.07105827,
        0.11144634,
    ];

    const NORM: f32 = 1.0;
    const R_NORM: f32 = 0.18812884;
    const EXPECTED_SCORE: f32 = -0.0023083051;

    const INDICES_16: [u8; 16] = [2, 0, 3, 5, 6, 2, 2, 3, 3, 3, 4, 3, 1, 0, 5, 2];

    const REF_PACKED: [u8; 144] = [
        0xc2, 0x6a, 0x69, 0x1b, 0x17, 0x54, 0x25, 0x0a, 0xbb, 0x8a, 0x30, 0x25,
        0xb2, 0xa4, 0x3b, 0x92, 0x62, 0x3a, 0x0d, 0x39, 0x55, 0x3e, 0x39, 0x30,
        0x61, 0xb4, 0x32, 0x84, 0xb3, 0x61, 0x55, 0x00, 0x6d, 0x19, 0x73, 0x4e,
        0xfa, 0x36, 0x89, 0xc1, 0x3c, 0x73, 0x9c, 0xcc, 0xd9, 0x4d, 0xbb, 0x2a,
        0xd9, 0xe8, 0xca, 0xa7, 0xd9, 0x0a, 0xac, 0x33, 0x66, 0xd5, 0xba, 0x35,
        0xac, 0x44, 0x91, 0x63, 0x6a, 0x41, 0x93, 0xca, 0x35, 0xa2, 0x26, 0xba,
        0x8c, 0x1b, 0x4b, 0x54, 0xc1, 0x94, 0x0d, 0xd7, 0x0a, 0x1a, 0x47, 0x8e,
        0x24, 0xc4, 0x6c, 0x0d, 0xb5, 0x31, 0x99, 0x1a, 0xb3, 0x4c, 0x39, 0x31,
        0x94, 0xdb, 0x7d, 0x62, 0x52, 0x55, 0xdc, 0xda, 0x9e, 0x13, 0x6b, 0x4a,
        0xa6, 0xb2, 0x71, 0xeb, 0x48, 0x27, 0xe2, 0xa2, 0x31, 0x5b, 0x3a, 0xaf,
        0x44, 0x19, 0xc7, 0x7c, 0xf8, 0x91, 0x1d, 0x41, 0xaa, 0xcc, 0xbc, 0x34,
        0x27, 0xb9, 0x6a, 0x92, 0x3b, 0xd5, 0xe4, 0xf4, 0xb5, 0x97, 0x96, 0x5a,
    ];

    /// Path to the testdata directory relative to the crate root.
    /// Tests that need full 384-dim vectors load from binary files here.
    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata");

    fn golden_packed() -> Vec<u8> {
        REF_PACKED.to_vec()
    }

    fn load_bin_f32(name: &str) -> Vec<f32> {
        let path = std::path::Path::new(TESTDATA).join(name);
        let bytes = std::fs::read(&path).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    fn load_bin_i8(name: &str) -> Vec<i8> {
        let path = std::path::Path::new(TESTDATA).join(name);
        let bytes = std::fs::read(&path).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    fn golden_q_rot() -> Vec<f32> {
        load_bin_f32("golden_q_rot.bin")
    }

    fn golden_q_qjl() -> Vec<f32> {
        load_bin_f32("golden_q_qjl.bin")
    }

    fn golden_stored_qjl() -> Vec<i8> {
        load_bin_i8("golden_stored_qjl.bin")
    }

    // ── Unpack tests ────────────────────────────────────────────────────

    #[test]
    fn test_unpack_3bit_first_16() {
        let packed = golden_packed();
        let indices = unpack_3bit(&packed, 384);
        assert_eq!(&indices[..16], &INDICES_16, "First 16 indices mismatch");
    }

    #[test]
    fn test_unpack_3bit_roundtrip() {
        let dim = 384;
        let mut indices = Vec::with_capacity(dim);
        let mut rng: u64 = 12345;
        for _ in 0..dim {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            indices.push(((rng >> 33) & 0x7) as u8);
        }

        let mut packed = vec![0u8; packed_size_3bit(dim)];
        pack_3bit(&indices, &mut packed);
        let unpacked = unpack_3bit(&packed, dim);

        assert_eq!(indices, unpacked, "Round-trip failed at dim=384");
    }

    // ── Golden Vector score test ────────────────────────────────────────

    #[test]
    fn test_eden_score_golden() {
        let packed = golden_packed();
        let q_rot = golden_q_rot();
        let q_qjl = golden_q_qjl();
        let stored_qjl = golden_stored_qjl();
        let dim = 384;

        let score = score_eden(
            &CENTROIDS,
            &q_rot,
            &q_qjl,
            &packed,
            NORM,
            R_NORM,
            &stored_qjl,
            dim,
        );

        let diff = (score - EXPECTED_SCORE).abs();
        assert!(
            diff < 1e-5,
            "Score mismatch: got {score:.10}, expected {EXPECTED_SCORE:.10}, diff {diff:.2e}"
        );
    }

    #[test]
    fn test_eden_score_simd_golden() {
        let packed = golden_packed();
        let q_rot = golden_q_rot();
        let q_qjl = golden_q_qjl();
        let stored_qjl = golden_stored_qjl();
        let dim = 384;

        let score = score_eden_simd(
            &CENTROIDS,
            &q_rot,
            &q_qjl,
            &packed,
            NORM,
            R_NORM,
            &stored_qjl,
            dim,
        );

        let diff = (score - EXPECTED_SCORE).abs();
        assert!(
            diff < 1e-5,
            "SIMD score mismatch: got {score:.10}, expected {EXPECTED_SCORE:.10}, diff {diff:.2e}"
        );
    }

    #[test]
    fn test_scalar_simd_match() {
        let packed = golden_packed();
        let q_rot = golden_q_rot();
        let q_qjl = golden_q_qjl();
        let stored_qjl = golden_stored_qjl();
        let dim = 384;

        let scalar = score_eden(
            &CENTROIDS,
            &q_rot,
            &q_qjl,
            &packed,
            NORM,
            R_NORM,
            &stored_qjl,
            dim,
        );

        let simd = score_eden_simd(
            &CENTROIDS,
            &q_rot,
            &q_qjl,
            &packed,
            NORM,
            R_NORM,
            &stored_qjl,
            dim,
        );

        let diff = (scalar - simd).abs();
        assert!(
            diff < 1e-6,
            "Scalar vs SIMD mismatch: {scalar:.10} vs {simd:.10}, diff {diff:.2e}"
        );
    }

    // ── State I/O test ──────────────────────────────────────────────────

    #[test]
    fn test_state_roundtrip() {
        let state = EdenState {
            dim: 384,
            bits: 3,
            mode: "unbiased".into(),
            codebook: CENTROIDS.to_vec(),
            rotation: vec![0.0f32; 384 * 384],
            qjl_matrix: vec![1.0f32; 384 * 384],
        };

        let path = std::format!("/tmp/test_eden_state_{}.bin", std::process::id());
        save_state(&path, &state).unwrap();
        let loaded = load_state(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.dim, 384);
        assert_eq!(loaded.bits, 3);
        assert_eq!(loaded.codebook, CENTROIDS);
        assert_eq!(loaded.qjl_matrix.len(), 384 * 384);
        assert_eq!(loaded.qjl_matrix[0], 1.0);
    }
}
