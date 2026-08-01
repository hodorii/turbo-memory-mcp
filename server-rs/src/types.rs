/// Magic bytes for .tmd file header: "TMCM" (Turbo Memory Compression Map)
pub const TMD_MAGIC: u32 = 0x544d434d;

/// Current format version
pub const TMD_VERSION: u16 = 2;

/// ---------------------------------------------------------------------------
/// .tmd binary file header
/// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FileHeader {
    pub magic: u32,
    pub version: u16,
    pub bits: u16,          // quantization bit-width (2, 3, or 4)
    pub dim: u32,
    pub count: u64,
    pub meta_size: u32,
    pub data_offset: u64,
}

impl FileHeader {
    /// Serialize to bytes matching repr(C) layout, including explicit padding bytes.
    /// This allows the file to be read back via unsafe pointer cast in MmapIndex.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(std::mem::size_of::<Self>());
        buf.extend_from_slice(&self.magic.to_le_bytes());       // offset  0, 4 bytes
        buf.extend_from_slice(&self.version.to_le_bytes());      // offset  4, 2 bytes
        buf.extend_from_slice(&self.bits.to_le_bytes());         // offset  6, 2 bytes
        buf.extend_from_slice(&self.dim.to_le_bytes());          // offset  8, 4 bytes
        buf.extend_from_slice(&[0u8; 4]);                        // offset 12, 4 bytes PADDING (u64 alignment)
        buf.extend_from_slice(&self.count.to_le_bytes());        // offset 16, 8 bytes
        buf.extend_from_slice(&self.meta_size.to_le_bytes());    // offset 24, 4 bytes
        buf.extend_from_slice(&[0u8; 4]);                        // offset 28, 4 bytes PADDING (u64 alignment)
        buf.extend_from_slice(&self.data_offset.to_le_bytes());  // offset 32, 8 bytes
        buf
    }
}

/// Per-vector structure of arrays accessor.
/// The .tmd file stores vectors as contiguous arrays (SoA layout) for SIMD
/// friendliness: all norms together, all r_norms together, etc.
#[derive(Debug, Clone)]
pub struct VectorIndex {
    pub id: String,
    pub norm: f32,
    pub r_norm: f32,
    pub scale: f32,
    pub qjl: Vec<i8>,
    pub packed: Vec<u8>,
}
