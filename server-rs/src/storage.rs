use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use thiserror::Error;

use crate::schema;
use crate::types::{FileHeader, VectorIndex, TMD_MAGIC, TMD_VERSION};

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic: expected 0x{expected:08x}, got 0x{actual:08x}")]
    InvalidMagic { expected: u32, actual: u32 },
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u16),
    #[error("Unexpected end of file")]
    UnexpectedEof,
}

/// Memory-mapped vector index.
///
/// The `.tmd` file uses struct-of-arrays layout:
///   [FileHeader] [norm: f32; N] [r_norm: f32; N] [scale: f32; N]
///               [qjl: i8; N*dim] [packed: u8; N*padded_size]
///
/// All vector data lives behind the mmap so startup is O(1) — no data is copied.
pub struct MmapIndex {
    pub header: FileHeader,
    pub mmap: Arc<Mmap>,
    dim: u32,
    count: u64,
    pub stride_qjl: u64,
    pub stride_packed: u64,
    norms_offset: u64,
    r_norms_offset: u64,
    scale_offset: u64,
    pub qjl_offset: u64,
    pub packed_offset: u64,
}

impl MmapIndex {
    /// Open and validate a .tmd file, mapping it into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        if mmap.len() < size_of::<FileHeader>() {
            return Err(StorageError::UnexpectedEof);
        }

        let header = unsafe { *(mmap.as_ptr() as *const FileHeader) };

        if header.magic != TMD_MAGIC {
            return Err(StorageError::InvalidMagic {
                expected: TMD_MAGIC,
                actual: header.magic,
            });
        }
        if header.version != TMD_VERSION {
            return Err(StorageError::UnsupportedVersion(header.version));
        }

        let dim = header.dim as u64;
        let count = header.count;
        let bits = header.bits as u64;
        let packed_per = (dim * bits + 7) / 8; // ceil(dim * bits / 8)

        // Offsets (SoA layout after the header)
        let hdr_sz = size_of::<FileHeader>() as u64;
        let norms_offset = hdr_sz;
        let r_norms_offset = norms_offset + count * 4;
        let scale_offset = r_norms_offset + count * 4;
        let qjl_offset = scale_offset + count * 4;
        let packed_offset = qjl_offset + count * dim;

        Ok(Self {
            header,
            mmap,
            dim: header.dim,
            count: header.count,
            stride_qjl: dim,
            stride_packed: packed_per,
            norms_offset,
            r_norms_offset,
            scale_offset,
            qjl_offset,
            packed_offset,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// SAFETY: Accesses raw mmap pointers. Only call when count > 0.
    unsafe fn slice_at<T>(&self, offset: u64, len: u64) -> &[T] {
        let byte_start = offset as usize;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(byte_start) as *const T,
                len as usize,
            )
        }
    }

    pub fn norms(&self) -> &[f32] {
        if self.count == 0 {
            return &[];
        }
        unsafe { self.slice_at(self.norms_offset, self.count) }
    }

    pub fn r_norms(&self) -> &[f32] {
        if self.count == 0 {
            return &[];
        }
        unsafe { self.slice_at(self.r_norms_offset, self.count) }
    }

    pub fn scales(&self) -> &[f32] {
        if self.count == 0 {
            return &[];
        }
        unsafe { self.slice_at(self.scale_offset, self.count) }
    }

    pub fn qjl(&self) -> &[i8] {
        if self.count == 0 {
            return &[];
        }
        unsafe { self.slice_at(self.qjl_offset, self.count * self.stride_qjl) }
    }

    pub fn packed(&self) -> &[u8] {
        if self.count == 0 {
            return &[];
        }
        let total_packed = self.count * self.stride_packed;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.packed_offset as usize),
                total_packed as usize,
            )
        }
    }

    /// Copy a single entry out into an owned VectorIndex
    #[allow(dead_code)]
    pub fn get(&self, idx: u64, entry_id: &str) -> Option<VectorIndex> {
        if idx >= self.count {
            return None;
        }
        let dim = self.dim as usize;
        let packed_per = self.stride_packed as usize;

        let norm = self.norms()[idx as usize];
        let r_norm = self.r_norms()[idx as usize];
        let scale = self.scales()[idx as usize];

        let qjl_start = (idx * self.stride_qjl) as usize;
        let qjl = self.qjl()[qjl_start..qjl_start + dim].to_vec();

        let packed_start = (idx * self.stride_packed) as usize;
        let packed = self.packed()[packed_start..packed_start + packed_per].to_vec();

        Some(VectorIndex {
            id: entry_id.to_string(),
            norm,
            r_norm,
            scale,
            qjl,
            packed,
        })
    }
}

/// Rebuild the .tmd file from all entries in the SQLite database.
///
/// Writes a complete SoA (Structure of Arrays) layout file that can be
/// memory-mapped by `MmapIndex` for zero-copy search.
///
/// `bits` is the quantization bit-width (2, 3, or 4) written into the header.
pub fn rebuild(path: &str, entries: &[schema::SearchEntry], bits: u16) -> Result<(), StorageError> {
    if entries.is_empty() {
        let header = FileHeader {
            magic: TMD_MAGIC,
            version: TMD_VERSION,
            bits: bits,
            dim: 384,
            count: 0,
            meta_size: 0,
            data_offset: size_of::<FileHeader>() as u64,
        };
        std::fs::write(path, header.to_bytes())?;
        return Ok(());
    }

    let dim = entries[0].qjl.len() as u64;
    let count = entries.len() as u64;
    let packed_per = (dim * bits as u64 + 7) / 8;

    let hdr_sz = size_of::<FileHeader>() as u64;
    let norms_size = count * 4;
    let r_norms_size = count * 4;
    let scales_size = count * 4;
    let qjl_size = count * dim;
    let packed_size = count * packed_per;

    let mut buf: Vec<u8> = Vec::with_capacity(
        (hdr_sz + norms_size + r_norms_size + scales_size + qjl_size + packed_size) as usize,
    );

    // Header
    let header = FileHeader {
        magic: TMD_MAGIC,
        version: TMD_VERSION,
        bits,
        dim: dim as u32,
        count,
        meta_size: 0,
        data_offset: hdr_sz,
    };
    buf.extend_from_slice(&header.to_bytes());

    // SoA arrays
    for e in entries.iter() {
        buf.extend_from_slice(bytemuck::bytes_of(&e.norm));
    }
    for e in entries.iter() {
        buf.extend_from_slice(bytemuck::bytes_of(&e.r_norm));
    }
    for e in entries.iter() {
        buf.extend_from_slice(bytemuck::bytes_of(&e.scale));
    }
    for e in entries.iter() {
        buf.extend_from_slice(bytemuck::cast_slice(&e.qjl));
    }
    for e in entries.iter() {
        buf.extend_from_slice(&e.packed);
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    file.sync_all()?;
    Ok(())
}
