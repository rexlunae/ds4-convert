//! Streaming GGUF v3 writer.  A full plan (metadata + tensor infos) is
//! committed up front, then tensor data is appended strictly in plan order,
//! padded to 32 bytes — byte-compatible with candle's `gguf_file::write`
//! while never holding more than one tensor's bytes in memory.

use anyhow::{anyhow, Result};
use std::io::{Seek, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum V {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Every element must be the same variant (the element type is written
    /// once); nested arrays are not supported by GGUF.
    Arr(Vec<V>),
}

impl V {
    fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::Str(_) => 8,
            Self::Arr(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }
    fn elem_type_id(v: &[V]) -> u32 {
        match v.first() {
            Some(v) => v.type_id(),
            None => 8, // empty array of strings
        }
    }
    fn write(&self, w: &mut impl Write) -> Result<()> {
        match self {
            Self::U8(v) => w.write_all(&v.to_le_bytes())?,
            Self::I8(v) => w.write_all(&v.to_le_bytes())?,
            Self::U16(v) => w.write_all(&v.to_le_bytes())?,
            Self::I16(v) => w.write_all(&v.to_le_bytes())?,
            Self::U32(v) => w.write_all(&v.to_le_bytes())?,
            Self::I32(v) => w.write_all(&v.to_le_bytes())?,
            Self::U64(v) => w.write_all(&v.to_le_bytes())?,
            Self::I64(v) => w.write_all(&v.to_le_bytes())?,
            Self::F32(v) => w.write_all(&v.to_le_bytes())?,
            Self::F64(v) => w.write_all(&v.to_le_bytes())?,
            Self::Bool(v) => w.write_all(&[u8::from(*v)])?,
            Self::Str(v) => write_str(w, v)?,
            Self::Arr(vs) => {
                w.write_all(&V::elem_type_id(vs).to_le_bytes())?;
                w.write_all(&(vs.len() as u64).to_le_bytes())?;
                for v in vs {
                    v.write(w)?;
                }
            }
        }
        Ok(())
    }
}

fn write_str(w: &mut impl Write, s: &str) -> Result<()> {
    let b = s.as_bytes();
    w.write_all(&(b.len() as u64).to_le_bytes())?;
    w.write_all(b)?;
    Ok(())
}

pub struct TPlan {
    pub name: String,
    /// Candle-order dims: `[out, ..., in]`, innermost last.
    pub dims: Vec<u64>,
    /// GGML type id (F32=0, F16=1, Q4_0=2, Q8_0=8, Q2K=10, Q3K=11, Q4K=12,
    /// Q5K=13, Q6K=14, BF16=30).
    pub dtype_id: u32,
    pub nbytes: u64,
}

pub struct Writer {
    file: std::io::BufWriter<std::fs::File>,
    plan: Vec<TPlan>,
    at: usize,
    pending: usize,
    pad: [u8; 32],
}

impl Writer {
    pub fn create(path: &Path, metadata: &[(String, V)], plan: Vec<TPlan>) -> Result<Self> {
        let mut file = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path)?);
        file.write_all(b"GGUF")?;
        file.write_all(&3u32.to_le_bytes())?; // version 3
        file.write_all(&(plan.len() as u64).to_le_bytes())?;
        file.write_all(&(metadata.len() as u64).to_le_bytes())?;
        for (k, v) in metadata {
            write_str(&mut file, k)?;
            file.write_all(&v.type_id().to_le_bytes())?;
            v.write(&mut file)?;
        }
        let mut offset = 0u64;
        for t in &plan {
            write_str(&mut file, &t.name)?;
            file.write_all(&(t.dims.len() as u32).to_le_bytes())?;
            for &d in t.dims.iter().rev() {
                file.write_all(&d.to_le_bytes())?;
            }
            file.write_all(&t.dtype_id.to_le_bytes())?;
            file.write_all(&offset.to_le_bytes())?;
            offset += (t.nbytes + 31) / 32 * 32;
        }
        // Pad to alignment before the data section.
        file.flush()?;
        let pos = file.get_mut().stream_position()?;
        let pad = (32 - (pos % 32)) % 32;
        file.write_all(&vec![0u8; pad as usize])?;
        Ok(Self { file, plan, at: 0, pending: 0, pad: [0u8; 32] })
    }

    /// Append data for the plan's current tensor; must be called in plan
    /// order with exactly `nbytes` bytes in total (split calls are fine).
    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        let t = &self.plan[self.at];
        let remaining = t.nbytes as usize - self.pending;
        if bytes.len() > remaining {
            return Err(anyhow!(
                "{}: wrote more than planned ({} > {} remaining)",
                t.name,
                bytes.len(),
                remaining
            ));
        }
        self.file.write_all(bytes)?;
        self.pending += bytes.len();
        if self.pending == t.nbytes as usize {
            let pad = ((32 - (t.nbytes % 32)) % 32) as usize;
            self.file.write_all(&self.pad[..pad])?;
            self.pending = 0;
            self.at += 1;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.at != self.plan.len() {
            return Err(anyhow!(
                "incomplete write: {} of {} tensors",
                self.at,
                self.plan.len()
            ));
        }
        self.file.flush()?;
        Ok(())
    }

    pub fn plan(&self) -> &[TPlan] {
        &self.plan
    }
    pub fn at(&self) -> usize {
        self.at
    }
}
