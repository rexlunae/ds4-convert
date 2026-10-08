//! HuggingFace safetensors dtypes → f32, and float encoders.

use anyhow::anyhow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E8M0,
    I8,
    U8,
    I32,
    I64,
    U64,
    F64,
}

impl StDType {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F64" => Self::F64,
            other => return Err(anyhow!("unsupported safetensors dtype {other:?}")),
        })
    }

    pub fn element_size(&self) -> usize {
        match self {
            Self::F8E4M3 | Self::F8E8M0 | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// FP8 E4M3 ("fn" flavour: no inf, max 448, e=15 & m!=0 is NaN).
pub fn f32_from_fp8_e4m3(b: u8) -> f32 {
    let sign = (b >> 7) as f32 * -2.0 + 1.0; // 0 → 1.0, 1 → -1.0
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    if e == 15 && m != 0 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        (m as f32) / 8.0 * 2.0f32.powi(-6)
    } else {
        (1.0 + m as f32 / 8.0) * 2.0f32.powi(e as i32 - 7)
    };
    sign * mag
}

/// E8M0 scale byte (ue8m0): pure exponent, value = 2^(b - 127).
pub fn f32_from_e8m0(b: u8) -> f32 {
    2.0f32.powi(b as i32 - 127)
}

/// FP4 E2M1 nibble: values {0, .5, 1, 1.5, 2, 3, 4, 6} with sign in bit 3.
fn e2m1(n: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = (n >> 3) as f32 * -2.0 + 1.0;
    sign * MAG[(n & 0x7) as usize]
}

/// Unpack FP4-packed data (two E2M1 values per byte) to `out.len()` f32s.
/// Default order is low-nibble-first (element 2i = low nibble); the
/// checkpoint convention can be flipped with `high_first`.
pub fn unpack_fp4(packed: &[u8], out: &mut [f32], high_first: bool) {
    assert!(out.len() <= packed.len() * 2, "fp4: {} elems > {} bytes", out.len(), packed.len());
    for (i, v) in out.iter_mut().enumerate() {
        let byte = packed[i / 2];
        let nib = if high_first != (i % 2 == 1) { byte & 0xF } else { byte >> 4 };
        *v = e2m1(nib);
    }
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    (((bits + 0x7FFF + lsb) >> 16) as u16) // round-to-nearest-even
}

/// Decode a whole (non-FP4) tensor to f32.
pub fn decode_to_f32(dtype: StDType, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let n = bytes.len() / dtype.element_size();
    let mut out = Vec::with_capacity(n);
    match dtype {
        StDType::F32 => out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        StDType::F16 => out.extend(bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())),
        StDType::BF16 => out.extend(bytes.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())))),
        StDType::F8E4M3 => out.extend(bytes.iter().map(|&b| f32_from_fp8_e4m3(b))),
        StDType::F8E8M0 => out.extend(bytes.iter().map(|&b| f32_from_e8m0(b))),
        StDType::I8 => out.extend(bytes.iter().map(|&b| b as i8 as f32)),
        StDType::U8 => out.extend(bytes.iter().map(|&b| b as f32)),
        StDType::I32 => return Err(anyhow!("I32 tensor needs the raw-copy path, not f32 decode")),
        StDType::I64 | StDType::U64 | StDType::F64 => {
            return Err(anyhow!("unsupported tensor dtype {dtype:?} for f32 decode"))
        }
    }
    Ok(out)
}
