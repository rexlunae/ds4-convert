//! Block encoders ported verbatim from the vendored candle implementations
//! (`quantized/k_quants.rs` + `quantized/utils.rs`), so anything this crate
//! writes decodes to exactly what joshua's decoders expect.  Encoders here
//! only choose the rounding; the block layouts are fixed by the format.

use anyhow::{anyhow, Result};

pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q2K,
    Q4K,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" | "f32" => Self::F32,
            "f16" | "f16" => Self::F16,
            "bf16" => Self::BF16,
            "q4_0" | "q40" => Self::Q4_0,
            "q8_0" | "q80" => Self::Q8_0,
            "q2_k" | "q2k" => Self::Q2K,
            "q4_k" | "q4k" => Self::Q4K,
            other => {
                return Err(anyhow!(
                    "unknown quant kind {other:?} (supported: f32 f16 bf16 q4_0 q8_0 q2_k q4_k)"
                ))
            }
        })
    }

    /// GGUF type id.
    pub fn type_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q4K => 12,
            Self::BF16 => 30,
        }
    }

    /// Elements per block (1 for the float types).
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q4K => QK_K,
            _ => 1,
        }
    }

    /// Bytes per block (float types: per element).
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q4K => 144,
        }
    }

    /// Total encoded bytes for candle-order dims.
    pub fn tensor_bytes(self, dims: &[usize]) -> Result<u64> {
        let n: usize = dims.iter().product();
        let bs = self.block_size();
        if bs > 1 {
            let inner = *dims.last().ok_or_else(|| anyhow!("empty dims"))?;
            if inner % bs != 0 {
                return Err(anyhow!(
                    "innermost dim {inner} is not a multiple of the {self:?} block {bs}"
                ));
            }
        }
        let elems = n as u64;
        Ok(elems / self.block_size() as u64 * self.block_bytes())
    }
}

fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// Port of candle `utils::make_qkx1_quants`: fit (scale, min) for an
/// unsigned `nmax`-level quantization of `x` with `ntry` refinement passes.
fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut l = vec![0u8; n];
    let min0 = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == min0 {
        return (0.0, 0.0);
    }
    let mut min = min0.min(0.0);
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0i32;
        let mut did_change = false;
        for (i, value) in x.iter().enumerate() {
            let li = nearest_int(iscale * (value - min)).clamp(0, nmax);
            let li = li as u8;
            if li != l[i] {
                l[i] = li;
                did_change = true;
            }
            sumlx += (value - min) * li as f32;
            suml2 += li as i32 * li as i32;
        }
        if suml2 == 0 {
            break;
        }
        scale = sumlx / suml2 as f32;
        let sum: f32 = x.iter().zip(l.iter()).map(|(xi, li)| xi - scale * *li as f32).sum();
        min = sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    (scale, -min)
}

fn f16_bytes(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

/// Encode candle-order `dims` from f32 (row-major, innermost last).
pub fn encode(kind: Kind, dims: &[usize], vals: &[f32]) -> Result<(u32, Vec<u8>)> {
    let n: usize = dims.iter().product();
    if vals.len() != n {
        return Err(anyhow!("encode: {} values for {} elements", vals.len(), n));
    }
    let _ = kind.tensor_bytes(dims)?;
    Ok(match kind {
        Kind::F32 => (0, vals.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Kind::F16 => (1, vals.iter().flat_map(|v| f16_bytes(*v)).collect()),
        Kind::BF16 => {
            let out: Vec<u8> = vals.iter().flat_map(|v| crate::dtypes::f32_to_bf16_bits(*v).to_le_bytes()).collect();
            (30, out)
        }
        Kind::Q8_0 => {
            let mut out = Vec::with_capacity(n / 32 * 34);
            for blk in vals.chunks_exact(32) {
                let amax = blk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                out.extend_from_slice(&f16_bytes(d));
                for &v in blk {
                    let q = if d != 0.0 { nearest_int(v / d).clamp(-127, 127) as i8 } else { 0 };
                    out.push(q as u8);
                }
            }
            (8, out)
        }
        Kind::Q4_0 => {
            // Verbatim candle BlockQ4_0::from_float.
            let mut out = Vec::with_capacity(n / 32 * 18);
            for blk in vals.chunks_exact(32) {
                let mut amax = 0f32;
                let mut max = 0f32;
                for &x in blk {
                    if amax < x.abs() {
                        amax = x.abs();
                        max = x;
                    }
                }
                let d = max / -8.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f16_bytes(d));
                for j in 0..16 {
                    let x0 = blk[j] * id;
                    let x1 = blk[16 + j] * id;
                    let xi0 = u8::min(15, (x0 + 8.5) as u8);
                    let xi1 = u8::min(15, (x1 + 8.5) as u8);
                    out.push(xi0 | (xi1 << 4));
                }
            }
            (2, out)
        }
        Kind::Q2K => {
            // Verbatim candle BlockQ2K::from_float (k_quants.rs:891).
            let mut out = Vec::with_capacity(n / QK_K * 84);
            for block in vals.chunks_exact(QK_K) {
                let mut scales = [0u8; 16];
                let mut d = 0f32;
                let mut dmin = 0f32;
                let mut mins = [0f32; 16];
                let mut fscales = [0f32; 16];
                for (j, sub) in block.chunks(16).enumerate() {
                    (fscales[j], mins[j]) = make_qkx1_quants(3, 5, sub);
                }
                let max_scale = fscales.iter().copied().fold(0.0, f32::max);
                let max_min = mins.iter().copied().fold(0.0, f32::max);
                if max_scale > 0.0 {
                    let iscale = 15.0 / max_scale;
                    for (j, s) in fscales.iter().enumerate().take(16) {
                        scales[j] = nearest_int(iscale * s) as u8;
                    }
                    d = max_scale / 15.0;
                }
                if max_min > 0.0 {
                    let iscale = 15.0 / max_min;
                    for (j, s) in mins.iter().enumerate().take(16) {
                        scales[j] |= (nearest_int(iscale * s) as u8) << 4;
                    }
                    dmin = max_min / 15.0;
                }
                out.extend_from_slice(&f16_bytes(d));
                out.extend_from_slice(&f16_bytes(dmin));
                let mut big_l = [0u8; QK_K];
                for j in 0..16 {
                    let dd = d * (scales[j] & 0xF) as f32;
                    if dd == 0.0 {
                        continue;
                    }
                    let dm = dmin * (scales[j] >> 4) as f32;
                    for ii in 0..16 {
                        big_l[16 * j + ii] = nearest_int((block[16 * j + ii] + dm) / dd).clamp(0, 3) as u8;
                    }
                }
                let mut qs = [0u8; 64];
                for j in (0..QK_K).step_by(128) {
                    for ll in 0..32 {
                        qs[j / 4 + ll] = big_l[j + ll]
                            | (big_l[j + ll + 32] << 2)
                            | (big_l[j + ll + 64] << 4)
                            | (big_l[j + ll + 96] << 6);
                    }
                }
                out.extend_from_slice(&scales);
                out.extend_from_slice(&qs);
            }
            (10, out)
        }
        Kind::Q4K => {
            // Verbatim candle BlockQ4K::from_float (k_quents:1542), with the
            // 6-bit scale packing read back by get_scale_min_k4.
            let mut out = Vec::with_capacity(n / QK_K * 144);
            for block in vals.chunks_exact(QK_K) {
                let mut scales = [0u8; 12];
                let mut d = 0f32;
                let mut dmin = 0f32;
                let mut mins = [0f32; 8];
                let mut fscales = [0f32; 8];
                for (j, sub) in block.chunks_exact(32).enumerate() {
                    (fscales[j], mins[j]) = make_qkx1_quants(15, 5, sub);
                }
                let max_scale = fscales.iter().copied().fold(0.0, f32::max);
                let max_min = mins.iter().copied().fold(0.0, f32::max);
                let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
                let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
                for j in 0..8 {
                    let ls = nearest_int(inv_scale * fscales[j]).min(63) as u8;
                    let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                    if j < 4 {
                        scales[j] = ls;
                        scales[j + 4] = lm;
                    } else {
                        scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                        scales[j - 4] |= (ls >> 4) << 6;
                        scales[j] |= (lm >> 4) << 6;
                    }
                }
                d = max_scale / 63.0;
                dmin = max_min / 63.0;
                out.extend_from_slice(&f16_bytes(d));
                out.extend_from_slice(&f16_bytes(dmin));
                out.extend_from_slice(&scales);
                let mut l = [0u8; QK_K];
                for j in 0..8 {
                    let (sc, m) = get_scale_min_k4(j, &scales);
                    let dd = d * sc as f32;
                    if dd == 0.0 {
                        continue;
                    }
                    let dm = dmin * m as f32;
                    for ii in 0..32 {
                        l[32 * j + ii] = nearest_int((block[32 * j + ii] + dm) / dd).clamp(0, 15) as u8;
                    }
                }
                let mut qs = [0u8; 128];
                for j in (0..QK_K).step_by(64) {
                    for ll in 0..32 {
                        let oi = (j / 64) * 32 + ll;
                        qs[oi] = l[j + ll] | (l[j + ll + 32] << 4);
                    }
                }
                out.extend_from_slice(&qs);
            }
            (12, out)
        }
    })
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Mirror decoders (for `--verify` and tests): decode block bytes to f32.
pub fn decode(kind: Kind, cols: usize, bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    match kind {
        Kind::F32 => out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        Kind::F16 => out.extend(bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())),
        Kind::BF16 => out.extend(bytes.chunks_exact(2).map(|c| crate::dtypes::bf16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())))),
        Kind::Q8_0 => {
            for blk in bytes.chunks_exact(34) {
                let d = half::f16::from_le_bytes(blk[0..2].try_into().unwrap()).to_f32();
                out.extend(blk[2..].iter().map(|&q| d * (q as i8 as f32)));
            }
        }
        Kind::Q4_0 => {
            for blk in bytes.chunks_exact(18) {
                let d = half::f16::from_le_bytes(blk[0..2].try_into().unwrap()).to_f32();
                for j in 0..16 {
                    out.push(((blk[2 + j] & 0x0F) as i8 - 8) as f32 * d);
                }
                for j in 0..16 {
                    out.push(((blk[2 + j] >> 4) as i8 - 8) as f32 * d);
                }
            }
        }
        Kind::Q2K => {
            for blk in bytes.chunks_exact(84) {
                let d = half::f16::from_le_bytes(blk[0..2].try_into().unwrap()).to_f32();
                let dmin = half::f16::from_le_bytes(blk[2..4].try_into().unwrap()).to_f32();
                let scales: [u8; 16] = blk[4..20].try_into().unwrap();
                let qs_all = &blk[20..];
                let mut chunk_out = [0f32; QK_K];
                let mut is = 0usize;
                for (qs, y_chunk) in qs_all.chunks_exact(32).zip(chunk_out.chunks_exact_mut(128)) {
                    for s in 0..4usize {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            y_chunk[32 * s + l] = dl * ((qs[l] >> (2 * s)) & 3) as f32 - ml;
                        }
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            y_chunk[32 * s + 16 + l] = dl * ((qs[16 + l] >> (2 * s)) & 3) as f32 - ml;
                        }
                    }
                }
                let _ = cols;
                out.extend_from_slice(&chunk_out);
            }
        }
        Kind::Q4K => {
            for blk in bytes.chunks_exact(144) {
                let d = half::f16::from_le_bytes(blk[0..2].try_into().unwrap()).to_f32();
                let dmin = half::f16::from_le_bytes(blk[2..4].try_into().unwrap()).to_f32();
                let scales: [u8; 12] = blk[4..16].try_into().unwrap();
                let qs = &blk[16..];
                // Unpack 6-bit scales/mins (8 each).
                let mut utmp = [0u32; 4];
                for i in 0..3 {
                    utmp[i] = u32::from_le_bytes(scales[4 * i..4 * i + 4].try_into().unwrap());
                }
                const KMASK1: u32 = 0x3f3f3f3f;
                const KMASK2: u32 = 0x0f0f0f0f;
                const KMASK3: u32 = 0x03030303;
                utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
                let uaux = utmp[1] & KMASK1;
                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[2] = uaux;
                utmp[0] &= KMASK1;
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                for i in 0..2 {
                    let b0 = utmp[i].to_le_bytes();
                    let b1 = utmp[i + 2].to_le_bytes();
                    for j in 0..4 {
                        sc[4 * i + j] = b0[j];
                        mn[4 * i + j] = b1[j];
                    }
                }
                // Unpack nibbles: elements 0..32 = low nibbles of qs[0..32], etc.
                let mut q = [0f32; QK_K];
                let mut a = 0usize;
                for j in 0..4 {
                    for l in 0..32 {
                        q[a + l] = (qs[j * 32 + l] & 0xF) as f32;
                    }
                    a += 32;
                    for l in 0..32 {
                        q[a + l] = (qs[j * 32 + l] >> 4) as f32;
                    }
                    a += 32;
                }
                for e in 0..QK_K {
                    out.push(d * sc[e / 32] as f32 * q[e] - dmin * mn[e / 32] as f32);
                }
            }
        }
    }
    Ok(())
}
