//! Minimal safetensors reader: single file, or a sharded repo via
//! `model.safetensors.index.json`.  Tensors are read by absolute byte range
//! so huge tensors (the 384 M-row engram tables) can be streamed in row
//! chunks without mapping the whole file.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::dtypes::StDType;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: StDType,
    pub shape: Vec<usize>,
    /// Absolute file offsets of the raw data.
    pub start: u64,
    pub end: u64,
}

impl TensorMeta {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    /// Byte length of one leading (row-major) row; only valid when the shape
    /// is 2+ dimensional and every row is element-packed uniformly.
    pub fn row_bytes(&self) -> u64 {
        let per_row: usize = self.shape[1..].iter().product::<usize>();
        (per_row * self.dtype.element_size()) as u64
    }
}

pub struct Model {
    files: Vec<std::fs::File>,
    tensors: HashMap<String, (usize, TensorMeta)>,
    names_sorted: Vec<String>,
}

fn parse_header(path: &Path) -> Result<(u64, HashMap<String, (StDType, Vec<usize>, u64, u64)>)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb)?;
    let hlen = u64::from_le_bytes(lenb);
    let mut hb = vec![0u8; hlen as usize];
    f.read_exact(&mut hb)?;
    let hdr: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hb)
        .with_context(|| format!("safetensors header of {}", path.display()))?;
    let data_start = 8 + hlen;
    let mut out = HashMap::new();
    for (name, meta) in hdr {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow!("{}: tensor {name} missing dtype", path.display()))?;
        let dtype = StDType::parse(dtype)?;
        let shape: Vec<usize> = meta
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|v| v.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
            .unwrap_or_default();
        let off = meta.get("data_offsets").and_then(|s| s.as_array()).map(|a| a.to_vec());
        let off = match off {
            Some(o) if o.len() == 2 => (o[0].as_u64().unwrap_or(0), o[1].as_u64().unwrap_or(0)),
            _ => return Err(anyhow!("{}: tensor {name} missing data_offsets", path.display())),
        };
        out.insert(
            name,
            (dtype, shape, data_start + off.0, data_start + off.1),
        );
    }
    Ok((data_start, out))
}

impl Model {
    /// `path`: a `.safetensors` file, or a directory containing one or more
    /// (sharded via `model.safetensors.index.json`).
    pub fn open(path: &Path) -> Result<Self> {
        let mut shard_paths: Vec<PathBuf> = Vec::new();
        let mut weight_map: HashMap<String, String> = HashMap::new();
        if path.is_file() {
            shard_paths.push(path.to_path_buf());
        } else {
            let index = path.join("model.safetensors.index.json");
            if index.exists() {
                let idx: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&index)?).context("read index.json")?;
                let wm = idx
                    .get("weight_map")
                    .and_then(|w| w.as_object())
                    .ok_or_else(|| anyhow!("index.json has no weight_map"))?;
                for (name, shard) in wm {
                    weight_map.insert(
                        name.clone(),
                        shard.as_str().ok_or_else(|| anyhow!("bad weight_map value"))?.to_string(),
                    );
                }
            }
            let mut seen = std::collections::BTreeSet::new();
            if weight_map.is_empty() {
                for e in std::fs::read_dir(path)? {
                    let e = e?;
                    if e.path().extension().and_then(|e| e.to_str()) == Some("safetensors") {
                        seen.insert(e.path());
                    }
                }
            } else {
                for shard in weight_map.values() {
                    seen.insert(path.join(shard));
                }
            }
            shard_paths.extend(seen);
        }

        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for (idx, sp) in shard_paths.iter().enumerate() {
            let (_, hdr) = parse_header(sp)?;
            files.push(std::fs::File::open(sp)?);
            for (name, (dtype, shape, start, end)) in hdr {
                tensors.insert(name, (idx, TensorMeta { dtype, shape, start, end }));
            }
        }
        if let Some(wm) = if weight_map.is_empty() { None } else { Some(weight_map) } {
            // Sharded: restrict to tensors the index actually references.
            tensors.retain(|name, _| wm.contains_key(name));
        }
        let mut names_sorted: Vec<String> = tensors.keys().cloned().collect();
        names_sorted.sort();
        Ok(Self { files, tensors, names_sorted })
    }

    pub fn names(&self) -> &[String] {
        &self.names_sorted
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta> {
        self.tensors
            .get(name)
            .map(|(_, m)| m)
            .ok_or_else(|| anyhow!("tensor {name:?} not found"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Read the whole raw tensor (up to ~1 GB in practice; huge tensors go
    /// through `read_rows`).
    pub fn read_tensor(&mut self, name: &str) -> Result<(StDType, Vec<usize>, Vec<u8>)> {
        let (si, m) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name:?} not found"))?
            .clone();
        let bytes = read_range(&mut self.files[si], m.start, m.end)?;
        Ok((m.dtype, m.shape, bytes))
    }

    /// Read rows `[from, to)` of a row-major tensor.
    pub fn read_rows(&mut self, name: &str, from: usize, to: usize) -> Result<(StDType, Vec<usize>, Vec<u8>)> {
        let (si, m) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name:?} not found"))?
            .clone();
        if m.shape.len() < 2 {
            return Err(anyhow!("{name}: read_rows needs a 2D+ tensor"));
        }
        let rb = m.row_bytes();
        let lo = m.start + (from as u64) * rb;
        let hi = m.start + (to as u64) * rb;
        let bytes = read_range(&mut self.files[si], lo, hi)?;
        Ok((m.dtype, vec![to - from, m.shape[1]], bytes))
    }
}

fn read_range(f: &mut std::fs::File, lo: u64, hi: u64) -> Result<Vec<u8>> {
    let len = (hi - lo) as usize;
    f.seek(SeekFrom::Start(lo))?;
    let mut v = vec![0u8; len];
    f.read_exact(&mut v)?;
    Ok(v)
}
