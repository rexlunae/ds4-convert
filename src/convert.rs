//! Conversion orchestration: classify every HF tensor, build the GGUF plan,
//! then stream tensor-by-tensor (decode HF quantization, encode GGUF blocks)
//! without ever holding a whole huge tensor in memory.

use anyhow::{anyhow, bail, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::dtypes::{decode_to_f32, f32_from_e8m0, unpack_fp4, StDType};
use crate::ggufw::{TPlan, V, Writer};
use crate::hfmap::{self, Action, Config};
use crate::quant::Kind;
use crate::st::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Embd,
    Head,
    Dense,
    Down,
    ShexpDown,
    Experts,
    Router,
    HcFn,
    EngramTable,
    EngramQk,
    StaticF32,
}

pub struct Options {
    pub preset: String,
    pub keep_mtp: bool,
    pub fp4_high_first: bool,
    pub token_map: Option<PathBuf>,
    pub multipliers: Option<PathBuf>,
    pub engram_constants: Option<PathBuf>,
    pub allow_unmapped: bool,
    pub dry_run: bool,
    pub overrides: BTreeMap<&'static str, Kind>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            preset: "balanced".into(),
            keep_mtp: false,
            fp4_high_first: false,
            token_map: None,
            multipliers: None,
            engram_constants: None,
            allow_unmapped: false,
            dry_run: false,
            overrides: BTreeMap::new(),
        }
    }
}

impl Options {
    fn kind_for(&self, class: Class) -> Result<Kind> {
        // Preset grades follow the proven-good published DeepSeek-V4-Flash
        // Q2_K mix: dense attention and shared-expert paths at Q8_0, the
        // precision-critical router/compressor/indexer/hc/embedding paths at
        // F16, and only the big expert stacks aggressively quantized.  The
        // earlier all-Q2_K preset passed the tiny fixture but wrecked the
        // real model's coherence (flat logits, token salad).
        let preset = match class {
            Class::StaticF32 => Kind::F32,
            Class::Router => match self.preset.as_str() {
                "parity" => Kind::F32,
                _ => Kind::F16,
            },
            Class::Embd => match self.preset.as_str() {
                "parity" => Kind::F32,
                _ => Kind::F16,
            },
            Class::Head => match self.preset.as_str() {
                "size" => Kind::Q4K,
                "parity" => Kind::F32,
                _ => Kind::Q8_0,
            },
            Class::Down => match self.preset.as_str() {
                "size" => Kind::Q2K,
                "parity" => Kind::F32,
                _ => Kind::Q4K,
            },
            Class::ShexpDown => match self.preset.as_str() {
                "size" => Kind::Q2K,
                "parity" => Kind::F32,
                _ => Kind::Q8_0,
            },
            Class::Dense | Class::Experts => match self.preset.as_str() {
                "parity" => Kind::F32,
                "size" => Kind::Q2K,
                _ => match class {
                    Class::Dense => Kind::Q8_0,
                    _ => Kind::Q4K,
                },
            },
            Class::HcFn | Class::EngramTable | Class::EngramQk => match self.preset.as_str() {
                "parity" => Kind::F32,
                _ => Kind::F16,
            },
        };
        let slot: &'static str = match class {
            Class::Embd => "embd",
            Class::Head => "head",
            Class::Dense => "dense",
            Class::Down => "down",
            Class::ShexpDown => "shexp-down",
            Class::Experts => "experts",
            Class::Router => "router",
            Class::HcFn => "hc",
            Class::EngramTable => "engram-table",
            Class::EngramQk => "engram-qk",
            Class::StaticF32 => "static",
        };
        Ok(self.overrides.get(slot).copied().unwrap_or(preset))
    }
}

pub struct Summary {
    pub tensors_out: usize,
    pub bytes_out: u64,
    pub skipped_vision: usize,
    pub skipped_mtp: usize,
    pub skipped_scales: usize,
}

fn classify_out(name: &str) -> Class {
    if name == "token_embd.weight" {
        Class::Embd
    } else if name == "output.weight" {
        Class::Head
    } else if name.ends_with("down_shexp.weight") {
        Class::ShexpDown
    } else if name.ends_with("down_exps.weight") {
        Class::Down
    } else if name.ends_with("gate_exps.weight") || name.ends_with("up_exps.weight") || name.ends_with("gate_shexp.weight") || name.ends_with("up_shexp.weight") {
        Class::Experts
    } else if name.ends_with("ffn_gate_inp.weight") {
        Class::Router
    } else if name.ends_with("hc_attn_fn.weight") || name.ends_with("hc_ffn_fn.weight") {
        Class::HcFn
    } else if name.ends_with("engram_embd.weight") {
        Class::EngramTable
    } else if name.ends_with("engram_q.weight") || name.ends_with("engram_k.weight") {
        Class::EngramQk
    } else if name.ends_with("norm.weight")
        || name.ends_with("attn_sinks.weight")
        || name.ends_with("exp_probs_b.bias")
        || name.ends_with("exp_probs_b_vl.bias")
        || name.ends_with("_base.weight")
        || name.ends_with("_scale.weight")
    {
        Class::StaticF32
    } else {
        Class::Dense
    }
}

/// Dequantize a row chunk into f32.
///
/// `scale` carries the chunk's E8M0 scale bytes plus the global block shape:
/// `w[r][c] = raw[r][c] * 2^(scale[r0/br + r/br][c/bc] - 127)`.  fp4 inputs
/// hold two E2M1 values per byte along `c`.
fn dequant_chunk(
    dtype: StDType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    scale: Option<(&[u8], usize, usize, usize, usize, usize)>, // bytes, srows, scols, br, bc, row0
    fp4_high_first: bool,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(rows * cols);
    if dtype == StDType::I8 {
        // fp4-packed: bytes per row = cols / 2.
        if bytes.len() != rows * (cols / 2) {
            bail!("fp4 chunk: {} bytes for {rows}x{cols}", bytes.len());
        }
        let mut tmp = vec![0f32; cols];
        for r in 0..rows {
            unpack_fp4(&bytes[r * (cols / 2)..(r + 1) * (cols / 2)], &mut tmp, fp4_high_first);
            out.extend_from_slice(&tmp);
        }
    } else {
        out.extend(decode_to_f32(dtype, bytes)?.into_iter());
    }
    if let Some((sb, srows, scols, br, bc, row0)) = scale {
        if out.len() != rows * cols {
            bail!("dequant: {} values for {rows}x{cols}", out.len());
        }
        let sr0 = row0 / br;
        for r in 0..rows {
            let sr = sr0 + r / br;
            if sr >= srows {
                bail!("scale row {sr} out of range ({srows})");
            }
            for c in 0..cols {
                out[r * cols + c] *= f32_from_e8m0(sb[sr * scols + (c / bc).min(scols - 1)]);
            }
        }
    }
    Ok(out)
}

pub fn run(input: &Path, out: &Path, opts: &Options) -> Result<Summary> {
    let dir = if input.is_dir() { input.to_path_buf() } else { input.parent().unwrap().to_path_buf() };
    let cfg_v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let cfg = Config::load(cfg_v)?;
    let mut model = Model::open(input)?;

    // Classify every HF tensor.
    let mut skipped_vision = 0usize;
    let mut skipped_mtp = 0usize;
    let mut skipped_scales = 0usize;
    let mut direct: Vec<(String, String)> = Vec::new();
    let mut stacks: BTreeMap<(usize, u8), Vec<(usize, String)>> = BTreeMap::new();
    for name in model.names() {
        match hfmap::map_name(name, &cfg, opts.keep_mtp) {
            Some(Action::Emit(g)) => direct.push((g, name.clone())),
            Some(Action::Expert(layer, eid, which)) => {
                stacks.entry((layer, which)).or_default().push((eid, name.clone()));
            }
            Some(Action::SkipVision) => skipped_vision += 1,
            Some(Action::SkipScale) => skipped_scales += 1,
            Some(Action::SkipMtp) => skipped_mtp += 1,
            None => {
                if opts.allow_unmapped {
                    eprintln!("warning: unmapped tensor {name:?} (skipped)");
                } else {
                    bail!("unmapped tensor {name:?} (pass --allow-unmapped to skip)");
                }
            }
        }
    }
    // The engram constants ship only when the checkpoint actually carries
    // the engram tensors (REAP prunes them; the official checkpoint keeps
    // them).  A config that advertises engram layers without the tensors
    // converts as a no-engram model — joshua's loader runs V4.1 happily
    // without `.engram.layer_ids`.
    let engram_present = cfg.has_engram
        && cfg.engram_layer_ids.iter().all(|&l| {
            ["engram.embed.weight", "engram.wkv.weight", "engram.q_weight", "engram.k_weight"]
                .iter()
                .all(|t| model.has(&format!("layers.{l}.{t}")))
        });
    if cfg.has_engram && !engram_present {
        eprintln!(
            "note: config advertises engram layers {:?} but the checkpoint has no engram tensors; writing a no-engram model",
            cfg.engram_layer_ids
        );
    }

    // Tokenizer + engram constants.
    let tok = if dir.join("tokenizer.json").exists() {
        Some(hfmap::load_tokenizer(&dir, 0, &cfg)?)
    } else {
        None
    };
    let (token_map, multipliers, primes, offsets) = if engram_present {
        let token_map = hfmap::load_token_map(opts.token_map.as_deref())?;
        if token_map.len() != cfg.vocab {
            bail!(
                "token map has {} entries but vocab is {} (provide --token-map)",
                token_map.len(),
                cfg.vocab
            );
        }
        if let Some(expected) = cfg.engram_compressed_vocab {
            let distinct: std::collections::HashSet<i32> = token_map.iter().copied().collect();
            if distinct.len() as u64 != expected {
                bail!(
                    "token map collapses to {} ids, config says {expected} (wrong --token-map?)",
                    distinct.len()
                );
            }
        }
        let multipliers: Vec<u64> = match &opts.multipliers {
            Some(p) => {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                v.get("multipliers")
                    .and_then(|m| m.as_array())
                    .ok_or_else(|| anyhow!("--multipliers: no multipliers array"))?
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .collect()
            }
            None => hfmap::default_multipliers()?,
        };
        let (primes, offsets) = match &opts.engram_constants {
            Some(p) => {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                let gp = |k: &str| -> Vec<u64> {
                    v.get(k)
                        .and_then(|m| m.as_array())
                        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                        .unwrap_or_default()
                };
                (gp("primes"), gp("offsets"))
            }
            None => hfmap::compute_primes_offsets(&cfg)?,
        };
        let need_m = cfg.engram_layer_ids.len() * cfg.engram_max_ngram;
        let need_p = cfg.engram_layer_ids.len() * (cfg.engram_max_ngram - 1) * cfg.engram_n_heads;
        if multipliers.len() != need_m {
            bail!("engram multipliers: {} entries, need {need_m}", multipliers.len());
        }
        if primes.len() != need_p || offsets.len() != need_p {
            bail!("engram primes/offsets: {}/{} entries, need {need_p}", primes.len(), offsets.len());
        }
        (token_map, multipliers, primes, offsets)
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    let metadata = hfmap::build_metadata(&cfg, tok.as_ref(), engram_present, &token_map, &multipliers, &primes, &offsets);

    // Row chunking: multiples of 512 rows so every scale block (32) and the
    // quant blocks stay row-aligned.
    let chunk_rows = |cols: usize| -> usize {
        let rows = 256_000_000usize / cols.max(1);
        (rows / 512).max(1) * 512
    };

    // Plan.
    let mut plan: Vec<TPlan> = Vec::new();
    let mut kind_of: Vec<Kind> = Vec::new();
    for (gname, hf) in &direct {
        let m = model.meta(hf)?;
        let class = classify_out(gname);
        let kind = opts.kind_for(class)?;
        let nb = kind.tensor_bytes(&m.shape)?;
        plan.push(TPlan {
            name: gname.clone(),
            dims: m.shape.iter().map(|&d| d as u64).collect(),
            dtype_id: kind.type_id(),
            nbytes: nb,
        });
        kind_of.push(kind);
    }
    let mut stack_plan: BTreeMap<(usize, u8), usize> = BTreeMap::new();
    for (&(layer, which), entries) in &stacks {
        if entries.len() != cfg.n_expert {
            bail!(
                "layer {layer} expert stack (which {which}): {} of {} experts present",
                entries.len(),
                cfg.n_expert
            );
        }
        let sample = &entries[0].1;
        let m = model.meta(sample)?;
        if m.shape.len() != 2 {
            bail!("expert tensor {sample} is not 2D: {:?}", m.shape);
        }
        // fp4 experts are I8-packed along the input dim: two E2M1 values per
        // byte, so the stacked GGUF dims use the logical (unpacked) width.
        let logical0 = if m.dtype == StDType::I8 { m.shape[1] * 2 } else { m.shape[1] };
        let mut dims: Vec<u64> = vec![m.shape[0] as u64, logical0 as u64];
        dims.insert(0, cfg.n_expert as u64);
        let class = if which == 2 { Class::Down } else { Class::Experts };
        let kind = opts.kind_for(class)?;
        let nb = kind.tensor_bytes(&dims.iter().map(|&d| d as usize).collect::<Vec<_>>())?;
        let name = format!(
            "blk.{layer}.{}",
            match which {
                0 => "ffn_gate_exps.weight",
                1 => "ffn_up_exps.weight",
                _ => "ffn_down_exps.weight",
            }
        );
        stack_plan.insert((layer, which), plan.len());
        plan.push(TPlan { name, dims, dtype_id: kind.type_id(), nbytes: nb });
        kind_of.push(kind);
    }

    if opts.dry_run {
        let total: u64 = plan.iter().map(|t| t.nbytes).sum();
        println!("dry run: {} tensors, {total} bytes ({:.1} GB)", plan.len(), total as f64 / 1e9);
        for t in &plan {
            println!("  {:<42} {:>12} B  {:?}", t.name, t.nbytes, t.dims);
        }
        return Ok(Summary {
            tensors_out: plan.len(),
            bytes_out: total,
            skipped_vision,
            skipped_mtp,
            skipped_scales,
        });
    }

    // Write.
    let mut w = Writer::create(out, &metadata, plan)?;
    let mut done = 0usize;
    let total_tensors = direct.len() + stacks.len();

    // Chunked 2D writer: reads weight rows + scale rows, dequantizes, encodes.
    // The scale block shape is derived from the tensors themselves:
    // `br = rows / scale_rows`, `bc = logical_cols / scale_cols` — which
    // covers both the dense [32, 32] convention (wq_a: scale (40, 160) over
    // (1280, 5120)) and the experts' per-row fp4 scales ((2304, 160) over
    // (2304, 5120 logical)).
    fn write_chunked(
        w: &mut Writer,
        model: &mut Model,
        hf: &str,
        kind: Kind,
        fp4_high_first: bool,
        chunk_rows: usize,
    ) -> Result<()> {
        let sb_name = hf.replace(".weight", ".scale");
        let has_scale = model.has(&sb_name);
        let m = model.meta(hf)?.clone();
        let rows_total = m.shape[0];
        let byte_cols = m.shape[1];
        let logical = if m.dtype == StDType::I8 { byte_cols * 2 } else { byte_cols };
        let (srows, scols, br, bc) = if has_scale {
            let sm = model.meta(&sb_name)?.clone();
            let srows = sm.shape[0];
            let scols = *sm.shape.last().unwrap();
            if rows_total % srows != 0 || logical % scols != 0 {
                bail!(
                    "{hf}: scale {:?} does not tile weight {:?}",
                    sm.shape,
                    m.shape
                );
            }
            (srows, scols, rows_total / srows, logical / scols)
        } else {
            (0, 0, 0, 0)
        };
        let per = chunk_rows.min(rows_total.max(1));
        let mut r0 = 0usize;
        while r0 < rows_total {
            let r1 = (r0 + per).min(rows_total);
            let (dtype_s, _, wb) = model.read_rows(hf, r0, r1)?;
            let scale = if has_scale {
                let (_, _, sbv) = model.read_rows(&sb_name, r0 / br, r1.div_ceil(br))?;
                Some((sbv, r1.div_ceil(br) - r0 / br, scols, br, bc, r0))
            } else {
                None
            };
            let vals = match &scale {
                Some((sb, srows_c, scols_c, br_c, bc_c, r0_c)) => {
                    dequant_chunk(dtype_s, &wb, r1 - r0, logical, Some((sb, *srows_c, *scols_c, *br_c, *bc_c, *r0_c)), fp4_high_first)?
                }
                None => dequant_chunk(dtype_s, &wb, r1 - r0, logical, None, fp4_high_first)?,
            };
            let (_id, bytes) = crate::quant::encode(kind, &[r1 - r0, logical], &vals)?;
            w.push(&bytes)?;
            r0 = r1;
        }
        Ok(())
    }

    for (gname, hf) in &direct {
        let kind = kind_of[done];
        let m = model.meta(hf)?.clone();
        let class = classify_out(gname);
        // Stackedness is a property of the tensor NAME: the *_exps tensors
        // are the per-expert stacks (emitted by the stack loop); shared
        // experts (*/shexp) are plain 2-D tensors in the direct path.
        let is_stack = gname.ends_with("_exps.weight");
        let cols = *m.shape.last().ok_or_else(|| anyhow!("{hf}: empty"))?;
        let rows = m.shape[0];
        let fp4 = m.dtype == StDType::I8;
        let _ = (&cols, &rows, &fp4, &class);
        // Route ANY 2-D fp8/fp4 tensor through the chunked path — however
        // small — so its E8M0 scale companion is always applied. (A size
        // threshold here silently dropped scales from small fp8 tensors
        // like attn_kv [512, 5120], corrupting their magnitudes.)
        if m.shape.len() == 2 && (fp4 || m.dtype == StDType::F8E4M3) {
            // Chunked; the scale tiling is derived from the tensors inside
            // write_chunked.
            if is_stack {
                bail!("{gname}: unexpected stacked tensor among direct emits");
            }
            write_chunked(&mut w, &mut model, hf, kind, opts.fp4_high_first, chunk_rows(m.shape[1]))?;
        } else {
            // Whole-tensor path (small tensors: norms, biases, tiny tables).
            let (dtype_s, shape_c, bytes) = model.read_tensor(hf)?;
            let vals = dequant_chunk(dtype_s, &bytes, 0, 0, None, opts.fp4_high_first)
                .or_else(|_| decode_to_f32(dtype_s, &bytes))?;
            let (_id, bytes) = crate::quant::encode(kind, &shape_c, &vals)?;
            w.push(&bytes)?;
        }
        done += 1;
        if done % 64 == 0 {
            eprintln!("  {done} / {total_tensors} tensors");
        }
    }

    // Expert stacks: emit expert-major so each expert's encoded blocks are
    // contiguous inside the stacked tensor.
    for (&(layer, which), entries) in &stacks {
        let mut sorted = entries.clone();
        sorted.sort_by_key(|(eid, _)| *eid);
        let class = if which == 2 { Class::Down } else { Class::Experts };
        let kind = opts.kind_for(class)?;
        // IO sequential, CPU parallel: reads are cheap (5.9 MB/expert), the
        // k-quant encode is the cost. Encode batches of experts across
        // threads, then push their bytes in expert order.
        const BATCH: usize = 16;
        for batch in sorted.chunks(BATCH) {
            let mut batch_vals: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(batch.len());
            for (_eid, name) in batch {
                let m = model.meta(name)?.clone();
                let rows = m.shape[0];
                let logical = if m.dtype == StDType::I8 { m.shape[1] * 2 } else { m.shape[1] };
                let sb_name = name.replace(".weight", ".scale");
                let has_scale = model.has(&sb_name);
                let (srows, scols) = if has_scale {
                    let sm = model.meta(&sb_name)?;
                    (sm.shape[0], *sm.shape.last().unwrap())
                } else {
                    (0, 0)
                };
                if has_scale && (srows != rows || logical % scols != 0) {
                    bail!("{name}: scale ({srows}, {scols}) does not tile ({rows}, {logical})");
                }
                let (_dt, _, wb) = model.read_tensor(name)?;
                let (_, _, sb) = if has_scale {
                    model.read_tensor(&sb_name)?
                } else {
                    (StDType::F8E8M0, vec![], Vec::new())
                };
                let vals = dequant_chunk(
                    m.dtype,
                    &wb,
                    rows,
                    logical,
                    if has_scale { Some((&sb, srows, scols, rows / srows, logical / scols, 0)) } else { None },
                    opts.fp4_high_first,
                )?;
                batch_vals.push((rows, logical, vals));
            }
            let results: Vec<_> = std::thread::scope(|s| {
                batch_vals
                    .iter()
                    .map(|(rows, logical, vals)| {
                        s.spawn(move || crate::quant::encode(kind, &[*rows, *logical], vals))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().expect("encode thread panicked"))
                    .collect()
            });
            for res in results {
                let (_id, bytes) = res?;
                w.push(&bytes)?;
            }
        }
        done += 1;
    }

    let bytes_out: u64 = w.plan().iter().map(|t| t.nbytes).sum();
    let tensors_out = w.plan().len();
    w.finish()?;

    // tokenizer.json beside the GGUF (the engine's directory layout).
    if let Some(parent) = out.parent() {
        let _ = std::fs::copy(dir.join("tokenizer.json"), parent.join("tokenizer.json"));
    }

    Ok(Summary { tensors_out, bytes_out, skipped_vision, skipped_mtp, skipped_scales })
}