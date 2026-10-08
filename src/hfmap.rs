//! HuggingFace → GGUF mapping: config.json fields, tensor names, and the
//! DeepSeek-V4.1 engram hash constants.  Everything here mirrors the real
//! `vcruz305` `deepseek41` GGUFs (parsed from the published file) and the
//! joshua loader's expectations (`quantized_deepseek4.rs`).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::ggufw::V;

pub const MULTIPLIERS_ASSET: &[u8] = include_bytes!("../assets/engram_constants.json");
pub const TOKEN_MAP_ASSET: &[u8] = include_bytes!("../assets/token_map_v41_flash.bin");
/// The V4 / V4.1-Flash chat template, as embedded in the published
/// `deepseek41` GGUFs (the HF repos ship none).
pub const CHAT_TEMPLATE_ASSET: &[u8] = include_bytes!("../assets/chat_template_v41.jinja");

pub struct Config {
    pub arch: &'static str,
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub n_expert: usize,
    pub n_hash_layers: usize,
    pub qblock: (usize, usize),
    pub head_dim: usize,
    pub engram_layer_ids: Vec<usize>,
    pub engram_vocab_size: Option<u64>,
    pub engram_n_heads: usize,
    pub engram_max_ngram: usize,
    pub engram_pad_token_id: i64,
    pub engram_compressed_vocab: Option<u64>,
    pub pad_token_id: i64,
    pub has_engram: bool,
    raw: Value,
}

impl Config {
    pub fn load(v: Value) -> Result<Self> {
        // V4.1 nests the text params under text_config; V4 keeps them flat.
        let tc = v.get("text_config").cloned().unwrap_or(Value::Null);
        let get = |k: &str| tc.get(k).or_else(|| v.get(k)).cloned().unwrap_or(Value::Null);
        let num = |k: &str| -> Result<i64> {
            get(k)
                .as_i64()
                .or_else(|| get(k).as_f64().map(|f| f as i64))
                .ok_or_else(|| anyhow!("config.json: missing numeric field {k:?}"))
        };
        // Arch comes from the TOP-LEVEL model_type: the text_config's own
        // model_type ("deepseek_v41_text") is a submodule tag and must not
        // shadow it.  Accept the submodule spellings defensively.
        let model_type = v
            .get("model_type")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let arch: &'static str = match model_type.as_str() {
            "deepseek_v41" | "deepseek_v41_text" => "deepseek41",
            "deepseek_v4" | "deepseek_v4_text" => "deepseek4",
            other => {
                bail!("config.json: model_type {other:?} is not deepseek_v4 / deepseek_v41")
            }
        };
        let engram_layer_ids: Vec<usize> = match get("engram_layer_ids").as_array() {
            Some(a) => a.iter().filter_map(|x| x.as_i64()).map(|x| x as usize).collect(),
            None => Vec::new(),
        };
        let has_engram = arch == "deepseek41" && !engram_layer_ids.is_empty();
        let (qr, qc) = match v.get("quantization_config").and_then(|q| q.get("weight_block_size")) {
            Some(Value::Array(b)) if b.len() >= 2 => (
                b[0].as_u64().unwrap_or(128) as usize,
                b[1].as_u64().unwrap_or(128) as usize,
            ),
            _ => (128, 128),
        };
        Ok(Self {
            arch,
            n_layer: num("num_hidden_layers")? as usize,
            hidden: num("hidden_size")? as usize,
            vocab: num("vocab_size")? as usize,
            n_expert: num("n_routed_experts")? as usize,
            n_hash_layers: num("num_hash_layers").unwrap_or(0) as usize,
            qblock: (qr, qc),
            head_dim: num("head_dim")? as usize,
            engram_layer_ids: engram_layer_ids.clone(),
            engram_vocab_size: get("engram_vocab_size").as_u64(),
            engram_n_heads: num("engram_n_heads").unwrap_or(0) as usize,
            engram_max_ngram: num("engram_max_ngram_size").unwrap_or(0) as usize,
            engram_pad_token_id: num("engram_pad_token_id").unwrap_or(2),
            engram_compressed_vocab: get("engram_compressed_vocab_size").as_u64(),
            pad_token_id: num("pad_token_id").unwrap_or(2),
            has_engram,
            raw: v,
        })
    }

    pub fn get(&self, k: &str) -> Value {
        self.raw
            .get("text_config")
            .and_then(|t| t.get(k))
            .or_else(|| self.raw.get(k))
            .cloned()
            .unwrap_or(Value::Null)
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let mut f = 41u64;
    while f * f <= n {
        if n % f == 0 || n % (f + 2) == 0 {
            return false;
        }
        f += 6;
    }
    true
}

fn next_prime(start: u64, seen: &mut std::collections::HashSet<u64>) -> u64 {
    let mut c = start + 1;
    while !is_prime(c) || seen.contains(&c) {
        c += 1;
    }
    seen.insert(c);
    c
}

/// The multipliers shipped in the real `deepseek41` GGUFs (numpy
/// `default_rng(10007 * layer_id)`, which this crate does not reimplement).
/// The values depend only on (layer_ids, max_ngram_size, compressed vocab
/// size), so they are stable across the V4.1-Flash checkpoint family.
pub fn default_multipliers() -> Result<Vec<u64>> {
    let v: Value = serde_json::from_slice(MULTIPLIERS_ASSET)?;
    let m = v
        .get("multipliers")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow!("engram asset: no multipliers"))?;
    Ok(m.iter().filter_map(|x| x.as_u64()).collect())
}

/// Load a token map: explicit override, else the embedded V4.1-Flash asset.
pub fn load_token_map(path: Option<&std::path::Path>) -> Result<Vec<i32>> {
    let bytes = match path {
        Some(p) => std::fs::read(p)?,
        None => TOKEN_MAP_ASSET.to_vec(),
    };
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        v.push(i32::from_le_bytes(c.try_into().unwrap()));
    }
    Ok(v)
}

/// Compute the engram primes + offsets exactly like the reference converter:
/// for every (layer, n-gram size) pair, `n_heads` fresh primes just above
/// `engram_vocab_size - 1`, all drawn from one shared `seen` set; offsets
/// are the per-layer cumulative sums in that same order.
pub fn compute_primes_offsets(cfg: &Config) -> Result<(Vec<u64>, Vec<u64>)> {
    let vocab = cfg
        .engram_vocab_size
        .ok_or_else(|| anyhow!("config.json: engram layers present but engram_vocab_size missing"))?;
    let n_cols = (cfg.engram_max_ngram - 1) * cfg.engram_n_heads;
    let mut seen = std::collections::HashSet::new();
    let mut primes = Vec::new();
    for _ in &cfg.engram_layer_ids {
        for _ in 0..(cfg.engram_max_ngram - 1) {
            let mut current = vocab - 1;
            for _ in 0..cfg.engram_n_heads {
                current = next_prime(current, &mut seen);
                primes.push(current);
            }
        }
    }
    // Exclusive per-layer cumsum: bucket k starts where the previous bucket
    // ended, and each layer restarts at 0 (matching the published GGUF).
    let mut offsets = Vec::with_capacity(primes.len());
    let mut acc = 0u64;
    for (i, p) in primes.iter().enumerate() {
        if i % n_cols == 0 {
            acc = 0;
        }
        offsets.push(acc);
        acc += p;
    }
    Ok((primes, offsets))
}

pub struct TokenizerMeta {
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub token_types: Vec<i32>,
    pub chat_template: Option<String>,
    /// `tokenizer.ggml.pre` (the real files carry `joyai-llm`).
    pub pre: String,
}

pub fn load_tokenizer(dir: &std::path::Path, unk_id: i64, cfg: &Config) -> Result<TokenizerMeta> {
    let tj: Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer.json"))?)
        .context("read tokenizer.json")?;
    let vocab = tj
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("tokenizer.json: no model.vocab"))?;
    // Base vocab + added_tokens form the full id space: the official file
    // has 128 000 base entries and 1 283 added tokens (ids 0..129 279;
    // three ids are shared with base rows by content), totaling the
    // model's 129 280 rows. An added token replaces a base entry with the
    // same id (HF's load order).
    let mut pairs: Vec<(usize, String)> = vocab
        .iter()
        .filter_map(|(t, id)| id.as_u64().map(|id| (id as usize, t.to_string())))
        .collect();
    if let Some(added) = tj.get("added_tokens").and_then(|a| a.as_array()) {
        for t in added {
            if let (Some(id), Some(c)) = (
                t.get("id").and_then(|i| i.as_u64()),
                t.get("content").and_then(|c| c.as_str()),
            ) {
                pairs.retain(|(i, _)| *i != id as usize);
                pairs.push((id as usize, c.to_string()));
            }
        }
    }
    pairs.sort();
    let n = pairs.len();
    let tokens: Vec<String> = pairs.iter().map(|(i, t)| t.clone()).collect();

    let mut specials: std::collections::HashMap<i64, bool> = std::collections::HashMap::new();
    if let Some(added) = tj.get("added_tokens").and_then(|a| a.as_array()) {
        for t in added {
            if let (Some(id), sp) = (t.get("id").and_then(|i| i.as_i64()), t.get("special")) {
                specials.insert(id, sp.map(|s| s.as_bool().unwrap_or(false)).unwrap_or(false));
            }
        }
    }
    let token_types: Vec<i32> = (0..n)
        .map(|i| {
            if specials.get(&(i as i64)).copied().unwrap_or(false) {
                3
            } else if i as i64 == unk_id {
                2
            } else {
                1
            }
        })
        .collect();

    let merges = tj
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|m| m.as_array())
        .map(|a| {
            a.iter()
                .map(|m| match m {
                    Value::String(s) => s.clone(),
                    Value::Array(p) => p
                        .iter()
                        .filter_map(|x| x.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    _ => String::new(),
                })
                .collect()
        })
        .unwrap_or_default();

    let chat_template = std::fs::read(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|tc| tc.get("chat_template").and_then(|c| c.as_str()).map(String::from));

    let _ = cfg;
    Ok(TokenizerMeta { tokens, merges, token_types, chat_template, pre: "joyai-llm".to_string() })
}

/// Build the full GGUF metadata list.
pub fn build_metadata(
    cfg: &Config,
    tok: Option<&TokenizerMeta>,
    engram_present: bool,
    token_map: &[i32],
    multipliers: &[u64],
    primes: &[u64],
    offsets: &[u64],
) -> Vec<(String, V)> {
    let a = cfg.arch;
    let key = |s: &str| format!("{a}.{s}");
    let name = match a {
        "deepseek41" => "DeepSeek V4.1 Flash",
        _ => "DeepSeek V4 Flash",
    };
    let num = |k: &str, default: i64| cfg.get(k).as_i64().unwrap_or(default);
    let flt = |k: &str| cfg.get(k).as_f64().unwrap_or(0.0);

    let mut md: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::Str(a.into())),
        ("general.name".into(), V::Str(name.into())),
        ("general.license".into(), V::Str("mit".into())),
        (key("vocab_size"), V::U32(cfg.vocab as u32)),
        (key("block_count"), V::U32(cfg.n_layer as u32)),
        (key("context_length"), V::U32(cfg.get("max_position_embeddings").as_u64().unwrap_or(0) as u32)),
        (key("embedding_length"), V::U32(cfg.hidden as u32)),
        (key("attention.head_count"), V::U32(num("num_attention_heads", 0) as u32)),
        (key("attention.head_count_kv"), V::U32(num("num_key_value_heads", 1) as u32)),
        (key("attention.layer_norm_rms_epsilon"), V::F32(flt("rms_norm_eps") as f32)),
        (key("attention.q_lora_rank"), V::U32(num("q_lora_rank", 0) as u32)),
        (key("attention.key_length"), V::U32(cfg.head_dim as u32)),
        (key("attention.value_length"), V::U32(cfg.head_dim as u32)),
        (key("rope.dimension_count"), V::U32(num("qk_rope_head_dim", 0) as u32)),
        (key("attention.compress_ratios"), V::Arr(cfg.get("compress_ratios").as_array().map(|r| r.iter().filter_map(|x| x.as_i64()).map(|x| V::I32(x as i32)).collect()).unwrap_or_default())),
        (key("attention.compress_rope_freq_base"), V::F32(flt("compress_rope_theta") as f32)),
        (key("attention.sliding_window"), V::U32(num("sliding_window", 128) as u32)),
        (key("attention.output_group_count"), V::U32(num("o_groups", 8) as u32)),
        (key("attention.output_lora_rank"), V::U32(num("o_lora_rank", 1024) as u32)),
        (key("attention.indexer.head_count"), V::U32(num("index_n_heads", 64) as u32)),
        (key("attention.indexer.key_length"), V::U32(num("index_head_dim", 128) as u32)),
        (key("attention.indexer.top_k"), V::U32(num("index_topk", 512) as u32)),
        (key("hyper_connection.count"), V::U32(num("hc_mult", 4) as u32)),
        (key("hyper_connection.sinkhorn_iterations"), V::U32(num("hc_sinkhorn_iters", 20) as u32)),
        (key("hyper_connection.epsilon"), V::F32(flt("hc_eps") as f32)),
        (key("expert_count"), V::U32(cfg.n_expert as u32)),
        (key("expert_used_count"), V::U32(num("num_experts_per_tok", 6) as u32)),
        (key("expert_feed_forward_length"), V::U32(num("moe_intermediate_size", 0) as u32)),
        (key("expert_shared_count"), V::U32(num("n_shared_experts", 1) as u32)),
        (key("expert_weights_scale"), V::F32(flt("routed_scaling_factor") as f32)),
        (key("expert_weights_norm"), V::Bool(cfg.get("norm_topk_prob").as_bool().unwrap_or(false))),
        (
            key("expert_gating_func"),
            V::U32(match cfg.get("scoring_func").as_str() {
                Some("softmax") => 1,
                Some("sigmoid") => 2,
                _ => 4, // sqrt(softplus), the V4/V4.1 default
            }),
        ),
        (key("hash_layer_count"), V::U32(cfg.n_hash_layers as u32)),
        (key("rope.freq_base"), V::F32(flt("rope_theta") as f32)),
    ];
    if let Some(limit) = cfg.get("swiglu_limit").as_f64() {
        md.push((key("swiglu_clamp_exp"), V::Arr(vec![V::F32(limit as f32); cfg.n_layer])));
        md.push((key("swiglu_clamp_shexp"), V::Arr(vec![V::F32(limit as f32); cfg.n_layer])));
    }
    // YaRN.
    let rs = cfg.get("rope_scaling");
    if rs.get("rope_type").and_then(|r| r.as_str()) == Some("yarn") {
        md.push((key("rope.scaling.type"), V::Str("yarn".into())));
        md.push((key("rope.scaling.factor"), V::F32(rs.get("factor").and_then(|f| f.as_f64()).unwrap_or(16.0) as f32)));
        md.push((key("rope.scaling.original_context_length"), V::U32(rs.get("original_max_position_embeddings").and_then(|f| f.as_u64()).unwrap_or(65536) as u32)));
        md.push((key("rope.scaling.yarn_beta_fast"), V::F32(rs.get("beta_fast").and_then(|f| f.as_f64()).unwrap_or(32.0) as f32)));
        md.push((key("rope.scaling.yarn_beta_slow"), V::F32(rs.get("beta_slow").and_then(|f| f.as_f64()).unwrap_or(1.0) as f32)));
    }
    // Engram.
    if cfg.has_engram && engram_present {
        md.push((key("engram.head_count"), V::U32(cfg.engram_n_heads as u32)));
        md.push((key("engram.key_length"), V::U32(num("engram_head_dim", 0) as u32)));
        md.push((key("engram.max_ngram_size"), V::U32(cfg.engram_max_ngram as u32)));
        md.push((key("engram.layer_ids"), V::Arr(cfg.engram_layer_ids.iter().map(|&l| V::I32(l as i32)).collect())));
        md.push((key("engram.multipliers"), V::Arr(multipliers.iter().map(|&m| V::U64(m)).collect())));
        md.push((key("engram.primes"), V::Arr(primes.iter().map(|&p| V::U64(p)).collect())));
        md.push((key("engram.offsets"), V::Arr(offsets.iter().map(|&o| V::U64(o)).collect())));
        md.push((key("engram.token_map"), V::Arr(token_map.iter().map(|&t| V::I32(t)).collect())));
        let pad = token_map
            .get(cfg.engram_pad_token_id.max(0) as usize)
            .copied()
            .unwrap_or(cfg.engram_pad_token_id as i32);
        md.push((key("engram.pad_id"), V::U32(pad as u32)));
    }
    // Tokenizer.
    if let Some(t) = tok {
        md.push(("tokenizer.ggml.model".into(), V::Str("gpt2".into())));
        md.push((
            "tokenizer.ggml.pre".into(),
            V::Str(t.pre.clone()),
        ));
        md.push(("tokenizer.chat_template".into(), V::Str(String::from_utf8_lossy(CHAT_TEMPLATE_ASSET).into_owned())));
        md.push(("tokenizer.ggml.tokens".into(), V::Arr(t.tokens.iter().map(|s| V::Str(s.clone())).collect())));
        md.push(("tokenizer.ggml.scores".into(), V::Arr(t.tokens.iter().map(|_| V::F32(0.0)).collect())));
        md.push(("tokenizer.ggml.token_type".into(), V::Arr(t.token_types.iter().map(|&x| V::I32(x)).collect())));
        md.push(("tokenizer.ggml.merges".into(), V::Arr(t.merges.iter().map(|s| V::Str(s.clone())).collect())));
        md.push(("tokenizer.ggml.bos_token_id".into(), V::U32(num("bos_token_id", 0) as u32)));
        md.push(("tokenizer.ggml.eos_token_id".into(), V::U32(num("eos_token_id", 1) as u32)));
        md.push(("tokenizer.ggml.padding_token_id".into(), V::U32(cfg.pad_token_id as u32)));
        md.push(("tokenizer.ggml.add_bos_token".into(), V::Bool(false)));
        md.push(("tokenizer.ggml.add_eos_token".into(), V::Bool(false)));
        if let Some(ct) = &t.chat_template {
            md.push(("tokenizer.chat_template".into(), V::Str(ct.clone())));
        }
    }
    md
}

/// What to do with one HuggingFace tensor.
pub enum Action {
    /// Emit under this GGUF name.
    Emit(String),
    /// Routed expert: (layer, expert id, which: 0 gate, 1 up, 2 down).
    Expert(usize, usize, u8),
    /// Vision tower / aligner / mmproj: intentionally not converted.
    SkipVision,
    /// MTP draft tensors: skipped unless --keep-mtp.
    SkipMtp,
    /// An fp8/fp4 `.scale` companion: consumed with its weight tensor.
    SkipScale,
}

/// Map an HF tensor name.  Returns None when the name is not part of this
/// architecture's surface (callers report it loudly).
pub fn map_name(name: &str, cfg: &Config, keep_mtp: bool) -> Option<Action> {
    use Action::*;
    if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") {
        return Some(SkipVision);
    }
    if name.ends_with(".scale") {
        return Some(SkipScale);
    }
    if let Some(rest) = name.strip_prefix("mtp.") {
        if keep_mtp {
            let (n, sub) = rest.split_once('.')?;
            let layer = cfg.n_layer + n.parse::<usize>().ok()?;
            return map_name(&format!("layers.{layer}.{sub}"), cfg, false)
                .or_else(|| Some(Emit(format!("blk.{layer}.{sub}"))));
        }
        return Some(SkipMtp);
    }
    match name {
        "embed.weight" => return Some(Action::Emit("token_embd.weight".into())),
        "head.weight" => return Some(Action::Emit("output.weight".into())),
        "norm.weight" => return Some(Action::Emit("output_norm.weight".into())),
        _ => {}
    }
    let rest = name.strip_prefix("layers.")?;
    let (idx, sub) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let p = format!("blk.{i}");
    Some(match sub {
        "attn_norm.weight" => Emit(format!("{p}.attn_norm.weight")),
        "ffn_norm.weight" => Emit(format!("{p}.ffn_norm.weight")),
        "attn.wq_a.weight" => Emit(format!("{p}.attn_q_a.weight")),
        "attn.wq_b.weight" => Emit(format!("{p}.attn_q_b.weight")),
        "attn.wkv.weight" => Emit(format!("{p}.attn_kv.weight")),
        "attn.wo_a.weight" => Emit(format!("{p}.attn_output_a.weight")),
        "attn.wo_b.weight" => Emit(format!("{p}.attn_output_b.weight")),
        "attn.q_norm.weight" => Emit(format!("{p}.attn_q_a_norm.weight")),
        "attn.kv_norm.weight" => Emit(format!("{p}.attn_kv_a_norm.weight")),
        "attn.attn_sink" => Emit(format!("{p}.attn_sinks.weight")),
        "attn.compressor.wkv.weight" => Emit(format!("{p}.attn_compressor_kv.weight")),
        "attn.compressor.wgate.weight" => Emit(format!("{p}.attn_compressor_gate.weight")),
        "attn.compressor.norm.weight" => Emit(format!("{p}.attn_compressor_norm.weight")),
        "attn.indexer.wq_b.weight" => Emit(format!("{p}.indexer.attn_q_b.weight")),
        "attn.indexer.weights_proj.weight" => Emit(format!("{p}.indexer.proj.weight")),
        "attn.indexer.wk.weight" => Emit(format!("{p}.indexer.attn_k.weight")),
        "attn.indexer.k_norm.weight" => Emit(format!("{p}.indexer.k_norm.weight")),
        "ffn.gate.weight" => Emit(format!("{p}.ffn_gate_inp.weight")),
        "ffn.gate.bias" => Emit(format!("{p}.exp_probs_b.bias")),
        "ffn.gate.bias_vl" => Emit(format!("{p}.exp_probs_b_vl.bias")),
        "hc_attn_fn" => Emit(format!("{p}.hc_attn_fn.weight")),
        "hc_attn_base" => Emit(format!("{p}.hc_attn_base.weight")),
        "hc_attn_scale" => Emit(format!("{p}.hc_attn_scale.weight")),
        "hc_ffn_fn" => Emit(format!("{p}.hc_ffn_fn.weight")),
        "hc_ffn_base" => Emit(format!("{p}.hc_ffn_base.weight")),
        "hc_ffn_scale" => Emit(format!("{p}.hc_ffn_scale.weight")),
        "engram.q_weight" => Emit(format!("{p}.engram_q.weight")),
        "engram.k_weight" => Emit(format!("{p}.engram_k.weight")),
        _ => {
            if let Some(e) = sub.strip_prefix("ffn.experts.") {
                let (e, which) = e.split_once('.')?;
                let eid: usize = e.parse().ok()?;
                let which = match which {
                    "w1.weight" => 0u8,
                    "w2.weight" => 2u8,
                    "w3.weight" => 1u8,
                    _ => return Some(SkipVision), // .scale tensors consumed implicitly
                };
                return Some(Expert(i, eid, which));
            }
            if let Some(rest) = sub.strip_prefix("ffn.shared_experts.") {
                match rest {
                    "w1.weight" => Emit(format!("{p}.ffn_gate_shexp.weight")),
                    "w2.weight" => Emit(format!("{p}.ffn_down_shexp.weight")),
                    "w3.weight" => Emit(format!("{p}.ffn_up_shexp.weight")),
                    _ => return Some(SkipVision), // scale
                }
            } else if let Some(rest) = sub.strip_prefix("engram.") {
                match rest {
                    "embed.weight" => Emit(format!("{p}.engram_embd.weight")),
                    "wkv.weight" => Emit(format!("{p}.engram_wkv.weight")),
                    _ => return Some(SkipVision), // engram embed.scale / wkv.scale
                }
            } else {
                return None;
            }
        }
    })
}
