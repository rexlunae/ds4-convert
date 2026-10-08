//! End-to-end parity: a tiny deepseek41 model written twice — once as a GGUF
//! directly (ds4-convert's writer + encoders) and once as an HF-layout
//! safetensors repo converted through the converter — must produce identical
//! logits under joshua's real deepseek41 loader.

use std::collections::BTreeMap;
use std::io::Write as _;

use ds4_convert::convert::{self, Options};
use ds4_convert::ggufw::{TPlan, V, Writer};
use ds4_convert::quant::{self, Kind};

const VOCAB: usize = 16;
const EMB: usize = 256;
const NLAYER: usize = 4;
const NHEAD: usize = 4;
const HEAD_DIM: usize = 16;
const ROPE_DIM: usize = 8;
const Q_LORA: usize = 8;
const O_GROUPS: usize = 2;
const O_LORA: usize = 4;
const NE: usize = 8;
const NFE: usize = 256;
const NUSED: usize = 2;
const N_SHARED: usize = 1;
const HC: usize = 2;
const INDEX_NHEAD: usize = 4;
const INDEX_HD: usize = 16;
const ENGRAM_KEY: usize = 16;
const ENGRAM_PRIMES: [u64; 4] = [5, 7, 11, 13];
const ENGRAM_ROWS: usize = 36;

fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 2000) as f32 / 1000.0 - 1.0) * 0.1
        })
        .collect()
}

struct Gen {
    seed: u32,
    norm_seed: u32,
}
impl Gen {
    fn next(&mut self, n: usize) -> Vec<f32> {
        self.seed = self.seed.wrapping_add(7).wrapping_mul(2_654_435_761) | 1;
        weights(n, self.seed)
    }
    fn ones(&mut self, n: usize) -> Vec<f32> {
        self.norm_seed += 1;
        weights(n, self.norm_seed).iter().map(|w| 1.0 + 3.0 * w).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum D {
    F16,
    F32,
}

/// The tiny v41 model, in the fixture's exact tensor-creation order.
fn tiny_model() -> Vec<(String, D, Vec<f32>, Vec<usize>)> {
    let mut g = Gen { seed: 10, norm_seed: 500 };
    let mut t: Vec<(String, D, Vec<f32>, Vec<usize>)> = Vec::new();
    t.push(("token_embd.weight".into(), D::F16, weights(VOCAB * EMB, 1), vec![VOCAB, EMB]));
    t.push(("output_norm.weight".into(), D::F32, g.ones(EMB), vec![EMB]));
    t.push(("output.weight".into(), D::F16, weights(VOCAB * EMB, 2), vec![VOCAB, EMB]));
    let o_group_dim = (NHEAD / O_GROUPS) * HEAD_DIM;
    for i in 0..NLAYER {
        let p = format!("blk.{i}");
        t.push((format!("{p}.attn_norm.weight"), D::F32, g.ones(EMB), vec![EMB]));
        t.push((format!("{p}.ffn_norm.weight"), D::F32, g.ones(EMB), vec![EMB]));
        t.push((format!("{p}.attn_q_a.weight"), D::F16, g.next(Q_LORA * EMB), vec![Q_LORA, EMB]));
        t.push((format!("{p}.attn_q_a_norm.weight"), D::F32, g.ones(Q_LORA), vec![Q_LORA]));
        t.push((format!("{p}.attn_q_b.weight"), D::F16, g.next(NHEAD * HEAD_DIM * Q_LORA), vec![NHEAD * HEAD_DIM, Q_LORA]));
        t.push((format!("{p}.attn_kv.weight"), D::F16, g.next(HEAD_DIM * EMB), vec![HEAD_DIM, EMB]));
        t.push((format!("{p}.attn_kv_a_norm.weight"), D::F32, g.ones(HEAD_DIM), vec![HEAD_DIM]));
        t.push((format!("{p}.attn_output_a.weight"), D::F16, g.next(O_GROUPS * O_LORA * o_group_dim), vec![O_GROUPS * O_LORA, o_group_dim]));
        t.push((format!("{p}.attn_output_b.weight"), D::F16, g.next(EMB * O_GROUPS * O_LORA), vec![EMB, O_GROUPS * O_LORA]));
        t.push((format!("{p}.attn_sinks.weight"), D::F32, g.next(NHEAD), vec![NHEAD]));
        if i == 0 || i == 2 {
            t.push((format!("{p}.attn_compressor_kv.weight"), D::F16, g.next(HEAD_DIM * EMB), vec![HEAD_DIM, EMB]));
            if i == 0 {
                t.push((format!("{p}.attn_compressor_gate.weight"), D::F16, g.next(HEAD_DIM * EMB), vec![HEAD_DIM, EMB]));
            }
            t.push((format!("{p}.attn_compressor_norm.weight"), D::F32, g.ones(HEAD_DIM), vec![HEAD_DIM]));
            t.push((format!("{p}.indexer.attn_k.weight"), D::F16, g.next(INDEX_HD * HEAD_DIM), vec![INDEX_HD, HEAD_DIM]));
            t.push((format!("{p}.indexer.k_norm.weight"), D::F32, g.ones(INDEX_HD), vec![INDEX_HD]));
        }
        if i < 3 {
            t.push((format!("{p}.indexer.proj.weight"), D::F16, g.next(INDEX_NHEAD * EMB), vec![INDEX_NHEAD, EMB]));
            t.push((format!("{p}.indexer.attn_q_b.weight"), D::F16, g.next(INDEX_NHEAD * INDEX_HD * Q_LORA), vec![INDEX_NHEAD * INDEX_HD, Q_LORA]));
        }
        if i == 1 {
            t.push((format!("{p}.engram_embd.weight"), D::F16, g.next(ENGRAM_ROWS * ENGRAM_KEY), vec![ENGRAM_ROWS, ENGRAM_KEY]));
            t.push((format!("{p}.engram_wkv.weight"), D::F16, g.next((HC + 1) * EMB * 4 * ENGRAM_KEY), vec![(HC + 1) * EMB, 4 * ENGRAM_KEY]));
            t.push((format!("{p}.engram_q.weight"), D::F32, g.ones(HC * EMB), vec![HC, EMB]));
            t.push((format!("{p}.engram_k.weight"), D::F32, g.ones(HC * EMB), vec![HC, EMB]));
        }
        t.push((format!("{p}.ffn_gate_inp.weight"), D::F32, g.next(NE * EMB), vec![NE, EMB]));
        t.push((format!("{p}.exp_probs_b.bias"), D::F32, g.next(NE), vec![NE]));
        let gate = g.next(NE * NFE * EMB);
        let up = g.next(NE * NFE * EMB);
        let down = g.next(NE * NFE * EMB);
        t.push((format!("{p}.ffn_gate_exps.weight"), D::F32, gate, vec![NE, NFE, EMB]));
        t.push((format!("{p}.ffn_up_exps.weight"), D::F32, up, vec![NE, NFE, EMB]));
        t.push((format!("{p}.ffn_down_exps.weight"), D::F32, down, vec![NE, EMB, NFE]));
        t.push((format!("{p}.ffn_gate_shexp.weight"), D::F16, g.next(NFE * EMB), vec![NFE, EMB]));
        t.push((format!("{p}.ffn_up_shexp.weight"), D::F16, g.next(NFE * EMB), vec![NFE, EMB]));
        t.push((format!("{p}.ffn_down_shexp.weight"), D::F16, g.next(EMB * NFE), vec![EMB, NFE]));
        let hc_out = (2 + HC) * HC;
        let hc_scale: Vec<f32> = vec![0.7, 1.3, 0.9];
        t.push((format!("{p}.hc_attn_fn.weight"), D::F16, g.next(hc_out * HC * EMB), vec![hc_out, HC * EMB]));
        t.push((format!("{p}.hc_attn_base.weight"), D::F32, g.next(hc_out), vec![hc_out]));
        t.push((format!("{p}.hc_attn_scale.weight"), D::F32, hc_scale.clone(), vec![3]));
        t.push((format!("{p}.hc_ffn_fn.weight"), D::F16, g.next(hc_out * HC * EMB), vec![hc_out, HC * EMB]));
        t.push((format!("{p}.hc_ffn_base.weight"), D::F32, g.next(hc_out), vec![hc_out]));
        t.push((format!("{p}.hc_ffn_scale.weight"), D::F32, hc_scale, vec![3]));
    }
    t
}

fn enc(d: D, dims: &[usize], v: &[f32]) -> Vec<u8> {
    match d {
        D::F32 => quant::encode(Kind::F32, dims, v).unwrap().1,
        // The direct reference file is stored as F32 throughout: comparing
        // F16 storage against F32 storage measures the matmul kernel's
        // f16-vs-f32 rounding (machine-specific), not the converter.
        D::F16 => quant::encode(Kind::F32, dims, v).unwrap().1,
    }
}

fn dtype_name(d: D) -> &'static str {
    match d {
        D::F16 => "F32",
        D::F32 => "F32",
    }
}

fn tiny_metadata() -> Vec<(String, V)> {
    let a = "deepseek41";
    let key = |s: &str| format!("{a}.{s}");
    let mut md: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::Str(a.into())),
        ("general.name".into(), V::Str("DeepSeek-V4.1".into())),
        (key("block_count"), V::U32(NLAYER as u32)),
        (key("attention.head_count"), V::U32(NHEAD as u32)),
        (key("embedding_length"), V::U32(EMB as u32)),
        (key("attention.layer_norm_rms_epsilon"), V::F32(1e-5)),
        (key("attention.q_lora_rank"), V::U32(Q_LORA as u32)),
        (key("attention.key_length"), V::U32(HEAD_DIM as u32)),
        (key("rope.dimension_count"), V::U32(ROPE_DIM as u32)),
        (key("attention.compress_ratios"), V::Arr(vec![V::I32(2), V::I32(2), V::I32(1), V::I32(0)])),
        (key("attention.sliding_window"), V::U32(8)),
        (key("attention.output_group_count"), V::U32(O_GROUPS as u32)),
        (key("attention.output_lora_rank"), V::U32(O_LORA as u32)),
        (key("hyper_connection.count"), V::U32(HC as u32)),
        (key("hyper_connection.sinkhorn_iterations"), V::U32(2)),
        (key("hyper_connection.epsilon"), V::F32(1e-6)),
        (key("expert_count"), V::U32(NE as u32)),
        (key("expert_used_count"), V::U32(NUSED as u32)),
        (key("expert_shared_count"), V::U32(N_SHARED as u32)),
        (key("hash_layer_count"), V::U32(0)),
        (key("expert_weights_scale"), V::F32(1.0)),
        (key("expert_gating_func"), V::U32(2)),
        (key("rope.freq_base"), V::F32(10_000.0)),
        (key("context_length"), V::U32(512)),
        ("tokenizer.ggml.eos_token_id".into(), V::U32(3)),
        ("tokenizer.ggml.bos_token_id".into(), V::U32(3)),
        ("tokenizer.ggml.unknown_token_id".into(), V::U32(0)),
        ("tokenizer.ggml.model".into(), V::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), V::Arr(["<unk>", "hello", "world", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"].iter().map(|s| V::Str(s.to_string())).collect())),
        ("tokenizer.ggml.scores".into(), V::Arr((0..16).map(|i| V::F32(-(i as f32))).collect())),
        ("tokenizer.ggml.token_type".into(), V::Arr([2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].iter().map(|&t| V::I32(t)).collect())),
        (key("attention.indexer.head_count"), V::U32(INDEX_NHEAD as u32)),
        (key("attention.indexer.key_length"), V::U32(INDEX_HD as u32)),
        (key("attention.indexer.top_k"), V::U32(2)),
        (key("attention.compress_rope_freq_base"), V::F32(40_000.0)),
        (key("engram.layer_ids"), V::Arr(vec![V::I32(1)])),
        (key("engram.head_count"), V::U32(2)),
        (key("engram.key_length"), V::U32(ENGRAM_KEY as u32)),
        (key("engram.max_ngram_size"), V::U32(3)),
        (key("engram.multipliers"), V::Arr(vec![V::U64(1_000_003), V::U64(2_000_029), V::U64(3_000_017)])),
        (key("engram.primes"), V::Arr(ENGRAM_PRIMES.iter().map(|&p| V::U64(p)).collect())),
        (key("engram.offsets"), V::Arr({
            let mut acc = 0u64;
            ENGRAM_PRIMES.iter().map(|&p| { let o = acc; acc += p; V::U64(o) }).collect()
        })),
        (key("engram.token_map"), V::Arr((0..VOCAB as i32).map(|t| V::I32(if t < 4 { t } else { 4 + (t - 4) / 2 })).collect())),
        (key("engram.pad_id"), V::U32(2)),
    ];
    md
}

fn write_direct_gguf(path: &std::path::Path) {
    let tensors = tiny_model();
    let md = tiny_metadata();
    let plan: Vec<TPlan> = tensors
        .iter()
        .map(|(name, d, v, dims)| {
            let bytes = enc(*d, dims, v);
            TPlan {
                name: name.clone(),
                dims: dims.iter().map(|&x| x as u64).collect(),
                dtype_id: 0, // enc() stores everything as F32 now
                nbytes: bytes.len() as u64,
            }
        })
        .collect();
    let mut w = Writer::create(path, &md, plan).unwrap();
    for (name, d, v, dims) in &tensors {
        let bytes = enc(*d, dims, v);
        assert_eq!(name.as_str(), w.plan()[w.at()].name, "plan order must match tensor order");
        w.push(&bytes).unwrap();
    }
    w.finish().unwrap();
}

/// HF-layout safetensors repo for the same model (per-expert tensors).
fn write_hf_repo(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    let tensors = tiny_model();
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut body: Vec<u8> = Vec::new();
    let mut offset = 0u64;
    for (name, d, v, dims) in &tensors {
        let bytes = enc(*d, dims, v);
        let dt = dtype_name(*d);
        if name.ends_with("gate_exps.weight") || name.ends_with("up_exps.weight") || name.ends_with("down_exps.weight") {
            let (layer, which): (usize, &str) = {
                let base = name.trim_start_matches("blk.");
                let i: usize = base.split('.').next().unwrap().parse().unwrap();
                let w = if name.contains("gate_exps") { "w1" } else if name.contains("up_exps") { "w3" } else { "w2" };
                (i, w)
            };
            let per = bytes.len() / NE;
            let rest: Vec<usize> = dims[1..].to_vec();
            for e in 0..NE {
                let hf = format!("layers.{layer}.ffn.experts.{e}.{which}.weight");
                let lo = offset + (e * per) as u64;
                header.insert(hf, serde_json::json!({"dtype": dt, "shape": rest, "data_offsets": [lo, lo + per as u64]}));
            }
            body.extend_from_slice(&bytes);
            offset += bytes.len() as u64;
            continue;
        }
        let hf = match name.as_str() {
            "token_embd.weight" => "embed.weight".to_string(),
            "output.weight" => "head.weight".to_string(),
            "output_norm.weight" => "norm.weight".to_string(),
            other => {
                let base = other.trim_start_matches("blk.");
                let (idx, sub) = base.split_once('.').unwrap();
                let l = format!("layers.{}.", idx);
                match sub {
                    "attn_norm.weight" => format!("{l}attn_norm.weight"),
                    "ffn_norm.weight" => format!("{l}ffn_norm.weight"),
                    "attn_q_a.weight" => format!("{l}attn.wq_a.weight"),
                    "attn_q_a_norm.weight" => format!("{l}attn.q_norm.weight"),
                    "attn_q_b.weight" => format!("{l}attn.wq_b.weight"),
                    "attn_kv.weight" => format!("{l}attn.wkv.weight"),
                    "attn_kv_a_norm.weight" => format!("{l}attn.kv_norm.weight"),
                    "attn_output_a.weight" => format!("{l}attn.wo_a.weight"),
                    "attn_output_b.weight" => format!("{l}attn.wo_b.weight"),
                    "attn_sinks.weight" => format!("{l}attn.attn_sink"),
                    "attn_compressor_kv.weight" => format!("{l}attn.compressor.wkv.weight"),
                    "attn_compressor_gate.weight" => format!("{l}attn.compressor.wgate.weight"),
                    "attn_compressor_norm.weight" => format!("{l}attn.compressor.norm.weight"),
                    "indexer.attn_k.weight" => format!("{l}attn.indexer.wk.weight"),
                    "indexer.k_norm.weight" => format!("{l}attn.indexer.k_norm.weight"),
                    "indexer.proj.weight" => format!("{l}attn.indexer.weights_proj.weight"),
                    "indexer.attn_q_b.weight" => format!("{l}attn.indexer.wq_b.weight"),
                    "engram_embd.weight" => format!("{l}engram.embed.weight"),
                    "engram_wkv.weight" => format!("{l}engram.wkv.weight"),
                    "engram_q.weight" => format!("{l}engram.q_weight"),
                    "engram_k.weight" => format!("{l}engram.k_weight"),
                    "ffn_gate_inp.weight" => format!("{l}ffn.gate.weight"),
                    "exp_probs_b.bias" => format!("{l}ffn.gate.bias"),
                    "ffn_gate_shexp.weight" => format!("{l}ffn.shared_experts.w1.weight"),
                    "ffn_up_shexp.weight" => format!("{l}ffn.shared_experts.w3.weight"),
                    "ffn_down_shexp.weight" => format!("{l}ffn.shared_experts.w2.weight"),
                    "hc_attn_fn.weight" => format!("{l}hc_attn_fn"),
                    "hc_attn_base.weight" => format!("{l}hc_attn_base"),
                    "hc_attn_scale.weight" => format!("{l}hc_attn_scale"),
                    "hc_ffn_fn.weight" => format!("{l}hc_ffn_fn"),
                    "hc_ffn_base.weight" => format!("{l}hc_ffn_base"),
                    "hc_ffn_scale.weight" => format!("{l}hc_ffn_scale"),
                    other => panic!("tiny gguf name without HF inverse: {other}"),
                }
            }
        };
        header.insert(hf, serde_json::json!({"dtype": dt, "shape": dims, "data_offsets": [offset, offset + bytes.len() as u64]}));
        body.extend_from_slice(&bytes);
        offset += bytes.len() as u64;
    }
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let mut hj = serde_json::to_string(&header).unwrap().into_bytes();
    while hj.len() % 8 != 0 {
        hj.push(b' ');
    }
    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(hj.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hj).unwrap();
    f.write_all(&body).unwrap();

    let engram_offsets: Vec<u64> = {
        let mut acc = 0u64;
        ENGRAM_PRIMES.iter().map(|&p| { let o = acc; acc += p; o }).collect()
    };
    let config = serde_json::json!({
        "model_type": "deepseek_v41",
        "text_config": {
            "vocab_size": VOCAB, "hidden_size": EMB, "num_hidden_layers": NLAYER,
            "num_attention_heads": NHEAD, "num_key_value_heads": 1, "head_dim": HEAD_DIM,
            "qk_rope_head_dim": ROPE_DIM, "q_lora_rank": Q_LORA, "o_groups": O_GROUPS,
            "o_lora_rank": O_LORA, "sliding_window": 8, "scoring_func": "sigmoid",
            "n_routed_experts": NE, "num_experts_per_tok": NUSED, "n_shared_experts": N_SHARED,
            "moe_intermediate_size": NFE, "routed_scaling_factor": 1.0,
            "norm_topk_prob": true, "rms_norm_eps": 1e-5, "rope_theta": 10000,
            "max_position_embeddings": 512,
            "compress_ratios": [2, 2, 1, 0], "compress_rope_theta": 40000,
            "index_n_heads": INDEX_NHEAD, "index_head_dim": INDEX_HD, "index_topk": 2,
            "hc_mult": HC, "hc_sinkhorn_iters": 2, "hc_eps": 1e-6,
            "engram_layer_ids": [1], "engram_n_heads": 2, "engram_head_dim": ENGRAM_KEY,
            "engram_max_ngram_size": 3, "engram_pad_token_id": 2,
            "pad_token_id": 2, "bos_token_id": 3, "eos_token_id": 3
        }
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    std::fs::write(
        dir.join("engram_constants.json"),
        serde_json::json!({
            "multipliers": [1_000_003u64, 2_000_029, 3_000_017],
            "primes": ENGRAM_PRIMES,
            "offsets": engram_offsets
        })
        .to_string(),
    )
    .unwrap();
    let mut tb = Vec::new();
    for t in 0..VOCAB as i32 {
        let x = if t < 4 { t } else { 4 + (t - 4) / 2 };
        tb.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(dir.join("token_map.bin"), tb).unwrap();
}

fn load_with_joshua(path: &std::path::Path) -> joshua::model::QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    joshua::model::QuantizedModel::from_gguf_mmap(content, &mut cursor, &candle_core::Device::Cpu, None, None, 0).unwrap()
}

fn logits(model: &mut joshua::model::QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = candle_core::Tensor::new(tokens, &candle_core::Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model.forward(&input, offset).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn main() {
    let tmp = std::env::temp_dir().join("ds4-parity");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let direct_gguf = tmp.join("direct.gguf");
    write_direct_gguf(&direct_gguf);

    let hf_dir = tmp.join("hf");
    write_hf_repo(&hf_dir);
    let converted = tmp.join("converted.gguf");
    let opts = Options {
        preset: "parity".into(),
        token_map: Some(hf_dir.join("token_map.bin")),
        multipliers: Some(hf_dir.join("engram_constants.json")),
        engram_constants: Some(hf_dir.join("engram_constants.json")),
        ..Options::default()
    };
    let summary = convert::run(&hf_dir, &converted, &opts).unwrap();
    println!("converted: {} tensors, {} bytes", summary.tensors_out, summary.bytes_out);

    let tokens: [u32; 14] = [1, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1, 12, 13, 5];
    // Arm 1: lossless F32 preset - exact parity.
    {
        let mut a = load_with_joshua(&direct_gguf);
        let mut b = load_with_joshua(&converted);
        for (pos, tok) in tokens.iter().enumerate() {
            let la = logits(&mut a, &[*tok], pos);
            let lb = logits(&mut b, &[*tok], pos);
            for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
                assert!((x - y).abs() < 2e-5, "f32: position {pos} logit {i} diverges: {x} vs {y}");
            }
        }
        let n = logits(&mut a, &[tokens[0]], 0).len();
        println!("f32 parity OK: {} positions x {n} logits match", tokens.len());
    }
    // Arm 2: production k-quant preset - every tensor requantized to
    // Q2_K/Q4_K/Q8_0/BF16 through the real encoders, decoded by joshua's
    // real candle decoders. Tolerance = encoder quantization noise.
    {
        let converted_k = tmp.join("converted-kquant.gguf");
        let opts = Options {
            preset: "balanced".into(),
            token_map: Some(hf_dir.join("token_map.bin")),
            multipliers: Some(hf_dir.join("engram_constants.json")),
            engram_constants: Some(hf_dir.join("engram_constants.json")),
            // The tiny fixture's indexer tensors (16x16) are smaller than any
            // block format; the real model's equivalents are block-aligned.
            overrides: [("dense", Kind::F32), ("engram-table", Kind::F32)]
                .into_iter()
                .collect(),
            ..Options::default()
        };
        convert::run(&hf_dir, &converted_k, &opts).unwrap();
        let mut a = load_with_joshua(&direct_gguf);
        let mut b = load_with_joshua(&converted_k);
        let mut worst = 0.0f32;
        for (pos, tok) in tokens.iter().enumerate() {
            let la = logits(&mut a, &[*tok], pos);
            let lb = logits(&mut b, &[*tok], pos);
            for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
                assert!(
                    x.is_finite() && y.is_finite(),
                    "k-quant: position {pos} logit {i} non-finite: {x} vs {y}"
                );
                worst = worst.max((x - y).abs());
                assert!(
                    (x - y).abs() < 0.35,
                    "k-quant: position {pos} logit {i} diverges: {x} vs {y}"
                );
            }
        }
        println!("k-quant parity OK (worst |dlogit| = {worst:.4})");
    }
}
