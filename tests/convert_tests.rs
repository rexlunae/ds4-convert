use ds4_convert::hfmap::{compute_primes_offsets, default_multipliers, map_name, Action, Config};
use ds4_convert::quant::{decode, encode, Kind};

fn v41_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v41",
        "text_config": {
            "vocab_size": 129280, "hidden_size": 5120, "num_hidden_layers": 40,
            "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 1280, "o_groups": 8, "o_lora_rank": 1024,
            "num_key_value_heads": 1, "sliding_window": 128,
            "n_routed_experts": 384, "num_experts_per_tok": 6, "n_shared_experts": 1,
            "moe_intermediate_size": 2304, "routed_scaling_factor": 1.5,
            "norm_topk_prob": true, "rms_norm_eps": 1e-20, "rope_theta": 10000,
            "max_position_embeddings": 1048576,
            "quantization_config": {"weight_block_size": [32, 32]},
            "engram_layer_ids": [1, 14], "engram_vocab_size": 16000000,
            "engram_n_heads": 8, "engram_max_ngram_size": 4,
            "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092
        }
    })
}

fn cfg() -> Config {
    Config::load(v41_config()).unwrap()
}

#[test]
fn hf_names_map_onto_joshua_gguf_names() {
    let cfg = cfg();
    let m = |n: &str| match map_name(n, &cfg, false) {
        Some(Action::Emit(g)) => Some(g),
        _ => None,
    };
    assert_eq!(m("embed.weight").unwrap(), "token_embd.weight");
    assert_eq!(m("head.weight").unwrap(), "output.weight");
    assert_eq!(m("norm.weight").unwrap(), "output_norm.weight");
    assert_eq!(m("layers.3.attn.wq_a.weight").unwrap(), "blk.3.attn_q_a.weight");
    assert_eq!(m("layers.3.attn.wkv.weight").unwrap(), "blk.3.attn_kv.weight");
    assert_eq!(m("layers.3.attn.q_norm.weight").unwrap(), "blk.3.attn_q_a_norm.weight");
    assert_eq!(m("layers.3.attn.compressor.wkv.weight").unwrap(), "blk.3.attn_compressor_kv.weight");
    assert_eq!(m("layers.7.attn.indexer.wq_b.weight").unwrap(), "blk.7.indexer.attn_q_b.weight");
    assert_eq!(m("layers.7.attn.indexer.weights_proj.weight").unwrap(), "blk.7.indexer.proj.weight");
    assert_eq!(m("layers.1.engram.embed.weight").unwrap(), "blk.1.engram_embd.weight");
    assert_eq!(m("layers.1.engram.q_weight").unwrap(), "blk.1.engram_q.weight");
    assert_eq!(m("layers.5.ffn.gate.weight").unwrap(), "blk.5.ffn_gate_inp.weight");
    assert_eq!(m("layers.5.ffn.gate.bias").unwrap(), "blk.5.exp_probs_b.bias");
    assert_eq!(m("layers.5.hc_attn_fn").unwrap(), "blk.5.hc_attn_fn.weight");
    assert_eq!(m("layers.5.ffn.shared_experts.w2.weight").unwrap(), "blk.5.ffn_down_shexp.weight");
    // w1 = gate, w3 = up, w2 = down (Megatron naming).
    match map_name("layers.5.ffn.experts.7.w1.weight", &cfg, false) {
        Some(Action::Expert(l, e, which)) => {
            assert_eq!((l, e, which), (5, 7, 0));
        }
        _ => panic!("expert tensor not routed"),
    }
    assert!(map_name("vision.patch.weight", &cfg, false).is_some()); // Skip*
    assert!(map_name("layers.5.mystery.weight", &cfg, false).is_none());
}

#[test]
fn engram_primes_offsets_match_the_real_file() {
    // The real deepseek41 GGUFs carry 48 primes just above 16M and cumulative
    // offsets; the deterministic search must reproduce them exactly.
    let cfg = cfg();
    let (primes, offsets) = compute_primes_offsets(&cfg).unwrap();
    let asset: serde_json::Value =
        serde_json::from_slice(ds4_convert::hfmap::MULTIPLIERS_ASSET).unwrap();
    let want_p: Vec<u64> = asset["primes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let want_o: Vec<u64> = asset["offsets"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(primes.len(), 48);
    assert_eq!(primes, want_p, "primes diverge from the published deepseek41 GGUF");
    assert_eq!(offsets, want_o, "offsets diverge from the published deepseek41 GGUF");
    let mult = default_multipliers().unwrap();
    assert_eq!(mult.len(), 8);
}

#[test]
fn token_map_asset_matches_the_published_header() {
    let tm = ds4_convert::hfmap::load_token_map(None).unwrap();
    assert_eq!(tm.len(), 129280);
    let distinct: std::collections::HashSet<i32> = tm.iter().copied().collect();
    assert_eq!(distinct.len(), 99092);
    assert_eq!(tm[2], 2, "pad id comes from token_map[pad_token_id]");
}

#[test]
fn k_quant_encoders_round_trip_within_tolerance() {
    // xorshift-ish deterministic signal, block-aligned length.
    let mut state = 12345u32;
    let vals: Vec<f32> = (0..256 * 7)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 2000) as f32 / 1000.0 - 1.0) * 0.1
        })
        .collect();
    for kind in [Kind::Q8_0, Kind::Q4_0, Kind::Q2K, Kind::Q4K] {
        let (id, bytes) = encode(kind, &[vals.len()], &vals).unwrap();
        assert_eq!(id, kind.type_id());
        assert_eq!(bytes.len() as u64, kind.tensor_bytes(&[vals.len()]).unwrap());
        let mut back = Vec::new();
        decode(kind, vals.len(), &bytes, &mut back).unwrap();
        assert_eq!(back.len(), vals.len());
        let max_err: f32 = vals.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let scale = vals.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let bound = match kind {
            Kind::Q8_0 => scale / 127.0 * 1.01,
            // candle's Q4_0 encoder truncates at the positive extreme
            // (min(15, trunc(x*id + 8.5))), allowing ~1 LSB of error there.
            Kind::Q4_0 => scale / 8.0 * 1.05,
            // 2-bit steps are coarse: one sub-block step can approach
            // (sub-block range)/3 ≈ (2·scale)/3, plus qkx1 refinement slack.
            Kind::Q2K => scale / 3.0 * 1.3,
            Kind::Q4K => scale / 15.0 * 1.05,
            _ => unreachable!(),
        };
        assert!(max_err <= bound, "{kind:?}: max_err {max_err} > {bound}");
    }
}

#[test]
fn f16_bf16_round_trip() {
    for kind in [Kind::F16, Kind::BF16] {
        let vals: Vec<f32> = [0.0, 1.0, -2.5, 0.125, -0.125, 4.0].to_vec();
        let (id, bytes) = encode(kind, &[vals.len()], &vals).unwrap();
        assert_eq!(id, kind.type_id());
        let mut back = Vec::new();
        decode(kind, vals.len(), &bytes, &mut back).unwrap();
        for (a, b) in vals.iter().zip(&back) {
            assert_eq!(a, b, "{kind:?} must be exact");
        }
    }
}
