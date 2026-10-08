use anyhow::{anyhow, Result};
use std::path::PathBuf;

use ds4_convert::convert::{self, Options};
use ds4_convert::quant::Kind;

fn usage() -> ! {
    eprintln!(
        "ds4-convert -- DeepSeek-V4 / V4.1-Flash safetensors to deepseek4 / deepseek41 GGUF

USAGE:
  ds4-convert --input <dir-or-file> [--out model.gguf] [flags]

FLAGS:
  --preset <parity|balanced|size>   quantization preset (default: balanced)
                                      parity   lossless F32 (for logit parity tests)
                                      balanced dense/experts Q2_K, down Q4_K,
                                               head Q8_0, router BF16
                                      size     like balanced but down Q2_K, head Q4_K
  --out <path>                      output GGUF (default: model.gguf)
  --token-map <file>                raw i32 token map (default: embedded V4.1-Flash asset)
  --multipliers <file>              JSON with engram multipliers (default: embedded asset)
  --engram-constants <file>         JSON {{multipliers, primes, offsets}} (skips prime search)
  --embd-quant --head-quant --dense-quant --experts-quant --down-quant
  --shexp-quant --router-quant --hc-quant --engram-quant --static-quant <kind>
                                    per-class override (f32 f16 bf16 q4_0 q8_0 q2_k q4_k)
  --keep-mtp                        include the mtp.N.* draft tensors as blk.(N+L).*
  --fp4-high-first                  FP4 packing uses high-nibble-first order
  --allow-unmapped                  skip unknown tensors with a warning
  --dry-run                         print the conversion plan without writing"
    );
    std::process::exit(2)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut out = PathBuf::from("model.gguf");
    let mut opts = Options::default();

    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" => input = Some(PathBuf::from(args.next().ok_or_else(|| anyhow!("--input needs a value"))?)),
            "--out" => out = PathBuf::from(args.next().ok_or_else(|| anyhow!("--out needs a value"))?),
            "--preset" => opts.preset = args.next().ok_or_else(|| anyhow!("--preset needs a value"))?,
            "--token-map" => opts.token_map = Some(PathBuf::from(args.next().ok_or_else(|| anyhow!("--token-map needs a value"))?)),
            "--multipliers" => opts.multipliers = Some(PathBuf::from(args.next().ok_or_else(|| anyhow!("--multipliers needs a value"))?)),
            "--engram-constants" => opts.engram_constants = Some(PathBuf::from(args.next().ok_or_else(|| anyhow!("--engram-constants needs a value"))?)),
            "--keep-mtp" => opts.keep_mtp = true,
            "--fp4-high-first" => opts.fp4_high_first = true,
            "--allow-unmapped" => opts.allow_unmapped = true,
            "--dry-run" => opts.dry_run = true,
            "--embd-quant" | "--head-quant" | "--dense-quant" | "--experts-quant"
            | "--down-quant" | "--shexp-quant" | "--router-quant" | "--hc-quant"
            | "--engram-quant" | "--static-quant" => {
                let slot = match a.as_str() {
                    "--embd-quant" => "embd",
                    "--head-quant" => "head",
                    "--dense-quant" => "dense",
                    "--experts-quant" => "experts",
                    "--down-quant" => "down",
                    "--shexp-quant" => "shexp-down",
                    "--router-quant" => "router",
                    "--hc-quant" => "hc",
                    "--engram-quant" => "engram-table",
                    _ => "static",
                };
                let kind = Kind::parse(&args.next().ok_or_else(|| anyhow!("{a} needs a value"))?)?;
                opts.overrides.insert(slot, kind);
            }
            other => eprintln!("unknown flag {other} (ignored)"),
        }
    }

    let input = input.ok_or_else(|| anyhow!("--input is required"))?;
    let summary = convert::run(&input, &out, &opts)?;
    println!(
        "wrote {}: {} tensors, {:.2} GB (skipped: vision {}, mtp {}, scales {})",
        out.display(),
        summary.tensors_out,
        summary.bytes_out as f64 / 1e9,
        summary.skipped_vision,
        summary.skipped_mtp,
        summary.skipped_scales
    );
    Ok(())
}
