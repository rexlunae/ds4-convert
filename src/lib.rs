//! ds4-convert: DeepSeek-V4 / V4.1-Flash safetensors to deepseek4 / deepseek41 GGUF.
//!
//! A standalone, streaming converter. It never materialises more than about
//! a gigabyte of intermediate state per tensor (the engram tables alone are
//! 384 M rows), maps HuggingFace tensor names and config.json fields onto
//! the llama.cpp "deepseek4" / "deepseek41" GGUF conventions that the joshua
//! engine loads, and requantizes FP8/FP4-packed weights to k-quants with the
//! same block encoders the joshua/candle decoders run.

pub mod convert;
pub mod dtypes;
pub mod ggufw;
pub mod hfmap;
pub mod quant;
pub mod st;
