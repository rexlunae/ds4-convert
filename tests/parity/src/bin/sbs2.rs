//! Minimal isolation: one Q2_K tensor written by ds4-convert's writer,
//! decoded by candle's real decoder vs my mirror.
use candle_core::Device;

fn main() {
    let dims = [16usize, 256];
    let vals: Vec<f32> = (0..4096)
        .map(|i| ((i % 251) as f32) * 0.003 - 0.35)
        .collect();
    let (dtype_id, bytes) = ds4_convert::quant::encode(ds4_convert::quant::Kind::Q2K, &dims, &vals).unwrap();
    assert_eq!(dtype_id, 10);
    let plan = [ds4_convert::ggufw::TPlan {
        name: "test.weight".to_string(),
        dims: dims.iter().map(|&d| d as u64).collect(),
        dtype_id,
        nbytes: bytes.len() as u64,
    }];
    let md: Vec<(String, ds4_convert::ggufw::V)> = vec![("general.architecture".to_string(), ds4_convert::ggufw::V::Str("test".into()))];
    let out = "/tmp/sbs2.gguf";
    let mut w = ds4_convert::ggufw::Writer::create(std::path::Path::new(out), &md, plan.into_iter().collect()).unwrap();
    w.push(&bytes).unwrap();
    w.finish().unwrap();

    let read = std::fs::read(out).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&read[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mut cursor = std::io::Cursor::new(&read[..]);
    let qt = content.tensor(&mut cursor, "test.weight", &Device::Cpu).unwrap();
    let candle_vals = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut mine = Vec::new();
    ds4_convert::quant::decode(ds4_convert::quant::Kind::Q2K, 256, &bytes, &mut mine).unwrap();
    let n_diff = candle_vals.iter().zip(&mine).filter(|(a, b)| (**a - **b).abs() > 1e-6).count();
    // candle's own encoder on the same vals: byte-diff against mine
    let src_t = candle_core::Tensor::from_vec(vals.clone(), dims, &Device::Cpu).unwrap();
    let candle_qt = candle_core::quantized::QTensor::quantize(&src_t, candle_core::quantized::GgmlDType::Q2K).unwrap();
    let cbytes = candle_qt.data().unwrap();
    let mut first_diffs: Vec<(usize, u8, u8)> = Vec::new();
    for (i, (a, b)) in cbytes.iter().zip(bytes.iter()).enumerate() {
        if a != b {
            if first_diffs.len() < 8 {
                first_diffs.push((i, *a, *b));
            }
        }
    }
    println!("byte diffs: {} of {} (first: {:?})", first_diffs.len() + 0, cbytes.len(), first_diffs.iter().take(4).map(|(i, a, b)| (i, a, b)).collect::<Vec<_>>());
    println!("candle-enc [0..12 bytes]: {:?}", &cbytes[..12.min(cbytes.len())]);
    println!("mine-enc    [0..12 bytes]: {:?}", &bytes[..12.min(bytes.len())]);
    println!("candle[0..8]: {:?}", &candle_vals[..8]);
    println!("mine  [0..8]: {:?}", &mine[..8]);
    println!("expected  [0..4]: {:?}", &vals[..4]);
    println!("differing: {}/{}", n_diff, candle_vals.len());
}