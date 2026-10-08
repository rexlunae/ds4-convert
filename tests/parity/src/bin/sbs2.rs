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
    println!("candle[0..8]: {:?}", &candle_vals[..8]);
    println!("mine  [0..8]: {:?}", &mine[..8]);
    println!("expected  [0..4]: {:?}", &vals[..4]);
    println!("differing: {}/{}", n_diff, candle_vals.len());
}