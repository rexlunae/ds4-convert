//! Decode the same Q2K bytes with my mirror and candle's real decoder.
use candle_core::Device;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let tensor = std::env::args().nth(2).expect("tensor name");
    let bytes = std::fs::read(&path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let qt = content
        .tensor(&mut cursor, &tensor, &Device::Cpu)
        .unwrap();
    let candle_vals = qt
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // my mirror: raw block bytes -> ds4_convert::quant::decode
    let dims = qt.shape().dims().to_vec();
    let nb: usize = dims.iter().product::<usize>() / 256;
    let raw = qt.data().unwrap();
    let mut mine = Vec::new();
    ds4_convert::quant::decode(ds4_convert::quant::Kind::Q2K, dims[dims.len() - 1], &raw[..nb * 84], &mut mine)
        .unwrap();
    println!(
        "candle[0..8]: {:?}",
        &candle_vals[..8],
    );
    println!("mine  [0..8]: {:?}", &mine[..8]);
    let n_diff = candle_vals
        .iter()
        .zip(&mine)
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    println!(
        "differing elements: {}/{} (candle min {} max {})",
        n_diff,
        candle_vals.len(),
        candle_vals.iter().cloned().fold(f32::INFINITY, f32::min),
        candle_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}