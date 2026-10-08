//! Decode a single tensor from a GGUF with candle's real decoders and print
//! value stats - for comparing my encoders against joshua's actual readers.
use candle_core::Device;

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe <gguf> <tensor>");
    let tensor = std::env::args().nth(2).expect("tensor name");
    let bytes = std::fs::read(&path).unwrap();
    let content = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes[..]))
        .unwrap()
        .to_candle_content()
        .unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let qt = content.tensor(&mut cursor, &tensor, &Device::Cpu).unwrap();
    let vals = qt.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let finite = vals.iter().filter(|v| v.is_finite()).count();
    let mx = vals.iter().cloned().fold(0.0f32, f32::max);
    let mn = vals.iter().cloned().fold(0.0f32, f32::min);
    println!(
        "{}: dtype={:?} dims={:?} n={} finite={} min={} max={}",
        tensor,
        qt.dtype(),
        qt.shape().dims(),
        vals.len(),
        finite,
        mn,
        mx
    );
    println!("first 8: {:?}", &vals[..8.min(vals.len())]);
}