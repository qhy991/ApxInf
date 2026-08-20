use std::fs;
use std::path::{Path, PathBuf};

use apxinf_core::Tensor;
use apxinf_model::qwen35::{Qwen35Config, Qwen35VisionEncoder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "usage: qwen35_vision_check <model_dir> <pixel_values_f32.npy> <T> <H> <W> <output.npy>"
        );
        std::process::exit(2);
    }
    let model_dir = PathBuf::from(&args[1]);
    let (shape, pixels) = read_npy_f32_to_bf16(Path::new(&args[2])).unwrap();
    let grid = [
        args[3].parse::<u32>().unwrap(),
        args[4].parse::<u32>().unwrap(),
        args[5].parse::<u32>().unwrap(),
    ];
    let pixels = Tensor::from_bf16(shape, &pixels).unwrap();
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json")).unwrap();
    let encoder = Qwen35VisionEncoder::load(&model_dir, &config).unwrap();
    let primary = match std::env::var("APXINF_VISION_DUMP_PREFIX") {
        Ok(prefix) => encoder.encode_cpu_debug(&pixels, grid, &prefix).unwrap(),
        Err(_) => encoder.encode_cpu(&pixels, grid).unwrap(),
    };
    write_npy_f32(Path::new(&args[6]), &primary).unwrap();
    let values = primary.to_f32_vec().unwrap();
    println!(
        "vision_primary shape={:?} sum={:.6} absmax={:.6}",
        primary.shape().dims(),
        values.iter().sum::<f32>(),
        values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max),
    );
}

fn read_npy_f32_to_bf16(path: &Path) -> Result<(Vec<usize>, Vec<half::bf16>), String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" || bytes[6] != 1 {
        return Err("expected NumPy v1 array".into());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&bytes[10..data_start]).map_err(|error| error.to_string())?;
    if !header.contains("<f4") {
        return Err("expected little-endian f32 NumPy array".into());
    }
    let shape = parse_shape(header)?;
    let values = bytes[data_start..]
        .chunks_exact(4)
        .map(|chunk| {
            half::bf16::from_f32(f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        })
        .collect::<Vec<_>>();
    if values.len() != shape.iter().product::<usize>() {
        return Err("NumPy payload length does not match shape".into());
    }
    Ok((shape, values))
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = header.find("shape").ok_or("missing shape")?;
    let open = key + header[key..].find('(').ok_or("missing shape open")? + 1;
    let close = open + header[open..].find(')').ok_or("missing shape close")?;
    header[open..close]
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn write_npy_f32(path: &Path, tensor: &Tensor) -> Result<(), String> {
    let values = tensor.to_f32_vec().map_err(|error| error.to_string())?;
    let shape = tensor
        .shape()
        .dims()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape},), }}");
    let padding = (64 - ((header.len() + 11) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut output = Vec::with_capacity(10 + header.len() + values.len() * 4);
    output.extend_from_slice(b"\x93NUMPY");
    output.extend_from_slice(&[1, 0]);
    output.extend_from_slice(&(header.len() as u16).to_le_bytes());
    output.extend_from_slice(header.as_bytes());
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, output).map_err(|error| error.to_string())
}
