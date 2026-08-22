//! Generate the deterministic tiny GGUF fixture used by the Task 18 spike.

use std::{env, fs, path::PathBuf, process};

use modelq_io::gguf::{inspect, write_q8_0};

fn main() {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| "gguf_q8_0_fixture".into());
    let Some(path) = arguments.next().map(PathBuf::from) else {
        eprintln!("usage: {} <output.gguf>", PathBuf::from(program).display());
        process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {} <output.gguf>", path.display());
        process::exit(2);
    }

    let values = (0..32)
        .map(|index| (index as f32 - 15.5) / 16.0)
        .collect::<Vec<_>>();
    if let Err(error) = write_q8_0(&path, "modelq.fixture.weight", &[32], &values) {
        eprintln!("could not write {}: {error}", path.display());
        process::exit(1);
    }
    let bytes = fs::read(&path).expect("the newly written fixture can be read");
    let summary = inspect(&bytes).expect("the newly written fixture is inspectable");
    println!(
        "wrote {} ({} bytes, {} tensor, data offset {})",
        path.display(),
        bytes.len(),
        summary.tensor_count,
        summary.data_offset
    );
}
