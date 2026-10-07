use std::{
    fs,
    path::{Path, PathBuf},
    process::id,
    sync::atomic::{AtomicU64, Ordering},
};

use modelq::{
    backend::cpu::ParallelConfig,
    io::{
        nvfp4::{Nvfp4Execution, Nvfp4WriterError},
        safetensors::{MappedSafetensors, TensorSource},
        te_container::{
            TE_CONTAINER_FORMAT, TeContainerError, TeLayoutError, TeManifestTensor, TeOutputRole,
            plan_te_output, read_te_container_manifest, te_global_scale, te_matrix_values,
            te_scale_shape, write_te_nvfp4_safetensors, write_te_nvfp4_safetensors_with,
        },
        transformer_engine::export_transformer_engine_nvfp4,
    },
    quant::nvfp4,
};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "modelq-te-container-{label}-{}-{}",
            id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("test directory is created");
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `(name, dtype, shape, payload)`
type Fixture = (String, &'static str, Vec<usize>, Vec<u8>);

fn values(count: usize, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32) * 0.37 + seed).sin() * (1.0 + (index % 11) as f32))
        .collect()
}

fn f32_matrix(name: &str, rows: usize, columns: usize, seed: f32) -> (Fixture, Vec<f32>) {
    let data = values(rows * columns, seed);
    let payload = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    ((name.to_owned(), "F32", vec![rows, columns], payload), data)
}

fn write_source(path: &Path, tensors: &[&Fixture]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, payload) in tensors {
        let start = data.len();
        data.extend_from_slice(payload);
        header.insert(
            name.clone(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes).unwrap();
}

struct Source {
    path: PathBuf,
    a: Vec<f32>,
    b: Vec<f32>,
}

/// Matrices exercising padding: `[64,64]` (the Task 28 shape) and `[48,96]`
/// (`M` not a multiple of 128, `K/16` not a multiple of 4), plus a preserved
/// vector and integer tensor.
fn source(dir: &TestDir) -> Source {
    let (a_fixture, a) = f32_matrix("a.weight", 64, 64, 0.0);
    let (b_fixture, b) = f32_matrix("b.weight", 48, 96, 1.5);
    let norm: Fixture = (
        "norm.weight".to_owned(),
        "F32",
        vec![4],
        [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    );
    let ids: Fixture = ("ids".to_owned(), "U8", vec![3], vec![1, 2, 3]);
    let path = dir.join("source.safetensors");
    write_source(&path, &[&a_fixture, &b_fixture, &norm, &ids]);
    Source { path, a, b }
}

fn names(items: &[&str]) -> Vec<String> {
    items.iter().map(|name| (*name).to_owned()).collect()
}

fn write_container(source: &MappedSafetensors, quantized: &[&str], path: &Path) {
    let plan = plan_te_output(&source.tensor_summaries(), &names(quantized)).expect("plans");
    write_te_nvfp4_safetensors(source, &plan, path).expect("writes");
}

fn field(file: &MappedSafetensors, name: &str) -> Vec<u8> {
    file.tensor_bytes(name).expect("field exists").to_vec()
}

#[test]
fn plans_fields_shapes_and_offsets() {
    let dir = TestDir::new("plan");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let plan = plan_te_output(&file.tensor_summaries(), &names(&["b.weight", "a.weight"])).unwrap();

    let order: Vec<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        order,
        [
            "a.weight.rowwise_data",
            "a.weight.rowwise_scale_inv",
            "a.weight.amax_rowwise",
            "b.weight.rowwise_data",
            "b.weight.rowwise_scale_inv",
            "b.weight.amax_rowwise",
            "ids",
            "norm.weight",
        ]
    );
    assert_eq!(plan.quantized_source_names(), ["a.weight", "b.weight"]);
    let a_scale = plan.tensor("a.weight.rowwise_scale_inv").unwrap();
    assert_eq!(a_scale.shape, [128, 4]);
    assert_eq!(a_scale.byte_len, 512);
    let b_data = plan.tensor("b.weight.rowwise_data").unwrap();
    assert_eq!(b_data.shape, [48, 48]);
    assert_eq!(b_data.role, TeOutputRole::RowwiseData);
    let b_scale = plan.tensor("b.weight.rowwise_scale_inv").unwrap();
    assert_eq!(b_scale.shape, [128, 8]);
    assert_eq!(te_scale_shape(48, 96), Some([128, 8]));
    assert_eq!(te_scale_shape(144, 80), Some([256, 8]));
    // Contiguous and covering the whole data section.
    let mut cursor = 0;
    for tensor in &plan.tensors {
        assert_eq!(tensor.data_offsets.start, cursor);
        cursor = tensor.data_offsets.end;
    }
    assert_eq!(plan.total_data_bytes, cursor);
}

#[test]
fn rejects_ineligible_selections_and_name_collisions() {
    let dir = TestDir::new("layout-errors");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let summaries = file.tensor_summaries();
    let plan = |selected: &[&str]| plan_te_output(&summaries, &names(selected));

    assert!(matches!(
        plan(&["norm.weight"]),
        Err(TeLayoutError::InvalidQuantizedShape { .. })
    ));
    assert!(matches!(
        plan(&["ids"]),
        Err(TeLayoutError::UnsupportedQuantizedDtype { .. })
    ));
    assert!(matches!(
        plan(&["missing"]),
        Err(TeLayoutError::SelectedNameNotFound { .. })
    ));
    assert!(matches!(
        plan(&["a.weight", "a.weight"]),
        Err(TeLayoutError::DuplicateSelectedName { .. })
    ));

    let mut odd = summaries.clone();
    for (name, shape) in [
        ("rows", vec![70, 64]),
        ("cols", vec![64, 40]),
        ("stack", vec![2, 64, 64]),
    ] {
        let mut candidate = summaries[0].clone();
        candidate.name = name.to_owned();
        candidate.shape = shape;
        odd.push(candidate);
        assert!(
            matches!(
                plan_te_output(&odd, &names(&[name])),
                Err(TeLayoutError::InvalidQuantizedShape { .. })
            ),
            "{name}"
        );
    }

    // A source already named like a generated field collides.
    let mut colliding = summaries.clone();
    let mut clash = summaries[0].clone();
    clash.name = "a.weight.rowwise_data".to_owned();
    colliding.push(clash);
    assert!(matches!(
        plan_te_output(&colliding, &names(&["a.weight"])),
        Err(TeLayoutError::OutputNameCollision { .. })
    ));
}

#[test]
fn writes_a_valid_container_whose_fields_match_the_native_quantizer() {
    let dir = TestDir::new("write");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let output = dir.join("out.safetensors");
    write_container(&file, &["a.weight", "b.weight"], &output);

    let container = MappedSafetensors::open(&output).unwrap();
    assert_eq!(
        container
            .metadata()
            .get("modelq.format")
            .map(String::as_str),
        Some(TE_CONTAINER_FORMAT)
    );
    let manifest = read_te_container_manifest(&container).expect("manifest is valid");
    assert_eq!(manifest.tensors.len(), 4);
    assert!(matches!(
        manifest.tensors["norm.weight"],
        TeManifestTensor::Preserved { .. }
    ));

    for (name, data, rows, columns) in [("a.weight", &src.a, 64, 64), ("b.weight", &src.b, 48, 96)]
    {
        let native = nvfp4::quantize_shaped(data, &[rows, columns]).unwrap();
        let packed = field(&container, &format!("{name}.rowwise_data"));
        assert_eq!(packed, native.packed(), "{name} data");

        let [padded_rows, padded_blocks] = te_scale_shape(rows, columns).unwrap();
        let padded = field(&container, &format!("{name}.rowwise_scale_inv"));
        assert_eq!(padded.len(), padded_rows * padded_blocks);
        let blocks = columns / 16;
        for row in 0..padded_rows {
            let stored = &padded[row * padded_blocks..(row + 1) * padded_blocks];
            if row < rows {
                assert_eq!(
                    &stored[..blocks],
                    &native.block_scales()[row * blocks..(row + 1) * blocks]
                );
                assert!(
                    stored[blocks..].iter().all(|&byte| byte == 0),
                    "{name} row {row}"
                );
            } else {
                assert!(
                    stored.iter().all(|&byte| byte == 0),
                    "{name} padding row {row}"
                );
            }
        }

        let amax = f32::from_le_bytes(
            field(&container, &format!("{name}.amax_rowwise"))[..]
                .try_into()
                .unwrap(),
        );
        assert_eq!(amax.to_bits(), (native.global_scale() * 2688.0).to_bits());

        // Decoding through the stored amax matches the native decode closely.
        let decoded: Vec<f32> = te_matrix_values(rows, columns, &packed, &padded, amax)
            .expect("valid matrix")
            .collect();
        let reference = native.dequantize().unwrap();
        assert_eq!(decoded.len(), reference.len());
        for (index, (a, b)) in decoded.iter().zip(&reference).enumerate() {
            assert!(
                (a - b).abs() <= b.abs() * 1e-6,
                "{name}[{index}]: {a} vs {b}"
            );
        }
    }
    assert_eq!(
        field(&container, "norm.weight"),
        field(&file, "norm.weight")
    );
    assert_eq!(field(&container, "ids"), [1, 2, 3]);
}

#[test]
fn fields_equal_the_hardware_validated_single_matrix_profile() {
    // Task 28's `export_transformer_engine_nvfp4` produced the artifact that
    // passed on a B200; the streaming writer must place exactly its bytes.
    let dir = TestDir::new("profile");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let output = dir.join("out.safetensors");
    write_container(&file, &["a.weight"], &output);
    let container = MappedSafetensors::open(&output).unwrap();

    let native = nvfp4::quantize_shaped(&src.a, &[64, 64]).unwrap();
    let profile = export_transformer_engine_nvfp4("a.weight", &[64, 64], &native).unwrap();
    assert_eq!(
        field(&container, "a.weight.rowwise_data"),
        profile.rowwise_data
    );
    assert_eq!(
        field(&container, "a.weight.rowwise_scale_inv"),
        profile.rowwise_scale_inv
    );
    assert_eq!(
        field(&container, "a.weight.amax_rowwise"),
        profile.amax_rowwise.to_le_bytes()
    );
    assert_eq!(profile.rowwise_scale_inv_shape, [128, 4]);
}

#[test]
fn an_all_zero_matrix_stores_a_zero_amax() {
    let dir = TestDir::new("zero");
    let zero: Fixture = (
        "z.weight".to_owned(),
        "F32",
        vec![16, 16],
        vec![0; 16 * 16 * 4],
    );
    let path = dir.join("source.safetensors");
    write_source(&path, &[&zero]);
    let file = MappedSafetensors::open(&path).unwrap();
    let output = dir.join("out.safetensors");
    write_container(&file, &["z.weight"], &output);

    let container = MappedSafetensors::open(&output).unwrap();
    read_te_container_manifest(&container).expect("valid");
    let amax = f32::from_le_bytes(
        field(&container, "z.weight.amax_rowwise")[..]
            .try_into()
            .unwrap(),
    );
    assert_eq!(amax, 0.0);
    assert_eq!(te_global_scale(amax), 1.0);
    let decoded: Vec<f32> = te_matrix_values(
        16,
        16,
        &field(&container, "z.weight.rowwise_data"),
        &field(&container, "z.weight.rowwise_scale_inv"),
        amax,
    )
    .unwrap()
    .collect();
    assert!(decoded.iter().all(|&value| value == 0.0));
}

#[test]
fn bf16_sources_are_recorded_and_quantized() {
    let dir = TestDir::new("bf16");
    let data = values(32 * 32, 0.3);
    let payload: Vec<u8> = data
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let bf16: Fixture = ("m.weight".to_owned(), "BF16", vec![32, 32], payload);
    let path = dir.join("source.safetensors");
    write_source(&path, &[&bf16]);
    let file = MappedSafetensors::open(&path).unwrap();
    let output = dir.join("out.safetensors");
    write_container(&file, &["m.weight"], &output);

    let container = MappedSafetensors::open(&output).unwrap();
    let manifest = read_te_container_manifest(&container).unwrap();
    match &manifest.tensors["m.weight"] {
        TeManifestTensor::Quantized { original_dtype, .. } => assert_eq!(original_dtype, "BF16"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn execution_modes_write_identical_bytes() {
    let dir = TestDir::new("execution");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let plan = plan_te_output(&file.tensor_summaries(), &names(&["a.weight", "b.weight"])).unwrap();

    let reference = dir.join("reference.safetensors");
    write_te_nvfp4_safetensors(&file, &plan, &reference).unwrap();
    let expected = fs::read(&reference).unwrap();
    for (workers, chunk_elements) in [(1, 16), (2, 64), (3, 1024), (8, 4096), (500, 64)] {
        let path = dir.join(&format!("p-{workers}-{chunk_elements}.safetensors"));
        write_te_nvfp4_safetensors_with(
            &file,
            &plan,
            &path,
            Nvfp4Execution::Parallel(ParallelConfig::new(workers, chunk_elements)),
        )
        .unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            expected,
            "{workers}/{chunk_elements}"
        );
    }
}

#[test]
fn failures_leave_no_output_and_never_touch_existing_files() {
    let dir = TestDir::new("failure");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    let plan = plan_te_output(&file.tensor_summaries(), &names(&["a.weight"])).unwrap();

    let existing = dir.join("existing.safetensors");
    fs::write(&existing, b"keep").unwrap();
    assert!(matches!(
        write_te_nvfp4_safetensors(&file, &plan, &existing),
        Err(Nvfp4WriterError::DestinationExists { .. })
    ));
    assert_eq!(fs::read(&existing).unwrap(), b"keep");

    let before = fs::read(&src.path).unwrap();
    assert!(matches!(
        write_te_nvfp4_safetensors(&file, &plan, &src.path),
        Err(Nvfp4WriterError::SourceDestinationConflict { .. })
    ));
    assert_eq!(fs::read(&src.path).unwrap(), before);

    // A plan for different tensors does not match the source.
    let other = TestDir::new("failure-other");
    let (fixture, _) = f32_matrix("a.weight", 32, 32, 9.0);
    let other_path = other.join("other.safetensors");
    write_source(&other_path, &[&fixture]);
    let other_file = MappedSafetensors::open(&other_path).unwrap();
    let output = dir.join("out.safetensors");
    assert!(matches!(
        write_te_nvfp4_safetensors(&other_file, &plan, &output),
        Err(Nvfp4WriterError::PlanMismatch)
    ));
    assert!(!output.exists());

    // A non-finite value fails mid-write and leaves nothing.
    let mut nan = f32_matrix("n.weight", 32, 32, 0.0).0;
    nan.3[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
    let nan_path = dir.join("nan.safetensors");
    write_source(&nan_path, &[&nan]);
    let nan_file = MappedSafetensors::open(&nan_path).unwrap();
    let nan_plan = plan_te_output(&nan_file.tensor_summaries(), &names(&["n.weight"])).unwrap();
    let nan_output = dir.join("nan-out.safetensors");
    assert!(write_te_nvfp4_safetensors(&nan_file, &nan_plan, &nan_output).is_err());
    assert!(!nan_output.exists());
    let leftovers: Vec<_> = fs::read_dir(&dir.0)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn the_validator_rejects_corrupted_matrices_and_foreign_files() {
    let dir = TestDir::new("validate");
    let src = source(&dir);
    let file = MappedSafetensors::open(&src.path).unwrap();
    // A plain SafeTensors file is not a container.
    assert!(matches!(
        read_te_container_manifest(&file),
        Err(TeContainerError::Invalid { .. })
    ));

    let output = dir.join("out.safetensors");
    write_container(&file, &["b.weight"], &output);
    let container = MappedSafetensors::open(&output).unwrap();
    let data = field(&container, "b.weight.rowwise_data");
    let scales = field(&container, "b.weight.rowwise_scale_inv");
    let amax = f32::from_le_bytes(
        field(&container, "b.weight.amax_rowwise")[..]
            .try_into()
            .unwrap(),
    );
    let decode = |data: &[u8], scales: &[u8], amax: f32| {
        te_matrix_values(48, 96, data, scales, amax).map(|values| values.count())
    };
    assert_eq!(decode(&data, &scales, amax).unwrap(), 48 * 96);

    // Non-zero padding in a used row, in a padding row, and in the tail columns.
    for position in [6, 7, 48 * 8 + 3, 127 * 8 + 7] {
        let mut corrupt = scales.clone();
        corrupt[position] = 0x38;
        assert!(
            decode(&data, &corrupt, amax).is_err(),
            "padding byte {position}"
        );
    }
    // Invalid scale bytes: negative sign bit, NaN pattern.
    for bad in [0xb8_u8, 0x7f] {
        let mut corrupt = scales.clone();
        corrupt[0] = bad;
        assert!(decode(&data, &corrupt, amax).is_err(), "scale {bad:#x}");
    }
    // Wrong lengths, bad amax values, zero amax with nonzero scales.
    assert!(decode(&data[1..], &scales, amax).is_err());
    assert!(decode(&data, &scales[1..], amax).is_err());
    for bad in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        assert!(decode(&data, &scales, bad).is_err(), "amax {bad}");
    }
    // A zero scale block must hold only zero values.
    let mut corrupt = scales.clone();
    corrupt[0] = 0;
    assert!(decode(&data, &corrupt, amax).is_err());
}
