//! Maintained FP32 recurrence contract, including the actual Backend dispatch.

use apxinf_core::{Backend, CpuBackend, DType, Device, Result, Shape, Tensor};

use crate::transfers::copy_cpu_to_cuda;
use crate::workspace::{prepare_with_workspace, with_workspace, GraphWorkspace};
use crate::{CudaBackend, CudaBuffer};

fn values(count: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 17 + salt) % 53) as f32 - 26.0) * scale / 26.0)
        .collect()
}

fn fixture(t: usize, hk: usize, hv: usize, kd: usize, vd: usize) -> [Tensor; 7] {
    let tensor = |shape: Vec<usize>, salt, scale| {
        Tensor::from_f32_vec(shape.clone(), values(shape.iter().product(), salt, scale)).unwrap()
    };
    [
        tensor(vec![t, hk, kd], 3, 1.0 / (kd as f32).sqrt()),
        tensor(vec![t, hk, kd], 7, 1.0 / (kd as f32).sqrt()),
        tensor(vec![t, hv, vd], 13, 0.8),
        tensor(vec![t, hv], 19, 2.0),
        tensor(vec![t, hv], 29, 4.0),
        tensor(vec![hv], 37, 0.5),
        tensor(vec![hv], 41, 0.7),
    ]
}

fn initial_state(hv: usize, kd: usize, vd: usize) -> Tensor {
    Tensor::from_f32_vec(vec![hv, kd, vd], values(hv * kd * vd, 11, 0.4)).unwrap()
}

fn upload(backend: &CudaBackend, tensors: &[Tensor; 7]) -> [Tensor; 7] {
    std::array::from_fn(|i| backend.to_device(&tensors[i]).unwrap())
}

fn run(
    backend: &dyn Backend,
    inputs: &[Tensor; 7],
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    backend.gated_delta_recurrent(
        &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], &inputs[6], state,
    )
}

fn close(backend: &CudaBackend, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), DType::F32);
    assert_eq!(actual.device(), backend.device());
    let actual = backend.to_cpu(actual).unwrap();
    for (index, (&actual, &expected)) in actual
        .as_f32()
        .unwrap()
        .iter()
        .zip(expected.as_f32().unwrap())
        .enumerate()
    {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite value at {index}"
        );
        assert!(
            (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "index {index}: actual {actual}, expected {expected}"
        );
    }
}

fn close_pair(backend: &CudaBackend, actual: &(Tensor, Tensor), expected: &(Tensor, Tensor)) {
    close(backend, &actual.0, &expected.0);
    close(backend, &actual.1, &expected.1);
}

#[test]
fn gated_delta_eager_matches_cpu_and_keeps_initial_state_immutable() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    for (t, hk, hv, kd, vd) in [
        (0, 1, 2, 3, 5),
        (0, 2, 4, 128, 129),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 128, 1),
        (7, 2, 6, 7, 11),
        (17, 2, 4, 33, 65),
        (32, 4, 8, 128, 128),
        (3, 2, 6, 128, 65),
        (3, 1, 2, 257, 129),
    ] {
        let host = fixture(t, hk, hv, kd, vd);
        let gpu = upload(&backend, &host);
        let initial = initial_state(hv, kd, vd);
        let gpu_initial = backend.to_device(&initial).unwrap();
        for use_state in [false, true] {
            let expected = run(&CpuBackend, &host, use_state.then_some(&initial)).unwrap();
            let actual = run(&backend, &gpu, use_state.then_some(&gpu_initial)).unwrap();
            close_pair(&backend, &actual, &expected);
            close(&backend, &gpu_initial, &initial);
            let repeated = run(&backend, &gpu, use_state.then_some(&gpu_initial)).unwrap();
            close_pair(&backend, &repeated, &expected);
        }
    }
}

#[test]
fn gated_delta_chunked_and_token_decode_match_prefill() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let (t, hk, hv, kd, vd) = (13, 2, 6, 17, 9);
    let host = fixture(t, hk, hv, kd, vd);
    let initial = initial_state(hv, kd, vd);
    for chunk_size in [1, 4] {
        let expected = run(&CpuBackend, &host, Some(&initial)).unwrap();
        let mut state = backend.to_device(&initial).unwrap();
        for start in (0..t).step_by(chunk_size) {
            let end = (start + chunk_size).min(t);
            let sliced: [Tensor; 7] = std::array::from_fn(|i| {
                if i >= 5 {
                    return host[i].clone();
                }
                let mut dims = host[i].shape().dims().to_vec();
                let stride = dims[1..].iter().product::<usize>();
                dims[0] = end - start;
                Tensor::from_f32(
                    dims,
                    &host[i].as_f32().unwrap()[start * stride..end * stride],
                )
                .unwrap()
            });
            let (output, next) = run(&backend, &upload(&backend, &sliced), Some(&state)).unwrap();
            let expected_slice = Tensor::from_f32(
                vec![end - start, hv, vd],
                &expected.0.as_f32().unwrap()[start * hv * vd..end * hv * vd],
            )
            .unwrap();
            close(&backend, &output, &expected_slice);
            state = next;
        }
        close(&backend, &state, &expected.1);
    }
}

#[test]
fn gated_delta_extreme_gates_remain_finite() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let mut host = fixture(4, 1, 2, 3, 5);
    let gates = [-1000.0, 1000.0, -21.0, 21.0, -20.0, 20.0, -0.0, 0.0];
    host[3] = Tensor::from_f32(vec![4, 2], &gates).unwrap();
    host[4] = host[3].clone();
    let initial = initial_state(2, 3, 5);
    let expected = run(&CpuBackend, &host, Some(&initial)).unwrap();
    let actual = run(
        &backend,
        &upload(&backend, &host),
        Some(&backend.to_device(&initial).unwrap()),
    )
    .unwrap();
    close_pair(&backend, &actual, &expected);
}

#[test]
fn gated_delta_rejects_shape_dtype_device_and_storage_errors() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let host = fixture(2, 1, 2, 3, 5);
    let gpu = upload(&backend, &host);
    // Every argument must have the requested dtype and context device.
    for index in 0..7 {
        let mut invalid = gpu.clone();
        invalid[index] = host[index].clone();
        assert!(run(&backend, &invalid, None)
            .unwrap_err()
            .to_string()
            .contains("device"));
        invalid[index] = Tensor::from_raw_parts(
            gpu[index].shape().clone(),
            DType::F16,
            gpu[index].device(),
            gpu[index].storage().clone(),
        );
        assert!(run(&backend, &invalid, None)
            .unwrap_err()
            .to_string()
            .contains("f16"));
        invalid[index] = Tensor::from_raw_parts(
            gpu[index].shape().clone(),
            DType::F32,
            Device::Cuda(1),
            gpu[index].storage().clone(),
        );
        assert!(run(&backend, &invalid, None).is_err());
    }
    for (index, shape) in [
        (0, vec![2, 3]),
        (1, vec![2, 1, 4]),
        (2, vec![3, 2, 5]),
        (3, vec![2, 1]),
        (4, vec![4]),
        (5, vec![1]),
        (6, vec![1, 2]),
        (0, vec![2, 0, 3]),
        (0, vec![2, 3, 3]),
    ] {
        let mut invalid = gpu.clone();
        invalid[index] = Tensor::from_raw_parts(
            Shape::new(shape),
            DType::F32,
            gpu[index].device(),
            gpu[index].storage().clone(),
        );
        assert!(run(&backend, &invalid, None).is_err());
    }
    let bad_state = backend
        .to_device(&Tensor::zeros(vec![2, 5, 3], DType::F32))
        .unwrap();
    assert!(run(&backend, &gpu, Some(&bad_state)).is_err());
    let host_state = initial_state(2, 3, 5);
    assert!(run(&backend, &gpu, Some(&host_state)).is_err());
    let gpu_state = backend.to_device(&host_state).unwrap();
    let bad_dtype = Tensor::from_raw_parts(
        gpu_state.shape().clone(),
        DType::BF16,
        gpu_state.device(),
        gpu_state.storage().clone(),
    );
    assert!(run(&backend, &gpu, Some(&bad_dtype)).is_err());

    let tiny = CudaBuffer::alloc(4, 0).unwrap();
    let mut invalid = gpu.clone();
    invalid[0] = tiny.clone().into_tensor(gpu[0].shape().clone(), DType::F32);
    assert!(run(&backend, &invalid, None)
        .unwrap_err()
        .to_string()
        .contains("requires"));
    let misaligned = CudaBuffer::alloc(25, 0).unwrap().view(1, 24).unwrap();
    invalid[0] = misaligned.into_tensor(gpu[0].shape().clone(), DType::F32);
    assert!(run(&backend, &invalid, None)
        .unwrap_err()
        .to_string()
        .contains("aligned"));

    // Malformed metadata must fail checked size arithmetic without an allocation.
    invalid[0] = tiny
        .clone()
        .into_tensor(Shape::new(vec![u32::MAX as usize + 1, 1, 3]), DType::F32);
    assert!(run(&backend, &invalid, None)
        .unwrap_err()
        .to_string()
        .contains("u32"));
    invalid[0] = tiny.into_tensor(
        Shape::new(vec![u32::MAX as usize, 1, u32::MAX as usize]),
        DType::F32,
    );
    assert!(run(&backend, &invalid, None)
        .unwrap_err()
        .to_string()
        .contains("overflow"));
}

#[test]
fn gated_delta_prepare_capture_replay_and_input_updates_match_eager() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    for (t, kd) in [(0, 7), (1, 7), (7, 7), (0, 128), (1, 128), (7, 128)] {
        for use_state in [false, true] {
            let host = fixture(t, 2, 4, kd, 11);
            let gpu = upload(&backend, &host);
            let initial = initial_state(4, kd, 11);
            let gpu_initial = backend.to_device(&initial).unwrap();
            let execute = || run(&backend, &gpu, use_state.then_some(&gpu_initial));
            let output_bytes = t * 4 * 11 * 4;
            let capacity = (output_bytes + 255) / 256 * 256 + 4 * kd * 11 * 4;
            let workspace = GraphWorkspace::new(capacity, 0).unwrap();
            let expected = run(&CpuBackend, &host, use_state.then_some(&initial)).unwrap();
            let prepared = prepare_with_workspace(&workspace, execute).unwrap();
            assert_eq!(workspace.used(), capacity);
            close_pair(&backend, &prepared, &expected);
            drop(prepared);
            backend.begin_capture().unwrap();
            let captured = with_workspace(&workspace, execute).unwrap();
            let graph = backend.end_capture().unwrap();
            for _ in 0..3 {
                graph.replay().unwrap();
                close_pair(&backend, &captured, &expected);
            }
            let mut changed = host.clone();
            let replacement = values(t * 4 * 11, 47, 1.7);
            changed[2] = Tensor::from_f32_vec(vec![t, 4, 11], replacement).unwrap();
            copy_cpu_to_cuda(&changed[2], &gpu[2]).unwrap();
            let changed_state =
                Tensor::from_f32_vec(vec![4, kd, 11], values(4 * kd * 11, 5, 0.9)).unwrap();
            copy_cpu_to_cuda(&changed_state, &gpu_initial).unwrap();
            let expected_changed =
                run(&CpuBackend, &changed, use_state.then_some(&changed_state)).unwrap();
            graph.replay().unwrap();
            close_pair(&backend, &captured, &expected_changed);
            if t > 0 {
                assert!(expected
                    .0
                    .as_f32()
                    .unwrap()
                    .iter()
                    .zip(expected_changed.0.as_f32().unwrap())
                    .any(|(a, b)| (a - b).abs() > 1e-4));
            }
            close(&backend, &gpu_initial, &changed_state);
            close_pair(&backend, &execute().unwrap(), &expected_changed);
        }
    }
}

#[test]
fn gated_delta_workspace_exhaustion_and_alias_errors_leave_stream_usable() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let host = fixture(1, 1, 2, 3, 5);
    let gpu = upload(&backend, &host);
    let workspace = GraphWorkspace::new(256, 0).unwrap();
    let error = prepare_with_workspace(&workspace, || run(&backend, &gpu, None)).unwrap_err();
    assert!(error.to_string().contains("workspace exhausted"));
    backend.begin_capture().unwrap();
    let failed = with_workspace(&workspace, || run(&backend, &gpu, None));
    assert!(failed
        .unwrap_err()
        .to_string()
        .contains("workspace exhausted"));
    // End capture even when the safe operator fails before issuing a launch.
    let graph = backend.end_capture().unwrap();
    drop(graph);

    let workspace = GraphWorkspace::new(1024, 0).unwrap();
    let first = prepare_with_workspace(&workspace, || run(&backend, &gpu, None)).unwrap();
    let expected = run(&CpuBackend, &host, None).unwrap();
    close_pair(&backend, &first, &expected);
    let error = with_workspace(&workspace, || run(&backend, &gpu, Some(&first.1))).unwrap_err();
    assert!(error.to_string().contains("overlaps input"));
    close_pair(&backend, &first, &expected);
    close_pair(&backend, &run(&backend, &gpu, None).unwrap(), &expected);
    backend.synchronize().unwrap();
}
