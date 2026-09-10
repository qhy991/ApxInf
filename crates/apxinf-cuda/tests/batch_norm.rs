use apxinf_core::Tensor;
use apxinf_cuda::{kernels::norm::batch_relu_bf16, transfers, CudaContext};

#[test]
#[ignore = "requires a broker-allocated CUDA device"]
fn batch_norm_relu_values_and_input_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let bf16 = |shape: Vec<usize>, values: &[f32]| {
        Tensor::from_bf16(
            shape,
            &values
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let upload = |t: &Tensor| transfers::to_cuda(t, 0).unwrap();
    let mean_cpu = bf16(vec![2], &[0., 1.]);
    let mean = upload(&mean_cpu);
    let invstd = upload(&Tensor::from_f32(vec![2], &[1., 2.]).unwrap());
    let weight = upload(&bf16(vec![2], &[2., 1.]));
    let bias = upload(&bf16(vec![2], &[1., -1.]));
    let x = upload(&bf16(vec![1, 2, 1, 2], &[-2., -1., 1., 2.]));
    for input in [&x, &x.reshape(vec![1, 2, 1, 1, 2]).unwrap()] {
        let y = batch_relu_bf16(&ctx, input, &mean, &invstd, &weight, &bias).unwrap();
        assert_eq!(y.shape(), input.shape());
        assert_eq!(
            transfers::to_cpu(&y).unwrap().to_f32_vec().unwrap(),
            [0., 0., 0., 1.]
        );
    }
    assert!(batch_relu_bf16(&ctx, &x, &mean_cpu, &invstd, &weight, &bias).is_err());
    assert!(batch_relu_bf16(&ctx, &x, &mean, &weight, &weight, &bias).is_err());
    let short = upload(&bf16(vec![1], &[0.]));
    assert!(batch_relu_bf16(&ctx, &x, &mean, &invstd, &weight, &short).is_err());
    let matrix = x.reshape(vec![2, 2]).unwrap();
    assert!(batch_relu_bf16(&ctx, &matrix, &mean, &invstd, &weight, &bias).is_err());
}
