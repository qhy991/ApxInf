use apxinf_core::Tensor;
use apxinf_cuda::{kernels::gemm::bf16_addmv, transfers, CudaContext};

#[test]
#[ignore = "requires a broker-allocated CUDA device"]
fn bf16_addmv_preserves_layout_bias_and_input_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let bf = |shape: Vec<usize>, values: &[f32]| {
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
    let cpu_weight = bf(vec![3, 2], &[1., 2., 3., 4., 5., 6.]);
    let weight = upload(&cpu_weight);
    let x = upload(&bf(vec![2], &[2., -3.]));
    let bias = upload(&bf(vec![3], &[0.5, -0.5, 1.]));
    let y = bf16_addmv(&ctx, &weight, &x, &bias).unwrap();
    assert_eq!(y.shape().dims(), [3]);
    assert_eq!(
        transfers::to_cpu(&y).unwrap().to_f32_vec().unwrap(),
        [-3.5, -6.5, -7.]
    );
    assert!(bf16_addmv(&ctx, &cpu_weight, &x, &bias).is_err());
    assert!(bf16_addmv(&ctx, &weight, &x.reshape(vec![1, 2]).unwrap(), &bias).is_err());
    assert!(bf16_addmv(&ctx, &weight, &x, &x).is_err());
}
