use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::{ffi, CudaBuffer, CudaContext, CudaDeviceAddress};

pub const QUERY_HEADS: usize = 16;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const WIDTH: usize = QUERY_HEADS * HEAD_DIM;
pub const MAX_SPLITS: usize = 48;

pub struct SplitCtaWorkspace {
    partial_max: CudaBuffer,
    partial_sum: CudaBuffer,
    partial_accumulator: CudaBuffer,
}

impl SplitCtaWorkspace {
    pub fn new(ctx: &CudaContext) -> Result<Self> {
        let scalars = QUERY_HEADS * MAX_SPLITS * std::mem::size_of::<f32>();
        let accumulators = scalars * HEAD_DIM;
        Ok(Self {
            partial_max: CudaBuffer::alloc(scalars, ctx.device_id()).map_err(Error::Cuda)?,
            partial_sum: CudaBuffer::alloc(scalars, ctx.device_id()).map_err(Error::Cuda)?,
            partial_accumulator: CudaBuffer::alloc(accumulators, ctx.device_id())
                .map_err(Error::Cuda)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn split_cta_write(
    ctx: &CudaContext,
    query: &Tensor,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &Tensor,
    workspace: &SplitCtaWorkspace,
    split_count: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    if query.dtype() != DType::BF16
        || query.device() != device
        || query.shape().dims() != [1, QUERY_HEADS, HEAD_DIM]
        || output.dtype() != DType::BF16
        || output.device() != device
        || output.shape().dims() != [1, WIDTH]
    {
        return Err(Error::Other(
            "Qwen2.5-Omni split-CTA tensor contract mismatch".into(),
        ));
    }
    if ctx.caps().sm != 89
        || !matches!(split_count, 4 | 8 | 16 | 32 | 40)
        || bucket_kv_len <= 11_264
        || bucket_kv_len > max_seq_len
        || max_seq_len > i32::MAX as usize
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "Qwen2.5-Omni split-CTA launch contract mismatch".into(),
        ));
    }
    let cache_bytes = KV_HEADS
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("Qwen2.5-Omni split-CTA cache overflow".into()))?;
    for (name, buffer) in [("key", key_cache), ("value", value_cache)] {
        if buffer.device() != ctx.device_id() || buffer.len() < cache_bytes {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni split-CTA {name} cache contract mismatch"
            )));
        }
    }
    if position.device() != ctx.device_id() || position.len() < 4 {
        return Err(Error::Other(
            "Qwen2.5-Omni split-CTA position contract mismatch".into(),
        ));
    }
    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_qwen25_omni_attention_flash_split_cta_bf16(
                query.ptr(),
                key_cache.ptr(),
                value_cache.ptr(),
                workspace.partial_max.ptr(),
                workspace.partial_sum.ptr(),
                workspace.partial_accumulator.ptr(),
                output.ptr(),
                split_count as i32,
                bucket_kv_len as i32,
                max_seq_len as i32,
                scale,
                position.ptr(),
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn grouped2_split_cta_write(
    ctx: &CudaContext,
    query: &Tensor,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &Tensor,
    workspace: &SplitCtaWorkspace,
    split_count: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    if query.dtype() != DType::BF16
        || query.device() != device
        || query.shape().dims() != [1, QUERY_HEADS, HEAD_DIM]
        || output.dtype() != DType::BF16
        || output.device() != device
        || output.shape().dims() != [1, WIDTH]
    {
        return Err(Error::Other(
            "Qwen2.5-Omni grouped split-CTA tensor contract mismatch".into(),
        ));
    }
    if ctx.caps().sm != 89
        || split_count != 48
        || bucket_kv_len <= 11_264
        || bucket_kv_len > max_seq_len
        || max_seq_len > i32::MAX as usize
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "Qwen2.5-Omni grouped split-CTA launch contract mismatch".into(),
        ));
    }
    let cache_bytes = KV_HEADS
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("Qwen2.5-Omni grouped split-CTA cache overflow".into()))?;
    for (name, buffer) in [("key", key_cache), ("value", value_cache)] {
        if buffer.device() != ctx.device_id() || buffer.len() < cache_bytes {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni grouped split-CTA {name} cache contract mismatch"
            )));
        }
    }
    if position.device() != ctx.device_id() || position.len() < 4 {
        return Err(Error::Other(
            "Qwen2.5-Omni grouped split-CTA position contract mismatch".into(),
        ));
    }
    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_qwen25_omni_attention_flash_grouped2_split_cta_bf16(
                query.ptr(),
                key_cache.ptr(),
                value_cache.ptr(),
                workspace.partial_max.ptr(),
                workspace.partial_sum.ptr(),
                workspace.partial_accumulator.ptr(),
                output.ptr(),
                split_count as i32,
                bucket_kv_len as i32,
                max_seq_len as i32,
                scale,
                position.ptr(),
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)
    }
}
