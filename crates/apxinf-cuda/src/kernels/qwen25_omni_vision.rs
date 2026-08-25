use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::check_cuda;
use crate::workspace::uninitialized_buffer;
use crate::{ffi, CudaBuffer, CudaContext};

pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 80;
pub const HIDDEN: usize = HEADS * HEAD_DIM;

#[allow(clippy::too_many_arguments)]
pub fn qkv_bias_rope(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
    theta: f32,
    positions: &CudaBuffer,
) -> Result<(Tensor, Tensor, Tensor)> {
    let sequence = query.shape().dims().first().copied().unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    let matrix_shape = [sequence, HIDDEN];
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || query.dtype() != DType::BF16
        || query.device() != device
        || query.shape().dims() != matrix_shape
        || key.dtype() != DType::BF16
        || key.device() != device
        || key.shape().dims() != matrix_shape
        || value.dtype() != DType::BF16
        || value.device() != device
        || value.shape().dims() != matrix_shape
        || [query_bias, key_bias, value_bias].iter().any(|bias| {
            bias.dtype() != DType::BF16
                || bias.device() != device
                || bias.shape().dims() != [HIDDEN]
        })
        || theta.to_bits() != 10_000.0f32.to_bits()
        || positions.device() != ctx.device_id()
        || positions.len() != sequence * 2 * std::mem::size_of::<u32>()
    {
        return Err(Error::Other(
            "Qwen2.5-Omni vision QKV bias/RoPE contract mismatch".into(),
        ));
    }

    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let query_bias = CudaBuffer::from_tensor(query_bias).map_err(Error::Cuda)?;
    let key_bias = CudaBuffer::from_tensor(key_bias).map_err(Error::Cuda)?;
    let value_bias = CudaBuffer::from_tensor(value_bias).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("vision QKV output size overflow".into()))?;
    let query_output = uninitialized_buffer(ctx, bytes)?;
    let key_output = uninitialized_buffer(ctx, bytes)?;
    let value_output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        ffi::apxinf_static_qwen25_omni_vision_qkv_bias_rope_bf16(
            query.ptr(),
            key.ptr(),
            value.ptr(),
            query_bias.ptr(),
            key_bias.ptr(),
            value_bias.ptr(),
            query_output.ptr(),
            key_output.ptr(),
            value_output.ptr(),
            sequence as i32,
            HEADS as i32,
            HEAD_DIM as i32,
            theta,
            positions.ptr(),
            ctx.stream().handle(),
        )
    })?;
    let shape = Shape::new(vec![sequence, HEADS, HEAD_DIM]);
    Ok((
        query_output.into_tensor(shape.clone(), DType::BF16),
        key_output.into_tensor(shape.clone(), DType::BF16),
        value_output.into_tensor(shape, DType::BF16),
    ))
}
