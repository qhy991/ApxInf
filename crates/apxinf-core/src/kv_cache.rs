//! Backend-agnostic KV cache interface.

use crate::{Error, Result, Tensor};

/// KV cache trait for transformer attention.
///
/// Each backend implements this with its own buffer type internally.
/// Object-safe.
pub trait KvCache {
    /// Append new K/V data for a layer.
    /// k, v: [append_len, n_kv_heads, head_dim]
    fn append(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, append_len: usize)
        -> Result<()>;

    /// Advance the sequence position by n tokens.
    fn advance(&mut self, n: usize);

    /// Current sequence length in the cache.
    fn seq_len(&self) -> usize;

    /// Reset the cache for a new generation.
    fn clear(&mut self) -> Result<()>;

    /// Number of layers.
    fn n_layers(&self) -> usize;

    /// Allow backends to downcast to their concrete type.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// CPU KV cache using flat contiguous allocations.
/// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
pub struct CpuKVCache {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    seq_len: usize,
    n_layers: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl CpuKVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let elements = n_layers
            .checked_mul(n_kv_heads)
            .and_then(|value| value.checked_mul(max_seq_len))
            .and_then(|value| value.checked_mul(head_dim))
            .expect("CPU KV cache shape overflow");

        Self {
            k_cache: vec![0.0; elements],
            v_cache: vec![0.0; elements],
            seq_len: 0,
            n_layers,
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    /// Get K and V for a layer up to current position.
    /// Returns flat layer slices with logical shape
    /// [n_kv_heads, max_seq_len, head_dim].
    pub fn get_kv(&self, layer_idx: usize) -> (&[f32], &[f32]) {
        let layer_stride = self.n_kv_heads * self.max_seq_len * self.head_dim;
        let start = layer_idx * layer_stride;
        let end = start + layer_stride;
        (&self.k_cache[start..end], &self.v_cache[start..end])
    }

    /// Offset of one head/position row within a layer slice returned by
    /// [`Self::get_kv`].
    #[inline]
    pub fn row_offset(&self, head: usize, position: usize) -> usize {
        (head * self.max_seq_len + position) * self.head_dim
    }
}

impl KvCache for CpuKVCache {
    fn append(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        if layer_idx >= self.n_layers {
            return Err(Error::Other(format!(
                "KV cache layer index {layer_idx} is out of range for {} layers",
                self.n_layers
            )));
        }
        let end = self
            .seq_len
            .checked_add(append_len)
            .ok_or_else(|| Error::Other("KV cache sequence length overflow".into()))?;
        if end > self.max_seq_len {
            return Err(Error::Other(format!(
                "KV cache capacity exceeded: {end} > {}",
                self.max_seq_len
            )));
        }
        let expected_shape = [append_len, self.n_kv_heads, self.head_dim];
        if k.shape().dims() != expected_shape || v.shape().dims() != expected_shape {
            return Err(Error::Other(format!(
                "KV cache append expected K/V shape {expected_shape:?}, got {:?} and {:?}",
                k.shape().dims(),
                v.shape().dims()
            )));
        }
        let k_data = k.as_f32()?;
        let v_data = v.as_f32()?;
        let layer_stride = self.n_kv_heads * self.max_seq_len * self.head_dim;
        let layer_start = layer_idx * layer_stride;
        for s in 0..append_len {
            let pos = self.seq_len + s;
            for h in 0..self.n_kv_heads {
                let source = (s * self.n_kv_heads + h) * self.head_dim;
                let destination = layer_start + (h * self.max_seq_len + pos) * self.head_dim;
                self.k_cache[destination..destination + self.head_dim]
                    .copy_from_slice(&k_data[source..source + self.head_dim]);
                self.v_cache[destination..destination + self.head_dim]
                    .copy_from_slice(&v_data[source..source + self.head_dim]);
            }
        }
        Ok(())
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        // Cached rows at positions >= seq_len are logically unreachable.
        // The next prompt overwrites every row before attention can read it,
        // so resetting the cursor is sufficient and avoids clearing up to
        // hundreds of MiB between requests.
        self.seq_len = 0;
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuKVCache, KvCache};
    use crate::Tensor;

    #[test]
    fn append_fails_before_writing_past_capacity() {
        let mut cache = CpuKVCache::new(1, 1, 2, 2);
        let first = Tensor::from_f32(vec![2, 1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        cache.append(0, &first, &first, 2).unwrap();
        cache.advance(2);

        let extra = Tensor::from_f32(vec![1, 1, 2], &[5.0, 6.0]).unwrap();
        let error = cache.append(0, &extra, &extra, 1).unwrap_err();
        assert!(error.to_string().contains("capacity exceeded"));
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn append_rejects_layer_and_shape_mismatches() {
        let mut cache = CpuKVCache::new(1, 1, 2, 2);
        let valid = Tensor::from_f32(vec![1, 1, 2], &[1.0, 2.0]).unwrap();
        assert!(cache.append(1, &valid, &valid, 1).is_err());

        let wrong = Tensor::from_f32(vec![1, 2], &[1.0, 2.0]).unwrap();
        assert!(cache.append(0, &wrong, &wrong, 1).is_err());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn append_uses_flat_head_major_rows_and_clear_resets_the_cursor() {
        let mut cache = CpuKVCache::new(1, 2, 2, 3);
        let k =
            Tensor::from_f32(vec![2, 2, 2], &[1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]).unwrap();
        let v =
            Tensor::from_f32(vec![2, 2, 2], &[5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0]).unwrap();
        cache.append(0, &k, &v, 2).unwrap();

        let (stored_k, stored_v) = cache.get_kv(0);
        assert_eq!(
            &stored_k[cache.row_offset(0, 0)..cache.row_offset(0, 0) + 2],
            &[1.0, 2.0]
        );
        assert_eq!(
            &stored_k[cache.row_offset(0, 1)..cache.row_offset(0, 1) + 2],
            &[3.0, 4.0]
        );
        assert_eq!(
            &stored_k[cache.row_offset(1, 0)..cache.row_offset(1, 0) + 2],
            &[10.0, 20.0]
        );
        assert_eq!(
            &stored_k[cache.row_offset(1, 1)..cache.row_offset(1, 1) + 2],
            &[30.0, 40.0]
        );
        assert_eq!(
            &stored_v[cache.row_offset(1, 1)..cache.row_offset(1, 1) + 2],
            &[70.0, 80.0]
        );

        cache.clear().unwrap();
        assert_eq!(cache.seq_len(), 0);

        let replacement_k = Tensor::from_f32(vec![1, 2, 2], &[9.0, 8.0, 90.0, 80.0]).unwrap();
        let replacement_v = Tensor::from_f32(vec![1, 2, 2], &[7.0, 6.0, 70.0, 60.0]).unwrap();
        cache.append(0, &replacement_k, &replacement_v, 1).unwrap();
        let (stored_k, stored_v) = cache.get_kv(0);
        assert_eq!(
            &stored_k[cache.row_offset(0, 0)..cache.row_offset(0, 0) + 2],
            &[9.0, 8.0]
        );
        assert_eq!(
            &stored_v[cache.row_offset(1, 0)..cache.row_offset(1, 0) + 2],
            &[70.0, 60.0]
        );
    }
}
