//! Qwen2.5-Omni BF16 decode workspace and CUDA Graph replay.

#![cfg(feature = "cuda")]

use apxinf_core::{Backend, DType, Error, Graph, KvCache, Result, Shape, Tensor};

use crate::accelerator::cuda::{
    kernels, Context as CudaContext, DeviceBuffer as CudaBuffer, KvCache as CudaKVCache,
    MappedBuffer as HostMappedBuffer, RuntimeBackend as CudaBackend,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen25OmniDecodeGraphConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub rms_norm_eps: f32,
}

pub struct Qwen25OmniDecodeGraphWeights<'a> {
    pub token_embedding: &'a Tensor,
    pub layers: Vec<Qwen25OmniDecodeLayerWeights<'a>>,
    pub output_norm: &'a Tensor,
    pub lm_head: &'a Tensor,
}

pub struct Qwen25OmniDecodeLayerWeights<'a> {
    pub attn_norm: &'a Tensor,
    pub wq: &'a Tensor,
    pub bq: &'a Tensor,
    pub wk: &'a Tensor,
    pub bk: &'a Tensor,
    pub wv: &'a Tensor,
    pub bv: &'a Tensor,
    pub wo: &'a Tensor,
    pub ffn_norm: &'a Tensor,
    pub w_gate: &'a Tensor,
    pub w_up: &'a Tensor,
    pub w_down: &'a Tensor,
}

struct DecodeWorkspace {
    x: CudaBuffer,
    norm: CudaBuffer,
    q: CudaBuffer,
    k: CudaBuffer,
    v: CudaBuffer,
    q_rope: CudaBuffer,
    k_rope: CudaBuffer,
    attn_out: CudaBuffer,
    attn_proj: CudaBuffer,
    ffn_norm: CudaBuffer,
    gate: CudaBuffer,
    gate_silu: CudaBuffer,
    up: CudaBuffer,
    mlp_hidden: CudaBuffer,
    mlp_out: CudaBuffer,
    logits: CudaBuffer,
    token: HostMappedBuffer,
    positions: HostMappedBuffer,
    selected_token: HostMappedBuffer,
    argmax_partials: CudaBuffer,
}

impl DecodeWorkspace {
    fn new(device: usize, config: &Qwen25OmniDecodeGraphConfig) -> Result<Self> {
        let element_bytes = DType::BF16.size_in_bytes();
        let allocate = |elements: usize| {
            CudaBuffer::alloc_zeros(elements * element_bytes, device).map_err(Error::Cuda)
        };
        let hidden = config.hidden_size;
        let kv = config.n_kv_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        Ok(Self {
            x: allocate(hidden)?,
            norm: allocate(hidden)?,
            q: allocate(hidden)?,
            k: allocate(kv)?,
            v: allocate(kv)?,
            q_rope: allocate(hidden)?,
            k_rope: allocate(kv)?,
            attn_out: allocate(hidden)?,
            attn_proj: allocate(hidden)?,
            ffn_norm: allocate(hidden)?,
            gate: allocate(intermediate)?,
            gate_silu: allocate(intermediate)?,
            up: allocate(intermediate)?,
            mlp_hidden: allocate(intermediate)?,
            mlp_out: allocate(hidden)?,
            logits: allocate(config.vocab_size)?,
            token: HostMappedBuffer::alloc(4, device).map_err(Error::Cuda)?,
            // [tmrope_t, tmrope_h, tmrope_w, linear_cache_position]
            positions: HostMappedBuffer::alloc(16, device).map_err(Error::Cuda)?,
            selected_token: HostMappedBuffer::alloc(4, device).map_err(Error::Cuda)?,
            argmax_partials: CudaBuffer::alloc_zeros(
                kernels::selection::ARGMAX_PARTIAL_BYTES,
                device,
            )
            .map_err(Error::Cuda)?,
        })
    }
}

fn weight_view(tensor: &Tensor, device: usize) -> Result<CudaBuffer> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.device() != device {
        return Err(Error::Other(format!(
            "decode graph weight is on CUDA {}, expected CUDA {device}",
            buffer.device()
        )));
    }
    Ok(buffer)
}

fn read_logits(workspace: &DecodeWorkspace, vocab_size: usize) -> Result<Tensor> {
    let mut bytes = vec![0u8; vocab_size * DType::BF16.size_in_bytes()];
    workspace
        .logits
        .copy_to_host(&mut bytes)
        .map_err(Error::Cuda)?;
    let bf16 = Tensor::from_raw(
        Shape::new(vec![1, vocab_size]),
        DType::BF16,
        apxinf_core::Device::Cpu,
        bytes,
    )?;
    Tensor::from_f32(vec![1, vocab_size], &bf16.to_f32_vec()?)
}

fn decode_forward_capturable(
    context: &CudaContext,
    workspace: &DecodeWorkspace,
    weights: &Qwen25OmniDecodeGraphWeights<'_>,
    cache: &mut dyn KvCache,
    config: &Qwen25OmniDecodeGraphConfig,
    select_token: bool,
) -> Result<()> {
    if weights.layers.len() != config.n_layers || config.n_layers == 0 {
        return Err(Error::Other(
            "Qwen2.5-Omni decode graph layer count mismatch".into(),
        ));
    }
    let device = context.device_id();
    let hidden = config.hidden_size;
    let kv_width = config.n_kv_heads * config.head_dim;
    let intermediate = config.intermediate_size;
    let positions = workspace.positions.address_at(0, 12).map_err(Error::Cuda)?;
    let cache_position = workspace.positions.address_at(12, 4).map_err(Error::Cuda)?;
    let cache = cache
        .as_any_mut()
        .downcast_mut::<CudaKVCache>()
        .ok_or_else(|| Error::Other("Qwen2.5-Omni decode graph requires CudaKVCache".into()))?;

    kernels::embedding::lookup_into(
        context,
        DType::BF16,
        &weight_view(weights.token_embedding, device)?,
        workspace.token.address(),
        &workspace.x,
        hidden,
        1,
    )?;

    for (index, layer) in weights.layers.iter().enumerate() {
        kernels::norm::rms_into(
            context,
            DType::BF16,
            &workspace.x,
            &weight_view(layer.attn_norm, device)?,
            &workspace.norm,
            hidden,
            1,
            config.rms_norm_eps,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            hidden,
            hidden,
            1.0,
            &workspace.norm,
            &weight_view(layer.wq, device)?,
            0.0,
            &workspace.q,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            kv_width,
            hidden,
            1.0,
            &workspace.norm,
            &weight_view(layer.wk, device)?,
            0.0,
            &workspace.k,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            kv_width,
            hidden,
            1.0,
            &workspace.norm,
            &weight_view(layer.wv, device)?,
            0.0,
            &workspace.v,
        )?;
        kernels::elementwise::add_bias_bf16_into(
            context,
            &workspace.q,
            &weight_view(layer.bq, device)?,
            &workspace.q,
            hidden,
            1,
        )?;
        kernels::elementwise::add_bias_bf16_into(
            context,
            &workspace.k,
            &weight_view(layer.bk, device)?,
            &workspace.k,
            kv_width,
            1,
        )?;
        kernels::elementwise::add_bias_bf16_into(
            context,
            &workspace.v,
            &weight_view(layer.bv, device)?,
            &workspace.v,
            kv_width,
            1,
        )?;
        kernels::rope::apply_tmrope_bf16_into(
            context,
            &workspace.q,
            &workspace.q_rope,
            config.head_dim,
            config.n_heads,
            config.rope_theta,
            config.mrope_section,
            positions,
        )?;
        kernels::rope::apply_tmrope_bf16_into(
            context,
            &workspace.k,
            &workspace.k_rope,
            config.head_dim,
            config.n_kv_heads,
            config.rope_theta,
            config.mrope_section,
            positions,
        )?;
        kernels::cache::append_at(
            context,
            DType::BF16,
            cache.k_buffer(index),
            &workspace.k_rope,
            config.n_kv_heads,
            config.head_dim,
            config.max_seq_len,
            cache_position,
        )?;
        kernels::cache::append_at(
            context,
            DType::BF16,
            cache.v_buffer(index),
            &workspace.v,
            config.n_kv_heads,
            config.head_dim,
            config.max_seq_len,
            cache_position,
        )?;
        kernels::attention::flash_bf16_into(
            context,
            &workspace.q_rope,
            cache.k_buffer(index),
            cache.v_buffer(index),
            &workspace.attn_out,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.max_seq_len,
            config.max_seq_len,
            (config.head_dim as f32).sqrt().recip(),
            cache_position,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            hidden,
            hidden,
            1.0,
            &workspace.attn_out,
            &weight_view(layer.wo, device)?,
            0.0,
            &workspace.attn_proj,
        )?;
        kernels::elementwise::add_into(
            context,
            DType::BF16,
            &workspace.x,
            &workspace.attn_proj,
            &workspace.x,
            hidden,
        )?;
        kernels::norm::rms_into(
            context,
            DType::BF16,
            &workspace.x,
            &weight_view(layer.ffn_norm, device)?,
            &workspace.ffn_norm,
            hidden,
            1,
            config.rms_norm_eps,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            intermediate,
            hidden,
            1.0,
            &workspace.ffn_norm,
            &weight_view(layer.w_gate, device)?,
            0.0,
            &workspace.gate,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            intermediate,
            hidden,
            1.0,
            &workspace.ffn_norm,
            &weight_view(layer.w_up, device)?,
            0.0,
            &workspace.up,
        )?;
        kernels::activation::silu_into(
            context,
            DType::BF16,
            &workspace.gate,
            &workspace.gate_silu,
            intermediate,
        )?;
        kernels::elementwise::mul_into(
            context,
            DType::BF16,
            &workspace.gate_silu,
            &workspace.up,
            &workspace.mlp_hidden,
            intermediate,
        )?;
        kernels::gemm::write(
            context,
            DType::BF16,
            1,
            hidden,
            intermediate,
            1.0,
            &workspace.mlp_hidden,
            &weight_view(layer.w_down, device)?,
            0.0,
            &workspace.mlp_out,
        )?;
        kernels::elementwise::add_into(
            context,
            DType::BF16,
            &workspace.x,
            &workspace.mlp_out,
            &workspace.x,
            hidden,
        )?;
    }

    kernels::norm::rms_into(
        context,
        DType::BF16,
        &workspace.x,
        &weight_view(weights.output_norm, device)?,
        &workspace.norm,
        hidden,
        1,
        config.rms_norm_eps,
    )?;
    kernels::gemm::write(
        context,
        DType::BF16,
        1,
        config.vocab_size,
        hidden,
        1.0,
        &workspace.norm,
        &weight_view(weights.lm_head, device)?,
        0.0,
        &workspace.logits,
    )?;
    if select_token {
        kernels::selection::argmax_bf16_into(
            context,
            &workspace.logits,
            &workspace.argmax_partials,
            workspace.selected_token.address(),
            config.vocab_size,
        )?;
    }
    Ok(())
}

pub struct Qwen25OmniDecodeGraph {
    config: Qwen25OmniDecodeGraphConfig,
    workspace: DecodeWorkspace,
    graph: Option<Box<dyn Graph>>,
    select_token: bool,
}

impl Qwen25OmniDecodeGraph {
    pub fn new(
        backend: &CudaBackend,
        config: Qwen25OmniDecodeGraphConfig,
        select_token: bool,
    ) -> Result<Self> {
        if config.n_layers == 0
            || config.n_heads == 0
            || config.n_kv_heads == 0
            || config.n_heads % config.n_kv_heads != 0
            || config.mrope_section.iter().sum::<usize>() != config.head_dim / 2
        {
            return Err(Error::Other(
                "invalid Qwen2.5-Omni decode graph configuration".into(),
            ));
        }
        Ok(Self {
            workspace: DecodeWorkspace::new(backend.device_id(), &config)?,
            config,
            graph: None,
            select_token,
        })
    }

    pub fn prewarm(
        &mut self,
        backend: &CudaBackend,
        weights: &Qwen25OmniDecodeGraphWeights<'_>,
        cache: &mut dyn KvCache,
    ) -> Result<()> {
        if self.graph.is_some() {
            return Ok(());
        }
        self.workspace.token.write_u32(0).map_err(Error::Cuda)?;
        self.workspace
            .positions
            .write_u32s(&[0, 0, 0, 0])
            .map_err(Error::Cuda)?;
        decode_forward_capturable(
            backend.context(),
            &self.workspace,
            weights,
            cache,
            &self.config,
            self.select_token,
        )?;
        backend.synchronize()?;
        cache.clear()?;

        backend.begin_capture_relaxed()?;
        let capture = decode_forward_capturable(
            backend.context(),
            &self.workspace,
            weights,
            cache,
            &self.config,
            self.select_token,
        );
        let graph = backend.end_capture()?;
        capture?;
        self.graph = Some(graph);
        Ok(())
    }

    fn execute(
        &mut self,
        backend: &CudaBackend,
        weights: &Qwen25OmniDecodeGraphWeights<'_>,
        cache: &mut dyn KvCache,
        token: u32,
        positions: [u32; 3],
        cache_position: u32,
    ) -> Result<()> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| Error::Other("Qwen2.5-Omni decode graph was not prewarmed".into()))?;
        self.workspace.token.write_u32(token).map_err(Error::Cuda)?;
        self.workspace
            .positions
            .write_u32s(&[positions[0], positions[1], positions[2], cache_position])
            .map_err(Error::Cuda)?;
        if std::env::var("APXINF_NO_GRAPH")
            .map(|value| !value.is_empty())
            .unwrap_or(false)
        {
            decode_forward_capturable(
                backend.context(),
                &self.workspace,
                weights,
                cache,
                &self.config,
                self.select_token,
            )?;
        } else {
            graph.replay()?;
        }
        backend.synchronize()
    }

    pub fn decode(
        &mut self,
        backend: &CudaBackend,
        weights: &Qwen25OmniDecodeGraphWeights<'_>,
        cache: &mut dyn KvCache,
        token: u32,
        positions: [u32; 3],
        cache_position: u32,
    ) -> Result<Tensor> {
        self.execute(backend, weights, cache, token, positions, cache_position)?;
        read_logits(&self.workspace, self.config.vocab_size)
    }

    pub fn selects_token(&self) -> bool {
        self.select_token
    }

    pub fn decode_token(
        &mut self,
        backend: &CudaBackend,
        weights: &Qwen25OmniDecodeGraphWeights<'_>,
        cache: &mut dyn KvCache,
        token: u32,
        positions: [u32; 3],
        cache_position: u32,
    ) -> Result<u32> {
        if !self.select_token {
            return Err(Error::Other(
                "Qwen2.5-Omni GPU token selection is disabled".into(),
            ));
        }
        self.execute(backend, weights, cache, token, positions, cache_position)?;
        self.workspace
            .selected_token
            .read_u32()
            .map_err(Error::Cuda)
    }
}
