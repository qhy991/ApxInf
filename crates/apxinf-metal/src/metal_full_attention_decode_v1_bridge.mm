#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kLayerSlots = 6;
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kQueryHeads = 8;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kRotaryDim = 64;
constexpr uint32_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr uint32_t kKvWidth = kKvHeads * kHeadDim;
constexpr uint32_t kQGKVRows = 2 * kQueryWidth + 2 * kKvWidth;
constexpr uint32_t kGroupSize = 64;
constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kMatVecThreads = kRowsPerThreadgroup * 32;
constexpr uint32_t kSimdThreads = 32;
constexpr float kRmsNormEps = 1.0e-6f;
constexpr float kRopeTheta = 10'000'000.0f;

struct FullAttentionParamsV1 {
    uint32_t max_context;
    uint32_t position;
    uint32_t layer_slot;
    uint32_t reserved;
    float rms_norm_eps;
    float rope_theta;
};

struct FullAttentionRuntimeReceiptV1 {
    uint32_t layer_slots;
    uint32_t hidden_size;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t max_context;
    uint32_t group_size;
    uint32_t command_buffers_per_decode;
    uint32_t compute_encoders_per_decode;
    uint32_t kernel_dispatches_per_decode;
    uint32_t explicit_buffer_barriers_per_decode;
    uint32_t commits_per_decode;
    uint32_t waits_per_decode;
    uint32_t fixed_shape_validated;
    uint32_t reserved;
    uint64_t successful_decodes;
    uint32_t last_layer_slot;
    uint32_t last_start_pos;
    uint32_t last_kv_length;
    uint32_t last_observed_command_buffers;
    uint32_t last_observed_compute_encoders;
    uint32_t last_observed_kernel_dispatches;
    uint32_t last_observed_explicit_buffer_barriers;
    uint32_t last_observed_commits;
    uint32_t last_observed_waits;
};

struct FullAttentionObservedTopologyV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
    uint32_t commits;
    uint32_t waits;
};

static_assert(sizeof(FullAttentionParamsV1) == 24,
              "Metal full-attention parameter ABI changed");
static_assert(sizeof(FullAttentionRuntimeReceiptV1) == 112,
              "Metal full-attention receipt ABI changed");

struct ApxinfMetalW8FullAttentionStack6V1Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> input_rms_pipeline;
    id<MTLComputePipelineState> qgkv_pipeline;
    id<MTLComputePipelineState> prepare_qkv_pipeline;
    id<MTLComputePipelineState> sdpa_gate_pipeline;
    id<MTLComputePipelineState> output_residual_pipeline;

    id<MTLBuffer> qgkv_weights;
    id<MTLBuffer> qgkv_scales;
    id<MTLBuffer> output_weights;
    id<MTLBuffer> output_scales;
    id<MTLBuffer> input_rms_weight;
    id<MTLBuffer> query_norm_weight;
    id<MTLBuffer> key_norm_weight;

    id<MTLBuffer> key_cache;
    id<MTLBuffer> value_cache;
    id<MTLBuffer> input;
    id<MTLBuffer> normalized;
    id<MTLBuffer> projected;
    id<MTLBuffer> query;
    id<MTLBuffer> gated_attention;
    id<MTLBuffer> output;

    uint32_t max_context;
    uint64_t successful_decodes;
    uint32_t last_layer_slot;
    uint32_t last_start_pos;
    FullAttentionObservedTopologyV1 last_observed;
    bool has_successful_decode;
};

#include "metal_full_attention_decode_v1_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal full-attention error" : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal full-attention error"
                             : error.localizedDescription.UTF8String);
}

bool checked_product(size_t left, size_t right, size_t *output) {
    if (output == nullptr ||
        (right != 0 && left > std::numeric_limits<size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

bool finite_f32(const float *values, uint32_t count) {
    if (values == nullptr) {
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString *name,
                                          NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        if (error != nullptr && *error == nil) {
            NSString *description =
                [NSString stringWithFormat:@"Metal function %@ is unavailable", name];
            *error = [NSError errorWithDomain:@"apxinf-metal"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey : description}];
        }
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

void buffer_barrier(id<MTLComputeCommandEncoder> encoder,
                    FullAttentionObservedTopologyV1 *observed) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++observed->explicit_buffer_barriers;
}

void fill_receipt(const ApxinfMetalW8FullAttentionStack6V1Handle *handle,
                  FullAttentionRuntimeReceiptV1 *receipt) {
    std::memset(receipt, 0, sizeof(*receipt));
    receipt->layer_slots = kLayerSlots;
    receipt->hidden_size = kHiddenSize;
    receipt->query_heads = kQueryHeads;
    receipt->kv_heads = kKvHeads;
    receipt->head_dim = kHeadDim;
    receipt->rotary_dim = kRotaryDim;
    receipt->max_context = handle->max_context;
    receipt->group_size = kGroupSize;
    receipt->command_buffers_per_decode = 1;
    receipt->compute_encoders_per_decode = 1;
    receipt->kernel_dispatches_per_decode = 5;
    receipt->explicit_buffer_barriers_per_decode = 4;
    receipt->commits_per_decode = 1;
    receipt->waits_per_decode = 1;
    receipt->fixed_shape_validated = 1;
    receipt->successful_decodes = handle->successful_decodes;
    receipt->last_layer_slot =
        handle->has_successful_decode ? handle->last_layer_slot : UINT32_MAX;
    receipt->last_start_pos =
        handle->has_successful_decode ? handle->last_start_pos : UINT32_MAX;
    receipt->last_kv_length =
        handle->has_successful_decode ? handle->last_start_pos + 1 : 0;
    if (handle->has_successful_decode) {
        receipt->last_observed_command_buffers =
            handle->last_observed.command_buffers;
        receipt->last_observed_compute_encoders =
            handle->last_observed.compute_encoders;
        receipt->last_observed_kernel_dispatches =
            handle->last_observed.kernel_dispatches;
        receipt->last_observed_explicit_buffer_barriers =
            handle->last_observed.explicit_buffer_barriers;
        receipt->last_observed_commits = handle->last_observed.commits;
        receipt->last_observed_waits = handle->last_observed.waits;
    }
}

}  // namespace

extern "C" int apxinf_metal_w8_full_attention_stack6_v1_create(
    const int8_t *qgkv_weights, const float *qgkv_scales,
    const int8_t *output_weights, const float *output_scales,
    const float *input_rms_weight, const float *query_norm_weight,
    const float *key_norm_weight, uint32_t max_context, uint32_t group_size,
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal full-attention output handle is null");
            return 1;
        }
        *output = nullptr;
        if (qgkv_weights == nullptr || qgkv_scales == nullptr ||
            output_weights == nullptr || output_scales == nullptr ||
            input_rms_weight == nullptr || query_norm_weight == nullptr ||
            key_norm_weight == nullptr || max_context == 0 ||
            group_size != kGroupSize) {
            write_error(error_output, error_capacity,
                        "invalid Metal full-attention packed-weight contract");
            return 1;
        }

        size_t qgkv_weight_bytes = 0;
        size_t qgkv_scale_count = 0;
        size_t output_weight_bytes = 0;
        size_t output_scale_count = 0;
        size_t cache_elements = 0;
        if (!checked_product(static_cast<size_t>(kLayerSlots) * kQGKVRows,
                             kHiddenSize, &qgkv_weight_bytes) ||
            !checked_product(static_cast<size_t>(kLayerSlots) * kQGKVRows,
                             kHiddenSize / kGroupSize, &qgkv_scale_count) ||
            !checked_product(static_cast<size_t>(kLayerSlots) * kHiddenSize,
                             kQueryWidth, &output_weight_bytes) ||
            !checked_product(static_cast<size_t>(kLayerSlots) * kHiddenSize,
                             kQueryWidth / kGroupSize, &output_scale_count) ||
            !checked_product(static_cast<size_t>(kLayerSlots) * kKvHeads * max_context,
                             kHeadDim, &cache_elements) ||
            cache_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            write_error(error_output, error_capacity,
                        "Metal full-attention buffer dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8FullAttentionStack6V1Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal full-attention handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "no system Metal device is available");
            return 1;
        }

        NSError *error = nil;
        NSString *source =
            [NSString stringWithUTF8String:kMetalFullAttentionDecodeSourceV1];
        id<MTLLibrary> library =
            [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->input_rms_pipeline =
            make_pipeline(handle->device, library, @"full_attention_input_rms_v1", &error);
        handle->qgkv_pipeline =
            make_pipeline(handle->device, library, @"full_attention_qgkv_v1", &error);
        handle->prepare_qkv_pipeline =
            make_pipeline(handle->device, library, @"full_attention_prepare_qkv_v1", &error);
        handle->sdpa_gate_pipeline =
            make_pipeline(handle->device, library, @"full_attention_sdpa_gate_v1", &error);
        handle->output_residual_pipeline = make_pipeline(
            handle->device, library, @"full_attention_output_residual_v1", &error);
        if (handle->input_rms_pipeline == nil || handle->qgkv_pipeline == nil ||
            handle->prepare_qkv_pipeline == nil || handle->sdpa_gate_pipeline == nil ||
            handle->output_residual_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        for (id<MTLComputePipelineState> pipeline in
             @[ handle->input_rms_pipeline, handle->qgkv_pipeline,
                handle->prepare_qkv_pipeline, handle->sdpa_gate_pipeline,
                handle->output_residual_pipeline ]) {
            if (pipeline.threadExecutionWidth != kSimdThreads ||
                pipeline.maxTotalThreadsPerThreadgroup < kSimdThreads) {
                delete handle;
                write_error(error_output, error_capacity,
                            "Metal full-attention pipelines require SIMD width 32");
                return 1;
            }
        }
        if (handle->qgkv_pipeline.maxTotalThreadsPerThreadgroup < kMatVecThreads ||
            handle->output_residual_pipeline.maxTotalThreadsPerThreadgroup < kMatVecThreads) {
            delete handle;
            write_error(error_output, error_capacity,
                        "Metal full-attention matvec pipelines require 256 threads");
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        const MTLResourceOptions private_storage = MTLResourceStorageModePrivate;
        handle->qgkv_weights = [handle->device newBufferWithBytes:qgkv_weights
                                                          length:qgkv_weight_bytes
                                                         options:shared];
        handle->qgkv_scales = [handle->device newBufferWithBytes:qgkv_scales
                                                         length:qgkv_scale_count * sizeof(float)
                                                        options:shared];
        handle->output_weights = [handle->device newBufferWithBytes:output_weights
                                                            length:output_weight_bytes
                                                           options:shared];
        handle->output_scales = [handle->device newBufferWithBytes:output_scales
                                                           length:output_scale_count * sizeof(float)
                                                          options:shared];
        handle->input_rms_weight = [handle->device newBufferWithBytes:input_rms_weight
                                                               length:static_cast<size_t>(kLayerSlots) *
                                                                      kHiddenSize * sizeof(float)
                                                              options:shared];
        handle->query_norm_weight = [handle->device newBufferWithBytes:query_norm_weight
                                                                length:static_cast<size_t>(kLayerSlots) *
                                                                       kHeadDim * sizeof(float)
                                                               options:shared];
        handle->key_norm_weight = [handle->device newBufferWithBytes:key_norm_weight
                                                              length:static_cast<size_t>(kLayerSlots) *
                                                                     kHeadDim * sizeof(float)
                                                             options:shared];
        const size_t cache_bytes = cache_elements * sizeof(float);
        handle->key_cache = [handle->device newBufferWithLength:cache_bytes options:shared];
        handle->value_cache = [handle->device newBufferWithLength:cache_bytes options:shared];
        handle->input = [handle->device newBufferWithLength:kHiddenSize * sizeof(float)
                                                    options:shared];
        handle->normalized = [handle->device newBufferWithLength:kHiddenSize * sizeof(float)
                                                         options:private_storage];
        handle->projected = [handle->device newBufferWithLength:kQGKVRows * sizeof(float)
                                                        options:private_storage];
        handle->query = [handle->device newBufferWithLength:kQueryWidth * sizeof(float)
                                                    options:private_storage];
        handle->gated_attention =
            [handle->device newBufferWithLength:kQueryWidth * sizeof(float)
                                         options:private_storage];
        handle->output = [handle->device newBufferWithLength:kHiddenSize * sizeof(float)
                                                     options:shared];
        if (handle->queue == nil || handle->qgkv_weights == nil ||
            handle->qgkv_scales == nil || handle->output_weights == nil ||
            handle->output_scales == nil || handle->input_rms_weight == nil ||
            handle->query_norm_weight == nil || handle->key_norm_weight == nil ||
            handle->key_cache == nil || handle->value_cache == nil ||
            handle->input == nil || handle->normalized == nil ||
            handle->projected == nil || handle->query == nil ||
            handle->gated_attention == nil || handle->output == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal full-attention buffers failed");
            return 1;
        }
        std::memset(handle->key_cache.contents, 0, cache_bytes);
        std::memset(handle->value_cache.contents, 0, cache_bytes);
        handle->max_context = max_context;
        handle->successful_decodes = 0;
        handle->last_layer_slot = UINT32_MAX;
        handle->last_start_pos = UINT32_MAX;
        handle->last_observed = FullAttentionObservedTopologyV1{};
        handle->has_successful_decode = false;
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_full_attention_stack6_v1_seed_cache(
    void *opaque_handle, uint32_t layer_slot, uint32_t start_pos,
    const float *keys, uint32_t key_count, const float *values,
    uint32_t value_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8FullAttentionStack6V1Handle *>(opaque_handle);
        const uint64_t expected =
            static_cast<uint64_t>(start_pos) * kKvHeads * kHeadDim;
        if (handle == nullptr || layer_slot >= kLayerSlots ||
            start_pos > (handle == nullptr ? 0 : handle->max_context) ||
            expected > UINT32_MAX || key_count != expected || value_count != expected ||
            (expected != 0 && (keys == nullptr || values == nullptr)) ||
            (expected != 0 &&
             (!finite_f32(keys, key_count) || !finite_f32(values, value_count)))) {
            write_error(error_output, error_capacity,
                        "invalid Metal full-attention cache seed");
            return 1;
        }
        if (expected == 0) {
            return 0;
        }
        float *key_cache = static_cast<float *>(handle->key_cache.contents);
        float *value_cache = static_cast<float *>(handle->value_cache.contents);
        for (uint32_t token = 0; token < start_pos; ++token) {
            for (uint32_t head = 0; head < kKvHeads; ++head) {
                const size_t source =
                    (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
                const size_t destination =
                    ((static_cast<size_t>(layer_slot) * kKvHeads + head) *
                         handle->max_context +
                     token) *
                    kHeadDim;
                std::memcpy(key_cache + destination, keys + source,
                            kHeadDim * sizeof(float));
                std::memcpy(value_cache + destination, values + source,
                            kHeadDim * sizeof(float));
            }
        }
        return 0;
    }
}

extern "C" int apxinf_metal_w8_full_attention_stack6_v1_decode(
    void *opaque_handle, uint32_t layer_slot, const float *input,
    uint32_t input_count, uint32_t start_pos, float *output,
    uint32_t output_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8FullAttentionStack6V1Handle *>(opaque_handle);
        if (handle == nullptr || layer_slot >= kLayerSlots || input == nullptr ||
            output == nullptr || input_count != kHiddenSize ||
            output_count != kHiddenSize || start_pos >= handle->max_context ||
            !finite_f32(input, input_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal full-attention decode input");
            return 1;
        }
        std::memcpy(handle->input.contents, input, kHiddenSize * sizeof(float));
        const FullAttentionParamsV1 params{
            handle->max_context, start_pos, layer_slot, 0, kRmsNormEps, kRopeTheta};
        FullAttentionObservedTopologyV1 observed{};

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal full-attention command buffer failed");
            return 1;
        }
        ++observed.command_buffers;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal full-attention compute encoder failed");
            return 1;
        }
        ++observed.compute_encoders;

        [encoder setComputePipelineState:handle->input_rms_pipeline];
        [encoder setBuffer:handle->input offset:0 atIndex:0];
        [encoder setBuffer:handle->input_rms_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kSimdThreads, 1, 1)];
        ++observed.kernel_dispatches;
        buffer_barrier(encoder, &observed);

        [encoder setComputePipelineState:handle->qgkv_pipeline];
        [encoder setBuffer:handle->qgkv_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->qgkv_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBuffer:handle->projected offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (kQGKVRows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup,
                    1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        ++observed.kernel_dispatches;
        buffer_barrier(encoder, &observed);

        [encoder setComputePipelineState:handle->prepare_qkv_pipeline];
        [encoder setBuffer:handle->projected offset:0 atIndex:0];
        [encoder setBuffer:handle->query_norm_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->key_norm_weight offset:0 atIndex:2];
        [encoder setBuffer:handle->query offset:0 atIndex:3];
        [encoder setBuffer:handle->key_cache offset:0 atIndex:4];
        [encoder setBuffer:handle->value_cache offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(params) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(kQueryHeads + kKvHeads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kSimdThreads, 1, 1)];
        ++observed.kernel_dispatches;
        buffer_barrier(encoder, &observed);

        [encoder setComputePipelineState:handle->sdpa_gate_pipeline];
        [encoder setBuffer:handle->query offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:handle->key_cache offset:0 atIndex:2];
        [encoder setBuffer:handle->value_cache offset:0 atIndex:3];
        [encoder setBuffer:handle->gated_attention offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(params) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(kQueryHeads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kSimdThreads, 1, 1)];
        ++observed.kernel_dispatches;
        buffer_barrier(encoder, &observed);

        [encoder setComputePipelineState:handle->output_residual_pipeline];
        [encoder setBuffer:handle->output_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->output_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->gated_attention offset:0 atIndex:2];
        [encoder setBuffer:handle->input offset:0 atIndex:3];
        [encoder setBuffer:handle->output offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(params) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (kHiddenSize + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup,
                    1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        ++observed.kernel_dispatches;
        [encoder endEncoding];

        [command commit];
        ++observed.commits;
        [command waitUntilCompleted];
        ++observed.waits;
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (observed.command_buffers != 1 || observed.compute_encoders != 1 ||
            observed.kernel_dispatches != 5 ||
            observed.explicit_buffer_barriers != 4 || observed.commits != 1 ||
            observed.waits != 1) {
            write_error(error_output, error_capacity,
                        "Metal full-attention observed topology violated its contract");
            return 1;
        }
        const float *device_output = static_cast<const float *>(handle->output.contents);
        if (!finite_f32(device_output, kHiddenSize)) {
            write_error(error_output, error_capacity,
                        "Metal full-attention decode produced non-finite output");
            return 1;
        }
        std::memcpy(output, device_output, kHiddenSize * sizeof(float));
        ++handle->successful_decodes;
        handle->last_layer_slot = layer_slot;
        handle->last_start_pos = start_pos;
        handle->last_observed = observed;
        handle->has_successful_decode = true;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_full_attention_stack6_v1_receipt(
    void *opaque_handle, FullAttentionRuntimeReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8FullAttentionStack6V1Handle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "invalid Metal full-attention receipt request");
            return 1;
        }
        fill_receipt(handle, receipt);
        return 0;
    }
}

extern "C" int apxinf_metal_w8_full_attention_stack6_v1_snapshot_cache_row(
    void *opaque_handle, uint32_t layer_slot, uint32_t position,
    float *key_output, uint32_t key_count, float *value_output,
    uint32_t value_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8FullAttentionStack6V1Handle *>(opaque_handle);
        if (handle == nullptr || layer_slot >= kLayerSlots ||
            position >= (handle == nullptr ? 0 : handle->max_context) ||
            key_output == nullptr || value_output == nullptr ||
            key_count != kKvWidth || value_count != kKvWidth) {
            write_error(error_output, error_capacity,
                        "invalid Metal full-attention cache-row snapshot");
            return 1;
        }
        const float *key_cache =
            static_cast<const float *>(handle->key_cache.contents);
        const float *value_cache =
            static_cast<const float *>(handle->value_cache.contents);
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const size_t source =
                ((static_cast<size_t>(layer_slot) * kKvHeads + head) *
                     handle->max_context +
                 position) *
                kHeadDim;
            const size_t destination = static_cast<size_t>(head) * kHeadDim;
            std::memcpy(key_output + destination, key_cache + source,
                        kHeadDim * sizeof(float));
            std::memcpy(value_output + destination, value_cache + source,
                        kHeadDim * sizeof(float));
        }
        return 0;
    }
}

extern "C" void apxinf_metal_w8_full_attention_stack6_v1_destroy(
    void *opaque_handle) {
    auto handle =
        static_cast<ApxinfMetalW8FullAttentionStack6V1Handle *>(opaque_handle);
    delete handle;
}
