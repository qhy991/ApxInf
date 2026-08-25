#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kGroupSize = 64;
constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kMatVecThreads = kRowsPerThreadgroup * 32;
constexpr uint32_t kElementThreads = 256;
constexpr uint32_t kTopK = 4;
constexpr uint32_t kTopKThreads = 256;
constexpr uint32_t kAllOutputsMask = 0b11;

struct LinearLayerParams {
    uint32_t hidden_size;
    float rms_norm_eps;
};

struct MlpParams {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t gate_up_groups_per_row;
    uint32_t down_groups_per_row;
};

struct KernelParams {
    uint32_t columns;
    uint32_t rows;
    uint32_t groups_per_row;
    uint32_t partial_count;
};

struct TailDescriptorV1 {
    const int8_t *gate_up_weights;
    const float *gate_up_scales;
    const int8_t *down_weights;
    const float *down_scales;
    const float *post_attention_rms_weight;
    const float *final_rms_weight;
    const int8_t *vocab_weights;
    const float *vocab_scales;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t vocab_size;
    float rms_norm_eps;
};

struct TailExecutionReceiptV1 {
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t buffer_barriers;
    uint32_t commits;
    uint32_t waits;
    uint32_t output_commits;
    uint32_t output_commit_mask;
};

struct TailVocabStorageReceiptV1 {
    uint32_t vocab_storage;
    uint32_t vocab_weights_storage;
    uint32_t vocab_scales_storage;
    uint32_t transient_staging_buffers;
    uint64_t vocab_weight_bytes;
    uint64_t vocab_scale_bytes;
    uint64_t transient_staging_bytes;
    uint64_t init_blit_bytes;
    uint32_t init_command_buffers;
    uint32_t init_blit_encoders;
    uint32_t init_copy_commands;
    uint32_t init_commits;
    uint32_t init_waits;
};

struct ApxinfMetalW8TailMlpHeadHandleV1 {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> rms_pipeline;
    id<MTLComputePipelineState> gate_up_pipeline;
    id<MTLComputePipelineState> activation_pipeline;
    id<MTLComputePipelineState> down_pipeline;
    id<MTLComputePipelineState> residual_pipeline;
    id<MTLComputePipelineState> rows_topk_pipeline;
    id<MTLComputePipelineState> final_topk_pipeline;
    id<MTLBuffer> rms_weights;
    id<MTLBuffer> gate_up_weights;
    id<MTLBuffer> gate_up_scales;
    id<MTLBuffer> down_weights;
    id<MTLBuffer> down_scales;
    id<MTLBuffer> vocab_weights;
    id<MTLBuffer> vocab_scales;
    id<MTLBuffer> hidden_a;
    id<MTLBuffer> hidden_b;
    id<MTLBuffer> output_tokens;
    id<MTLBuffer> gate_up;
    id<MTLBuffer> activated;
    id<MTLBuffer> partial_topk;
    LinearLayerParams layer_params;
    MlpParams mlp_params;
    KernelParams head_params;
    uint32_t vocab_storage;
    uint32_t transient_staging_buffers;
    uint64_t transient_staging_bytes;
    uint64_t init_blit_bytes;
    uint32_t init_command_buffers;
    uint32_t init_blit_encoders;
    uint32_t init_copy_commands;
    uint32_t init_commits;
    uint32_t init_waits;
    bool terminal_error;
};

#include "metal_w8_tail_mlp_head_v1_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(output, capacity, "%s",
                  message == nullptr ? "unknown Metal error" : message);
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal error"
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

bool checked_sum(size_t left, size_t right, size_t *output) {
    if (output == nullptr || left > std::numeric_limits<size_t>::max() - right) {
        return false;
    }
    *output = left + right;
    return true;
}

uint32_t normalized_vocab_storage(id<MTLBuffer> buffer) {
    if (buffer == nil) {
        return UINT32_MAX;
    }
    switch (buffer.storageMode) {
        case MTLStorageModeShared:
            return 0;
        case MTLStorageModePrivate:
            return 1;
        default:
            return UINT32_MAX;
    }
}

bool all_finite(const float *values, size_t count) {
    if (values == nullptr) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

bool all_positive_finite(const float *values, size_t count) {
    if (!all_finite(values, count)) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (values[index] <= 0.0f) {
            return false;
        }
    }
    return true;
}

bool valid_candidates(const uint32_t *tokens, uint32_t vocab_size) {
    if (tokens == nullptr) {
        return false;
    }
    for (uint32_t index = 0; index < kTopK; ++index) {
        if (tokens[index] >= vocab_size) {
            return false;
        }
        for (uint32_t earlier = 0; earlier < index; ++earlier) {
            if (tokens[index] == tokens[earlier]) {
                return false;
            }
        }
    }
    return true;
}

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
    NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

void buffer_barrier(id<MTLComputeCommandEncoder> encoder) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

bool buffers_valid(const ApxinfMetalW8TailMlpHeadHandleV1 *handle) {
    return handle->rms_weights != nil && handle->gate_up_weights != nil &&
           handle->gate_up_scales != nil && handle->down_weights != nil &&
           handle->down_scales != nil && handle->vocab_weights != nil &&
           handle->vocab_scales != nil && handle->hidden_a != nil &&
           handle->hidden_b != nil && handle->output_tokens != nil &&
           handle->gate_up != nil && handle->activated != nil &&
           handle->partial_topk != nil;
}

void encode_tail(ApxinfMetalW8TailMlpHeadHandleV1 *handle,
                 id<MTLComputeCommandEncoder> encoder) {
    const size_t hidden_bytes =
        static_cast<size_t>(handle->layer_params.hidden_size) * sizeof(float);

    [encoder setComputePipelineState:handle->rms_pipeline];
    [encoder setBuffer:handle->hidden_a offset:0 atIndex:0];
    [encoder setBuffer:handle->rms_weights offset:0 atIndex:1];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:2];
    [encoder setBytes:&handle->layer_params
                length:sizeof(handle->layer_params)
               atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gate_up_pipeline];
    [encoder setBuffer:handle->gate_up_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->gate_up_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:2];
    [encoder setBuffer:handle->gate_up offset:0 atIndex:3];
    [encoder setBytes:&handle->mlp_params
                length:sizeof(handle->mlp_params)
               atIndex:4];
    const uint32_t gate_up_rows = handle->mlp_params.intermediate_size * 2;
    [encoder dispatchThreadgroups:MTLSizeMake(
                (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->activation_pipeline];
    [encoder setBuffer:handle->gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->activated offset:0 atIndex:1];
    [encoder setBytes:&handle->mlp_params
                length:sizeof(handle->mlp_params)
               atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(handle->mlp_params.intermediate_size, 1,
                                         1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->down_pipeline];
    [encoder setBuffer:handle->down_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->down_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->activated offset:0 atIndex:2];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:3];
    [encoder setBytes:&handle->mlp_params
                length:sizeof(handle->mlp_params)
               atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                (handle->mlp_params.hidden_size + kRowsPerThreadgroup - 1) /
                    kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:handle->hidden_a offset:0 atIndex:0];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:1];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:2];
    [encoder setBytes:&handle->layer_params
                length:sizeof(handle->layer_params)
               atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(handle->layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->rms_pipeline];
    [encoder setBuffer:handle->hidden_b offset:0 atIndex:0];
    [encoder setBuffer:handle->rms_weights offset:hidden_bytes atIndex:1];
    [encoder setBuffer:handle->hidden_a offset:0 atIndex:2];
    [encoder setBytes:&handle->layer_params
                length:sizeof(handle->layer_params)
               atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->rows_topk_pipeline];
    [encoder setBuffer:handle->vocab_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->vocab_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->hidden_a offset:0 atIndex:2];
    [encoder setBuffer:handle->partial_topk offset:0 atIndex:3];
    [encoder setBytes:&handle->head_params
                length:sizeof(handle->head_params)
               atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(handle->head_params.partial_count,
                                              1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->final_topk_pipeline];
    [encoder setBuffer:handle->partial_topk offset:0 atIndex:0];
    [encoder setBuffer:handle->output_tokens offset:0 atIndex:1];
    [encoder setBytes:&handle->head_params
                length:sizeof(handle->head_params)
               atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kTopKThreads, 1, 1)];
}

int create_tail_mlp_head_v1(
    const TailDescriptorV1 *descriptor, void **output, char *error_output,
    size_t error_capacity, uint32_t vocab_storage) {
    if (output == nullptr) {
        write_error(error_output, error_capacity,
                    "Metal W8 tail MLP+head v1 output handle is null");
        return 1;
    }
    *output = nullptr;
    if (vocab_storage > 1) {
        write_error(error_output, error_capacity,
                    "invalid Metal W8 tail MLP+head v1 vocab storage selector");
        return 1;
    }
    if (descriptor == nullptr || descriptor->gate_up_weights == nullptr ||
        descriptor->gate_up_scales == nullptr ||
        descriptor->down_weights == nullptr ||
        descriptor->down_scales == nullptr ||
        descriptor->post_attention_rms_weight == nullptr ||
        descriptor->final_rms_weight == nullptr ||
        descriptor->vocab_weights == nullptr ||
        descriptor->vocab_scales == nullptr || descriptor->hidden_size == 0 ||
        descriptor->intermediate_size == 0 || descriptor->vocab_size < kTopK ||
        descriptor->hidden_size % kGroupSize != 0 ||
        descriptor->intermediate_size % kGroupSize != 0 ||
        descriptor->intermediate_size > UINT32_MAX / 2 ||
        !std::isfinite(descriptor->rms_norm_eps) ||
        descriptor->rms_norm_eps < 0.0f) {
        write_error(error_output, error_capacity,
                    "invalid Metal W8 tail MLP+head v1 packed contract");
        return 1;
    }

    const size_t hidden_size = descriptor->hidden_size;
    const size_t intermediate_size = descriptor->intermediate_size;
    const size_t vocab_size = descriptor->vocab_size;
    size_t hidden_intermediate = 0;
    size_t gate_up_weight_bytes = 0;
    size_t gate_up_scale_count = 0;
    size_t down_scale_count = 0;
    size_t vocab_weight_bytes = 0;
    size_t vocab_scale_count = 0;
    size_t vocab_scale_bytes = 0;
    size_t vocab_init_bytes = 0;
    if (!checked_product(hidden_size, intermediate_size,
                         &hidden_intermediate) ||
        !checked_product(hidden_intermediate, 2, &gate_up_weight_bytes) ||
        !checked_product(2 * intermediate_size, hidden_size / kGroupSize,
                         &gate_up_scale_count) ||
        !checked_product(hidden_size, intermediate_size / kGroupSize,
                         &down_scale_count) ||
        !checked_product(vocab_size, hidden_size, &vocab_weight_bytes) ||
        !checked_product(vocab_size, hidden_size / kGroupSize,
                         &vocab_scale_count) ||
        !checked_product(vocab_scale_count, sizeof(float),
                         &vocab_scale_bytes) ||
        !checked_sum(vocab_weight_bytes, vocab_scale_bytes,
                     &vocab_init_bytes)) {
        write_error(error_output, error_capacity,
                    "Metal W8 tail MLP+head v1 buffer dimensions overflow");
        return 1;
    }
    if (!all_finite(descriptor->post_attention_rms_weight, hidden_size) ||
        !all_finite(descriptor->final_rms_weight, hidden_size) ||
        !all_positive_finite(descriptor->gate_up_scales,
                             gate_up_scale_count) ||
        !all_positive_finite(descriptor->down_scales, down_scale_count) ||
        !all_positive_finite(descriptor->vocab_scales, vocab_scale_count)) {
        write_error(error_output, error_capacity,
                    "Metal W8 tail MLP+head v1 parameters are non-finite or invalid");
        return 1;
    }

    auto handle = new (std::nothrow) ApxinfMetalW8TailMlpHeadHandleV1{};
    if (handle == nullptr) {
        write_error(error_output, error_capacity,
                    "allocate Metal W8 tail MLP+head v1 handle failed");
        return 1;
    }
    const auto fail = [&](const char *message, NSError *failure_error) -> int {
        delete handle;
        if (message != nullptr) {
            write_error(error_output, error_capacity, message);
        } else {
            write_nserror(error_output, error_capacity, failure_error);
        }
        return 1;
    };

    handle->device = MTLCreateSystemDefaultDevice();
    if (handle->device == nil) {
        return fail("no system Metal device is available", nil);
    }
    NSError *error = nil;
    NSString *source =
        [NSString stringWithUTF8String:kMetalTailMlpHeadSourceV1];
    id<MTLLibrary> library =
        [handle->device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        return fail(nullptr, error);
    }
    handle->rms_pipeline = make_pipeline(
        handle->device, library, @"linear_layer_rms_norm", &error);
    handle->gate_up_pipeline = make_pipeline(
        handle->device, library, @"w8_mlp_gate_up", &error);
    handle->activation_pipeline = make_pipeline(
        handle->device, library, @"w8_mlp_silu_mul", &error);
    handle->down_pipeline = make_pipeline(handle->device, library,
                                          @"w8_mlp_down", &error);
    handle->residual_pipeline = make_pipeline(
        handle->device, library, @"linear_layer_residual_add", &error);
    handle->rows_topk_pipeline = make_pipeline(
        handle->device, library, @"w8_rows_topk4", &error);
    handle->final_topk_pipeline = make_pipeline(
        handle->device, library, @"w8_final_topk4", &error);
    if (handle->rms_pipeline == nil || handle->gate_up_pipeline == nil ||
        handle->activation_pipeline == nil || handle->down_pipeline == nil ||
        handle->residual_pipeline == nil || handle->rows_topk_pipeline == nil ||
        handle->final_topk_pipeline == nil) {
        return fail(nullptr, error);
    }

    handle->queue = [handle->device newCommandQueue];
    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    const MTLResourceOptions private_storage = MTLResourceStorageModePrivate;
    const size_t hidden_bytes = hidden_size * sizeof(float);
    handle->rms_weights =
        [handle->device newBufferWithLength:2 * hidden_bytes options:shared];
    if (handle->rms_weights != nil) {
        std::memcpy(handle->rms_weights.contents,
                    descriptor->post_attention_rms_weight, hidden_bytes);
        std::memcpy(static_cast<uint8_t *>(handle->rms_weights.contents) +
                        hidden_bytes,
                    descriptor->final_rms_weight, hidden_bytes);
    }
    handle->gate_up_weights =
        [handle->device newBufferWithBytes:descriptor->gate_up_weights
                                    length:gate_up_weight_bytes
                                   options:shared];
    handle->gate_up_scales =
        [handle->device newBufferWithBytes:descriptor->gate_up_scales
                                    length:gate_up_scale_count * sizeof(float)
                                   options:shared];
    handle->down_weights =
        [handle->device newBufferWithBytes:descriptor->down_weights
                                    length:hidden_intermediate
                                   options:shared];
    handle->down_scales =
        [handle->device newBufferWithBytes:descriptor->down_scales
                                    length:down_scale_count * sizeof(float)
                                   options:shared];

    id<MTLBuffer> vocab_weights_staging = nil;
    id<MTLBuffer> vocab_scales_staging = nil;
    if (vocab_storage == 0) {
        handle->vocab_weights =
            [handle->device newBufferWithBytes:descriptor->vocab_weights
                                        length:vocab_weight_bytes
                                       options:shared];
        handle->vocab_scales =
            [handle->device newBufferWithBytes:descriptor->vocab_scales
                                        length:vocab_scale_bytes
                                       options:shared];
    } else {
        handle->vocab_weights =
            [handle->device newBufferWithLength:vocab_weight_bytes
                                         options:private_storage];
        handle->vocab_scales =
            [handle->device newBufferWithLength:vocab_scale_bytes
                                         options:private_storage];
        vocab_weights_staging =
            [handle->device newBufferWithBytes:descriptor->vocab_weights
                                        length:vocab_weight_bytes
                                       options:shared];
        vocab_scales_staging =
            [handle->device newBufferWithBytes:descriptor->vocab_scales
                                        length:vocab_scale_bytes
                                       options:shared];
    }

    handle->hidden_a =
        [handle->device newBufferWithLength:hidden_bytes options:shared];
    handle->hidden_b =
        [handle->device newBufferWithLength:hidden_bytes options:shared];
    handle->output_tokens =
        [handle->device newBufferWithLength:kTopK * sizeof(uint32_t)
                                     options:shared];
    handle->gate_up = [handle->device
        newBufferWithLength:2 * intermediate_size * sizeof(float)
                     options:private_storage];
    handle->activated = [handle->device
        newBufferWithLength:intermediate_size * sizeof(float)
                     options:private_storage];
    const uint32_t partial_count =
        static_cast<uint32_t>((vocab_size + kRowsPerThreadgroup - 1) /
                              kRowsPerThreadgroup);
    handle->partial_topk = [handle->device
        newBufferWithLength:static_cast<size_t>(partial_count) * kTopK * 8
                     options:private_storage];
    if (handle->queue == nil || !buffers_valid(handle)) {
        return fail(
            "allocate persistent Metal W8 tail MLP+head v1 buffers failed",
            nil);
    }

    const uint32_t vocab_weights_storage =
        normalized_vocab_storage(handle->vocab_weights);
    const uint32_t vocab_scales_storage =
        normalized_vocab_storage(handle->vocab_scales);
    if (vocab_weights_storage != vocab_storage ||
        vocab_scales_storage != vocab_storage) {
        return fail(
            "Metal W8 tail MLP+head v1 vocab destination storage mode mismatch",
            nil);
    }
    handle->vocab_storage = vocab_weights_storage;

    if (vocab_storage == 1) {
        if (vocab_weights_staging == nil || vocab_scales_staging == nil) {
            return fail(
                "allocate transient Metal W8 tail MLP+head v1 vocab staging buffers failed",
                nil);
        }
        if (normalized_vocab_storage(vocab_weights_staging) != 0 ||
            normalized_vocab_storage(vocab_scales_staging) != 0) {
            return fail(
                "Metal W8 tail MLP+head v1 vocab staging storage mode mismatch",
                nil);
        }
        if (vocab_weights_staging.length != vocab_weight_bytes ||
            vocab_scales_staging.length != vocab_scale_bytes) {
            return fail(
                "Metal W8 tail MLP+head v1 vocab staging length mismatch",
                nil);
        }

        id<MTLCommandBuffer> init_command = [handle->queue commandBuffer];
        if (init_command == nil) {
            return fail(
                "create Metal W8 tail MLP+head v1 vocab init command buffer failed",
                nil);
        }
        id<MTLBlitCommandEncoder> init_blit =
            [init_command blitCommandEncoder];
        if (init_blit == nil) {
            return fail(
                "create Metal W8 tail MLP+head v1 vocab init blit encoder failed",
                nil);
        }
        [init_blit copyFromBuffer:vocab_weights_staging
                     sourceOffset:0
                         toBuffer:handle->vocab_weights
                destinationOffset:0
                             size:vocab_weight_bytes];
        [init_blit copyFromBuffer:vocab_scales_staging
                     sourceOffset:0
                         toBuffer:handle->vocab_scales
                destinationOffset:0
                             size:vocab_scale_bytes];
        [init_blit endEncoding];
        [init_command commit];
        [init_command waitUntilCompleted];
        if (init_command.status != MTLCommandBufferStatusCompleted) {
            NSError *init_error = init_command.error;
            if (init_error != nil) {
                return fail(nullptr, init_error);
            }
            return fail(
                "Metal W8 tail MLP+head v1 vocab init command did not complete",
                nil);
        }

        handle->transient_staging_buffers = 2;
        handle->transient_staging_bytes =
            static_cast<uint64_t>(vocab_init_bytes);
        handle->init_blit_bytes = static_cast<uint64_t>(vocab_init_bytes);
        handle->init_command_buffers = 1;
        handle->init_blit_encoders = 1;
        handle->init_copy_commands = 2;
        handle->init_commits = 1;
        handle->init_waits = 1;
    }

    handle->layer_params = LinearLayerParams{
        descriptor->hidden_size, descriptor->rms_norm_eps};
    handle->mlp_params = MlpParams{
        descriptor->hidden_size,
        descriptor->intermediate_size,
        descriptor->hidden_size / kGroupSize,
        descriptor->intermediate_size / kGroupSize,
    };
    handle->head_params = KernelParams{
        descriptor->hidden_size,
        descriptor->vocab_size,
        descriptor->hidden_size / kGroupSize,
        partial_count,
    };
    handle->terminal_error = false;
    *output = handle;
    return 0;
}

}  // namespace

extern "C" int apxinf_metal_w8_tail_mlp_head_create_v1(
    const TailDescriptorV1 *descriptor, void **output, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        return create_tail_mlp_head_v1(descriptor, output, error_output,
                                       error_capacity, 0);
    }
}

extern "C" int apxinf_metal_w8_tail_mlp_head_create_with_vocab_storage_v1(
    const TailDescriptorV1 *descriptor, uint32_t vocab_storage, void **output,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        return create_tail_mlp_head_v1(descriptor, output, error_output,
                                       error_capacity, vocab_storage);
    }
}

extern "C" int apxinf_metal_w8_tail_mlp_head_vocab_storage_receipt_v1(
    void *opaque_handle, TailVocabStorageReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 vocab storage receipt is null");
            return 1;
        }
        *receipt = TailVocabStorageReceiptV1{};
        auto handle =
            static_cast<ApxinfMetalW8TailMlpHeadHandleV1 *>(opaque_handle);
        if (handle == nullptr || handle->vocab_weights == nil ||
            handle->vocab_scales == nil) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 vocab storage receipt handle is invalid");
            return 1;
        }

        receipt->vocab_weights_storage =
            normalized_vocab_storage(handle->vocab_weights);
        receipt->vocab_scales_storage =
            normalized_vocab_storage(handle->vocab_scales);
        receipt->vocab_storage =
            receipt->vocab_weights_storage == receipt->vocab_scales_storage &&
                    receipt->vocab_weights_storage == handle->vocab_storage
                ? handle->vocab_storage
                : UINT32_MAX;
        receipt->transient_staging_buffers =
            handle->transient_staging_buffers;
        receipt->vocab_weight_bytes =
            static_cast<uint64_t>(handle->vocab_weights.length);
        receipt->vocab_scale_bytes =
            static_cast<uint64_t>(handle->vocab_scales.length);
        receipt->transient_staging_bytes = handle->transient_staging_bytes;
        receipt->init_blit_bytes = handle->init_blit_bytes;
        receipt->init_command_buffers = handle->init_command_buffers;
        receipt->init_blit_encoders = handle->init_blit_encoders;
        receipt->init_copy_commands = handle->init_copy_commands;
        receipt->init_commits = handle->init_commits;
        receipt->init_waits = handle->init_waits;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_tail_mlp_head_decode_v1(
    void *opaque_handle, const float *input, uint32_t input_count,
    float *normalized_hidden, uint32_t normalized_hidden_count,
    uint32_t *candidate_token_ids, uint32_t candidate_count,
    uint32_t fault_mode,
    TailExecutionReceiptV1 *receipt, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        if (receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 execution receipt is null");
            return 1;
        }
        *receipt = TailExecutionReceiptV1{};
        auto handle =
            static_cast<ApxinfMetalW8TailMlpHeadHandleV1 *>(opaque_handle);
        if (handle == nullptr || input == nullptr || normalized_hidden == nullptr ||
            candidate_token_ids == nullptr ||
            (handle != nullptr && input_count != handle->layer_params.hidden_size) ||
            normalized_hidden_count != input_count || candidate_count != kTopK) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 tail MLP+head v1 decode input or output");
            return 1;
        }
        if (handle->terminal_error) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 is terminal until reset");
            return 1;
        }
        if (!all_finite(input, input_count)) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 input is non-finite");
            return 1;
        }
        const size_t hidden_bytes =
            static_cast<size_t>(input_count) * sizeof(float);
        std::memcpy(handle->hidden_a.contents, input, hidden_bytes);
        receipt->host_to_device_bytes = hidden_bytes;

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "create Metal W8 tail MLP+head v1 command buffer failed");
            return 1;
        }
        receipt->command_buffers = 1;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "create Metal W8 tail MLP+head v1 compute encoder failed");
            return 1;
        }
        receipt->compute_encoders = 1;
        encode_tail(handle, encoder);
        receipt->kernel_dispatches = 8;
        receipt->buffer_barriers = 7;
        [encoder endEncoding];
        [command commit];
        receipt->commits = 1;
        [command waitUntilCompleted];
        receipt->waits = 1;
        if (command.status != MTLCommandBufferStatusCompleted) {
            handle->terminal_error = true;
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (fault_mode == 1) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "injected Metal W8 tail MLP+head v1 failure after GPU execution");
            return 1;
        }
        if (fault_mode == 2) {
            static_cast<float *>(handle->hidden_a.contents)[0] = NAN;
        } else if (fault_mode == 3) {
            uint32_t *tokens =
                static_cast<uint32_t *>(handle->output_tokens.contents);
            tokens[1] = tokens[0];
        } else if (fault_mode == 4) {
            static_cast<uint32_t *>(handle->output_tokens.contents)[0] =
                handle->head_params.rows;
        }
        const float *published_hidden =
            static_cast<const float *>(handle->hidden_a.contents);
        const uint32_t *published_tokens =
            static_cast<const uint32_t *>(handle->output_tokens.contents);
        if (!all_finite(published_hidden, input_count) ||
            !valid_candidates(published_tokens, handle->head_params.rows)) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 GPU output failed validation");
            return 1;
        }
        std::memcpy(normalized_hidden, published_hidden, hidden_bytes);
        std::memcpy(candidate_token_ids, published_tokens,
                    kTopK * sizeof(uint32_t));
        receipt->device_to_host_bytes = hidden_bytes + kTopK * sizeof(uint32_t);
        receipt->output_commits = 2;
        receipt->output_commit_mask = kAllOutputsMask;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_tail_mlp_head_reset_v1(
    void *opaque_handle, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8TailMlpHeadHandleV1 *>(opaque_handle);
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 tail MLP+head v1 reset handle is null");
            return 1;
        }
        const size_t hidden_bytes =
            static_cast<size_t>(handle->layer_params.hidden_size) * sizeof(float);
        std::memset(handle->hidden_a.contents, 0, hidden_bytes);
        std::memset(handle->hidden_b.contents, 0, hidden_bytes);
        std::memset(handle->output_tokens.contents, 0xff,
                    kTopK * sizeof(uint32_t));
        handle->terminal_error = false;
        return 0;
    }
}

extern "C" void apxinf_metal_w8_tail_mlp_head_destroy_v1(
    void *opaque_handle) {
    auto handle =
        static_cast<ApxinfMetalW8TailMlpHeadHandleV1 *>(opaque_handle);
    delete handle;
}
