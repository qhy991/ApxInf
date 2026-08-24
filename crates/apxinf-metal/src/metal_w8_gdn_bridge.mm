#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace {

constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kMatVecThreads = kRowsPerThreadgroup * 32;
constexpr uint32_t kElementThreads = 256;

struct GdnParams {
    uint32_t hidden_size;
    uint32_t key_heads;
    uint32_t value_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t conv_kernel_size;
    uint32_t key_width;
    uint32_t value_width;
    uint32_t qkv_width;
    uint32_t input_rows;
    uint32_t input_groups_per_row;
    uint32_t output_groups_per_row;
    float rms_norm_eps;
};

static_assert(sizeof(GdnParams) == 52, "Metal W8 GDN parameter ABI changed");

struct ApxinfMetalW8GdnHandle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> input_pipeline;
    id<MTLComputePipelineState> depthwise_pipeline;
    id<MTLComputePipelineState> normalize_pipeline;
    id<MTLComputePipelineState> recurrent_pipeline;
    id<MTLComputePipelineState> norm_gate_pipeline;
    id<MTLComputePipelineState> output_pipeline;
    id<MTLBuffer> input_weights;
    id<MTLBuffer> input_scales;
    id<MTLBuffer> output_weights;
    id<MTLBuffer> output_scales;
    id<MTLBuffer> conv_weight;
    id<MTLBuffer> a_log;
    id<MTLBuffer> dt_bias;
    id<MTLBuffer> norm_weight;
    id<MTLBuffer> input;
    id<MTLBuffer> projected;
    id<MTLBuffer> processed;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;
    id<MTLBuffer> output;
    id<MTLBuffer> query_state;
    id<MTLBuffer> key_state;
    id<MTLBuffer> value_state;
    id<MTLBuffer> recurrent_state;
    id<MTLBuffer> query_scratch;
    id<MTLBuffer> key_scratch;
    id<MTLBuffer> value_scratch;
    id<MTLBuffer> recurrent_scratch;
    GdnParams params;
};

#include "metal_w8_gdn_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal error" : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String);
}

bool checked_product(size_t left, size_t right, size_t *output) {
    if (output == nullptr || (right != 0 && left > std::numeric_limits<size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name, NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function == nil ? nil : [device newComputePipelineStateWithFunction:function error:error];
}

id<MTLBuffer> make_shared_f32(id<MTLDevice> device, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:count * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (buffer != nil) {
        std::memset(buffer.contents, 0, count * sizeof(float));
    }
    return buffer;
}

}  // namespace

extern "C" int apxinf_metal_w8_gdn_create(
    const int8_t *input_weights, const float *input_scales,
    const int8_t *output_weights, const float *output_scales,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, uint32_t hidden_size, uint32_t key_heads,
    uint32_t value_heads, uint32_t key_dim, uint32_t value_dim,
    uint32_t conv_kernel_size, float rms_norm_eps, uint32_t group_size,
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity, "Metal W8 GDN output handle is null");
            return 1;
        }
        *output = nullptr;
        if (input_weights == nullptr || input_scales == nullptr ||
            output_weights == nullptr || output_scales == nullptr || conv_weight == nullptr ||
            a_log == nullptr || dt_bias == nullptr || norm_weight == nullptr || hidden_size == 0 ||
            key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
            conv_kernel_size == 0 || value_heads % key_heads != 0 || group_size != 64) {
            write_error(error_output, error_capacity, "invalid Metal W8 GDN packed contract");
            return 1;
        }
        if (!std::isfinite(rms_norm_eps) || rms_norm_eps < 0.0f) {
            write_error(error_output, error_capacity, "invalid Metal W8 GDN RMS epsilon");
            return 1;
        }
        const uint64_t key_width64 = static_cast<uint64_t>(key_heads) * key_dim;
        const uint64_t value_width64 = static_cast<uint64_t>(value_heads) * value_dim;
        const uint64_t qkv_width64 = 2 * key_width64 + value_width64;
        const uint64_t input_rows64 = qkv_width64 + value_width64 + 2 * value_heads;
        if (key_width64 > UINT32_MAX || value_width64 > UINT32_MAX ||
            qkv_width64 > UINT32_MAX || input_rows64 > UINT32_MAX ||
            hidden_size % group_size != 0 || value_width64 % group_size != 0) {
            write_error(error_output, error_capacity, "Metal W8 GDN dimensions exceed ABI");
            return 1;
        }
        const uint32_t key_width = static_cast<uint32_t>(key_width64);
        const uint32_t value_width = static_cast<uint32_t>(value_width64);
        const uint32_t qkv_width = static_cast<uint32_t>(qkv_width64);
        const uint32_t input_rows = static_cast<uint32_t>(input_rows64);
        size_t input_weight_bytes = 0;
        size_t input_scale_count = 0;
        size_t output_weight_bytes = 0;
        size_t output_scale_count = 0;
        size_t conv_count = 0;
        size_t query_state_count = 0;
        size_t value_state_count = 0;
        size_t recurrent_count = 0;
        if (!checked_product(input_rows, hidden_size, &input_weight_bytes) ||
            !checked_product(input_rows, hidden_size / group_size, &input_scale_count) ||
            !checked_product(hidden_size, value_width, &output_weight_bytes) ||
            !checked_product(hidden_size, value_width / group_size, &output_scale_count) ||
            !checked_product(qkv_width, conv_kernel_size, &conv_count) ||
            !checked_product(key_width, conv_kernel_size, &query_state_count) ||
            !checked_product(value_width, conv_kernel_size, &value_state_count) ||
            !checked_product(value_heads, key_dim, &recurrent_count) ||
            !checked_product(recurrent_count, value_dim, &recurrent_count)) {
            write_error(error_output, error_capacity, "Metal W8 GDN dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8GdnHandle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity, "allocate Metal W8 GDN handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity, "no system Metal device is available");
            return 1;
        }
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:kMetalGdnSource];
        id<MTLLibrary> library =
            [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->input_pipeline =
            make_pipeline(handle->device, library, @"gdn_w8_input_projection", &error);
        handle->depthwise_pipeline =
            make_pipeline(handle->device, library, @"gdn_depthwise_preprocess", &error);
        handle->normalize_pipeline =
            make_pipeline(handle->device, library, @"gdn_normalize_qk", &error);
        handle->recurrent_pipeline =
            make_pipeline(handle->device, library, @"gdn_recurrent_update", &error);
        handle->norm_gate_pipeline =
            make_pipeline(handle->device, library, @"gdn_norm_gate", &error);
        handle->output_pipeline =
            make_pipeline(handle->device, library, @"gdn_w8_output_projection", &error);
        if (handle->input_pipeline == nil || handle->depthwise_pipeline == nil ||
            handle->normalize_pipeline == nil || handle->recurrent_pipeline == nil ||
            handle->norm_gate_pipeline == nil || handle->output_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        const MTLResourceOptions private_storage = MTLResourceStorageModePrivate;
        handle->input_weights =
            [handle->device newBufferWithBytes:input_weights length:input_weight_bytes options:shared];
        handle->input_scales = [handle->device newBufferWithBytes:input_scales
                                                           length:input_scale_count * sizeof(float)
                                                          options:shared];
        handle->output_weights = [handle->device newBufferWithBytes:output_weights
                                                             length:output_weight_bytes
                                                            options:shared];
        handle->output_scales = [handle->device newBufferWithBytes:output_scales
                                                            length:output_scale_count * sizeof(float)
                                                           options:shared];
        handle->conv_weight = [handle->device newBufferWithBytes:conv_weight
                                                          length:conv_count * sizeof(float)
                                                         options:shared];
        handle->a_log = [handle->device newBufferWithBytes:a_log
                                                    length:value_heads * sizeof(float)
                                                   options:shared];
        handle->dt_bias = [handle->device newBufferWithBytes:dt_bias
                                                      length:value_heads * sizeof(float)
                                                     options:shared];
        handle->norm_weight = [handle->device newBufferWithBytes:norm_weight
                                                          length:value_dim * sizeof(float)
                                                         options:shared];
        handle->input = make_shared_f32(handle->device, hidden_size);
        handle->projected = [handle->device newBufferWithLength:input_rows * sizeof(float)
                                                        options:private_storage];
        handle->processed = [handle->device newBufferWithLength:qkv_width * sizeof(float)
                                                        options:private_storage];
        handle->core = [handle->device newBufferWithLength:value_width * sizeof(float)
                                                   options:private_storage];
        handle->gated = [handle->device newBufferWithLength:value_width * sizeof(float)
                                                    options:private_storage];
        handle->output = make_shared_f32(handle->device, hidden_size);
        handle->query_state = make_shared_f32(handle->device, query_state_count);
        handle->key_state = make_shared_f32(handle->device, query_state_count);
        handle->value_state = make_shared_f32(handle->device, value_state_count);
        handle->recurrent_state = make_shared_f32(handle->device, recurrent_count);
        handle->query_scratch = make_shared_f32(handle->device, query_state_count);
        handle->key_scratch = make_shared_f32(handle->device, query_state_count);
        handle->value_scratch = make_shared_f32(handle->device, value_state_count);
        handle->recurrent_scratch = make_shared_f32(handle->device, recurrent_count);
        if (handle->queue == nil || handle->input_weights == nil || handle->input_scales == nil ||
            handle->output_weights == nil || handle->output_scales == nil ||
            handle->conv_weight == nil || handle->a_log == nil || handle->dt_bias == nil ||
            handle->norm_weight == nil || handle->input == nil || handle->projected == nil ||
            handle->processed == nil || handle->core == nil || handle->gated == nil ||
            handle->output == nil || handle->query_state == nil || handle->key_state == nil ||
            handle->value_state == nil || handle->recurrent_state == nil ||
            handle->query_scratch == nil || handle->key_scratch == nil ||
            handle->value_scratch == nil || handle->recurrent_scratch == nil) {
            delete handle;
            write_error(error_output, error_capacity, "allocate persistent Metal W8 GDN buffers failed");
            return 1;
        }
        handle->params = GdnParams{
            hidden_size, key_heads, value_heads, key_dim, value_dim, conv_kernel_size,
            key_width, value_width, qkv_width, input_rows, hidden_size / group_size,
            value_width / group_size, rms_norm_eps,
        };
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_gdn_seed_state(
    void *opaque_handle, const float *query_state, uint32_t query_count,
    const float *key_state, uint32_t key_count, const float *value_state,
    uint32_t value_count, const float *recurrent_state, uint32_t recurrent_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8GdnHandle *>(opaque_handle);
        const size_t expected_query = static_cast<size_t>(handle == nullptr ? 0 : handle->params.key_width) *
                                      (handle == nullptr ? 0 : handle->params.conv_kernel_size);
        const size_t expected_value = static_cast<size_t>(handle == nullptr ? 0 : handle->params.value_width) *
                                      (handle == nullptr ? 0 : handle->params.conv_kernel_size);
        const size_t expected_recurrent = static_cast<size_t>(handle == nullptr ? 0 : handle->params.value_heads) *
                                          (handle == nullptr ? 0 : handle->params.key_dim) *
                                          (handle == nullptr ? 0 : handle->params.value_dim);
        if (handle == nullptr || query_state == nullptr || key_state == nullptr ||
            value_state == nullptr || recurrent_state == nullptr || query_count != expected_query ||
            key_count != expected_query || value_count != expected_value ||
            recurrent_count != expected_recurrent) {
            write_error(error_output, error_capacity, "invalid Metal W8 GDN seed state");
            return 1;
        }
        const size_t query_bytes = expected_query * sizeof(float);
        const size_t value_bytes = expected_value * sizeof(float);
        const size_t recurrent_bytes = expected_recurrent * sizeof(float);
        std::memcpy(handle->query_state.contents, query_state, query_bytes);
        std::memcpy(handle->query_scratch.contents, query_state, query_bytes);
        std::memcpy(handle->key_state.contents, key_state, query_bytes);
        std::memcpy(handle->key_scratch.contents, key_state, query_bytes);
        std::memcpy(handle->value_state.contents, value_state, value_bytes);
        std::memcpy(handle->value_scratch.contents, value_state, value_bytes);
        std::memcpy(handle->recurrent_state.contents, recurrent_state, recurrent_bytes);
        std::memcpy(handle->recurrent_scratch.contents, recurrent_state, recurrent_bytes);
        return 0;
    }
}

extern "C" int apxinf_metal_w8_gdn_decode(
    void *opaque_handle, const float *input, uint32_t input_count,
    float *output, uint32_t output_count, uint8_t inject_failure_after_execution,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8GdnHandle *>(opaque_handle);
        if (handle == nullptr || input == nullptr || output == nullptr ||
            input_count != handle->params.hidden_size || output_count != handle->params.hidden_size) {
            write_error(error_output, error_capacity, "invalid Metal W8 GDN input or output");
            return 1;
        }
        std::memcpy(handle->input.contents, input, input_count * sizeof(float));
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (command == nil || encoder == nil) {
            write_error(error_output, error_capacity, "create Metal W8 GDN command failed");
            return 1;
        }

        [encoder setComputePipelineState:handle->input_pipeline];
        [encoder setBuffer:handle->input_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->input_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->input offset:0 atIndex:2];
        [encoder setBuffer:handle->projected offset:0 atIndex:3];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (handle->params.input_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];

        [encoder setComputePipelineState:handle->depthwise_pipeline];
        [encoder setBuffer:handle->projected offset:0 atIndex:0];
        [encoder setBuffer:handle->conv_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->query_state offset:0 atIndex:2];
        [encoder setBuffer:handle->key_state offset:0 atIndex:3];
        [encoder setBuffer:handle->value_state offset:0 atIndex:4];
        [encoder setBuffer:handle->query_scratch offset:0 atIndex:5];
        [encoder setBuffer:handle->key_scratch offset:0 atIndex:6];
        [encoder setBuffer:handle->value_scratch offset:0 atIndex:7];
        [encoder setBuffer:handle->processed offset:0 atIndex:8];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:9];
        [encoder dispatchThreads:MTLSizeMake(handle->params.qkv_width, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];

        [encoder setComputePipelineState:handle->normalize_pipeline];
        [encoder setBuffer:handle->processed offset:0 atIndex:0];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(2 * handle->params.key_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, 2 * handle->params.key_heads), 1, 1)];

        [encoder setComputePipelineState:handle->recurrent_pipeline];
        [encoder setBuffer:handle->processed offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:handle->a_log offset:0 atIndex:2];
        [encoder setBuffer:handle->dt_bias offset:0 atIndex:3];
        [encoder setBuffer:handle->recurrent_state offset:0 atIndex:4];
        [encoder setBuffer:handle->recurrent_scratch offset:0 atIndex:5];
        [encoder setBuffer:handle->core offset:0 atIndex:6];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(handle->params.value_heads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];

        [encoder setComputePipelineState:handle->norm_gate_pipeline];
        [encoder setBuffer:handle->core offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:handle->norm_weight offset:0 atIndex:2];
        [encoder setBuffer:handle->gated offset:0 atIndex:3];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(handle->params.value_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, handle->params.value_heads), 1, 1)];

        [encoder setComputePipelineState:handle->output_pipeline];
        [encoder setBuffer:handle->output_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->output_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->gated offset:0 atIndex:2];
        [encoder setBuffer:handle->output offset:0 atIndex:3];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (handle->params.hidden_size + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];

        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (inject_failure_after_execution) {
            write_error(error_output, error_capacity,
                        "injected Metal W8 GDN failure after scratch execution");
            return 1;
        }
        std::swap(handle->query_state, handle->query_scratch);
        std::swap(handle->key_state, handle->key_scratch);
        std::swap(handle->value_state, handle->value_scratch);
        std::swap(handle->recurrent_state, handle->recurrent_scratch);
        std::memcpy(output, handle->output.contents, output_count * sizeof(float));
        return 0;
    }
}

extern "C" int apxinf_metal_w8_gdn_snapshot_state(
    void *opaque_handle, float *query_state, uint32_t query_count,
    float *key_state, uint32_t key_count, float *value_state,
    uint32_t value_count, float *recurrent_state, uint32_t recurrent_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8GdnHandle *>(opaque_handle);
        const size_t expected_query = static_cast<size_t>(handle == nullptr ? 0 : handle->params.key_width) *
                                      (handle == nullptr ? 0 : handle->params.conv_kernel_size);
        const size_t expected_value = static_cast<size_t>(handle == nullptr ? 0 : handle->params.value_width) *
                                      (handle == nullptr ? 0 : handle->params.conv_kernel_size);
        const size_t expected_recurrent = static_cast<size_t>(handle == nullptr ? 0 : handle->params.value_heads) *
                                          (handle == nullptr ? 0 : handle->params.key_dim) *
                                          (handle == nullptr ? 0 : handle->params.value_dim);
        if (handle == nullptr || query_state == nullptr || key_state == nullptr ||
            value_state == nullptr || recurrent_state == nullptr || query_count != expected_query ||
            key_count != expected_query || value_count != expected_value ||
            recurrent_count != expected_recurrent) {
            write_error(error_output, error_capacity, "invalid Metal W8 GDN state snapshot");
            return 1;
        }
        std::memcpy(query_state, handle->query_state.contents, expected_query * sizeof(float));
        std::memcpy(key_state, handle->key_state.contents, expected_query * sizeof(float));
        std::memcpy(value_state, handle->value_state.contents, expected_value * sizeof(float));
        std::memcpy(recurrent_state, handle->recurrent_state.contents,
                    expected_recurrent * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_gdn_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8GdnHandle *>(opaque_handle);
    delete handle;
}
