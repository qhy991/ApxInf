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

struct MlpParams {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t gate_up_groups_per_row;
    uint32_t down_groups_per_row;
};

struct LinearLayerParams {
    uint32_t hidden_size;
    float rms_norm_eps;
};

struct LinearLayerExecutionReceipt {
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t commits;
    uint32_t waits;
    uint32_t state_commits;
    uint32_t reserved;
};

static_assert(sizeof(GdnParams) == 52, "Metal W8 GDN parameter ABI changed");
static_assert(sizeof(MlpParams) == 16, "Metal W8 MLP parameter ABI changed");
static_assert(sizeof(LinearLayerParams) == 8,
              "Metal W8 linear-layer parameter ABI changed");
static_assert(sizeof(LinearLayerExecutionReceipt) == 40,
              "Metal W8 linear-layer receipt ABI changed");

struct ApxinfMetalW8LinearLayerHandle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    id<MTLComputePipelineState> layer_rms_pipeline;
    id<MTLComputePipelineState> residual_pipeline;
    id<MTLComputePipelineState> gdn_input_pipeline;
    id<MTLComputePipelineState> gdn_depthwise_pipeline;
    id<MTLComputePipelineState> gdn_normalize_pipeline;
    id<MTLComputePipelineState> gdn_recurrent_pipeline;
    id<MTLComputePipelineState> gdn_norm_gate_pipeline;
    id<MTLComputePipelineState> gdn_output_pipeline;
    id<MTLComputePipelineState> mlp_gate_up_pipeline;
    id<MTLComputePipelineState> mlp_activation_pipeline;
    id<MTLComputePipelineState> mlp_down_pipeline;

    // 14 resident packed-weight/parameter buffers, all shared and uploaded once.
    id<MTLBuffer> gdn_input_weights;
    id<MTLBuffer> gdn_input_scales;
    id<MTLBuffer> gdn_output_weights;
    id<MTLBuffer> gdn_output_scales;
    id<MTLBuffer> conv_weight;
    id<MTLBuffer> a_log;
    id<MTLBuffer> dt_bias;
    id<MTLBuffer> gdn_norm_weight;
    id<MTLBuffer> mlp_gate_up_weights;
    id<MTLBuffer> mlp_gate_up_scales;
    id<MTLBuffer> mlp_down_weights;
    id<MTLBuffer> mlp_down_scales;
    id<MTLBuffer> input_rms_weight;
    id<MTLBuffer> post_attention_rms_weight;

    // Two shared host-visible H rows and eight private reusable activations.
    id<MTLBuffer> input;
    id<MTLBuffer> output;
    id<MTLBuffer> normalized;
    id<MTLBuffer> projected;
    id<MTLBuffer> processed;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;
    id<MTLBuffer> branch_output;
    id<MTLBuffer> mlp_gate_up;
    id<MTLBuffer> mlp_activated;

    // Four active and four scratch state buffers. Only pointer swaps commit.
    id<MTLBuffer> query_state;
    id<MTLBuffer> key_state;
    id<MTLBuffer> value_state;
    id<MTLBuffer> recurrent_state;
    id<MTLBuffer> query_scratch;
    id<MTLBuffer> key_scratch;
    id<MTLBuffer> value_scratch;
    id<MTLBuffer> recurrent_scratch;

    GdnParams gdn_params;
    MlpParams mlp_params;
    LinearLayerParams layer_params;
};

#include "metal_w8_linear_layer_source.inc"

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
    if (output == nullptr ||
        (right != 0 && left > std::numeric_limits<size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name, NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function == nil ? nil
                           : [device newComputePipelineStateWithFunction:function error:error];
}

id<MTLBuffer> make_shared_f32(id<MTLDevice> device, size_t count) {
    size_t bytes = 0;
    if (!checked_product(count, sizeof(float), &bytes)) {
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (buffer != nil) {
        std::memset(buffer.contents, 0, bytes);
    }
    return buffer;
}

id<MTLBuffer> make_private_f32(id<MTLDevice> device, size_t count) {
    size_t bytes = 0;
    if (!checked_product(count, sizeof(float), &bytes)) {
        return nil;
    }
    return [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

void buffer_barrier(id<MTLComputeCommandEncoder> encoder) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

bool valid_state_counts(ApxinfMetalW8LinearLayerHandle *handle,
                        uint32_t query_count, uint32_t key_count,
                        uint32_t value_count, uint32_t recurrent_count) {
    if (handle == nullptr) {
        return false;
    }
    const size_t expected_query =
        static_cast<size_t>(handle->gdn_params.key_width) *
        handle->gdn_params.conv_kernel_size;
    const size_t expected_value =
        static_cast<size_t>(handle->gdn_params.value_width) *
        handle->gdn_params.conv_kernel_size;
    const size_t expected_recurrent =
        static_cast<size_t>(handle->gdn_params.value_heads) *
        handle->gdn_params.key_dim * handle->gdn_params.value_dim;
    return query_count == expected_query && key_count == expected_query &&
           value_count == expected_value && recurrent_count == expected_recurrent;
}

}  // namespace

static int create_linear_layer_impl(
    const int8_t *gdn_input_weights, const float *gdn_input_scales,
    const int8_t *gdn_output_weights, const float *gdn_output_scales,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *gdn_norm_weight, const int8_t *mlp_gate_up_weights,
    const float *mlp_gate_up_scales, const int8_t *mlp_down_weights,
    const float *mlp_down_scales, const float *input_rms_weight,
    const float *post_attention_rms_weight, uint32_t hidden_size,
    uint32_t key_heads, uint32_t value_heads, uint32_t key_dim,
    uint32_t value_dim, uint32_t conv_kernel_size, float gdn_rms_norm_eps,
    uint32_t intermediate_size, float layer_rms_norm_eps, uint32_t group_size,
    uint32_t gdn_output_group_size, NSString *gdn_output_kernel,
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 linear-layer output handle is null");
            return 1;
        }
        *output = nullptr;
        if (gdn_input_weights == nullptr || gdn_input_scales == nullptr ||
            gdn_output_weights == nullptr || gdn_output_scales == nullptr ||
            conv_weight == nullptr || a_log == nullptr || dt_bias == nullptr ||
            gdn_norm_weight == nullptr || mlp_gate_up_weights == nullptr ||
            mlp_gate_up_scales == nullptr || mlp_down_weights == nullptr ||
            mlp_down_scales == nullptr || input_rms_weight == nullptr ||
            post_attention_rms_weight == nullptr || hidden_size == 0 ||
            key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 ||
            conv_kernel_size == 0 || intermediate_size == 0 ||
            value_heads % key_heads != 0 || group_size != 64 ||
            (gdn_output_group_size != 64 && gdn_output_group_size != 32) ||
            gdn_output_kernel == nil ||
            hidden_size % group_size != 0 || intermediate_size % group_size != 0 ||
            hidden_size % 4 != 0 || intermediate_size % 4 != 0 ||
            intermediate_size > UINT32_MAX / 2) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 linear-layer packed contract");
            return 1;
        }
        if (!std::isfinite(gdn_rms_norm_eps) || gdn_rms_norm_eps < 0.0f ||
            !std::isfinite(layer_rms_norm_eps) || layer_rms_norm_eps < 0.0f) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 linear-layer RMS epsilon");
            return 1;
        }

        const uint64_t key_width64 = static_cast<uint64_t>(key_heads) * key_dim;
        const uint64_t value_width64 = static_cast<uint64_t>(value_heads) * value_dim;
        const uint64_t qkv_width64 = 2 * key_width64 + value_width64;
        const uint64_t input_rows64 = qkv_width64 + value_width64 + 2 * value_heads;
        if (key_width64 > UINT32_MAX || value_width64 > UINT32_MAX ||
            qkv_width64 > UINT32_MAX || input_rows64 > UINT32_MAX ||
            value_width64 % gdn_output_group_size != 0 || value_width64 % 4 != 0) {
            write_error(error_output, error_capacity,
                        "Metal W8 linear-layer dimensions exceed ABI");
            return 1;
        }
        const uint32_t key_width = static_cast<uint32_t>(key_width64);
        const uint32_t value_width = static_cast<uint32_t>(value_width64);
        const uint32_t qkv_width = static_cast<uint32_t>(qkv_width64);
        const uint32_t input_rows = static_cast<uint32_t>(input_rows64);
        const size_t gate_up_rows = static_cast<size_t>(intermediate_size) * 2;

        size_t gdn_input_weight_bytes = 0;
        size_t gdn_input_scale_count = 0;
        size_t gdn_output_weight_bytes = 0;
        size_t gdn_output_scale_count = 0;
        size_t conv_count = 0;
        size_t mlp_gate_up_weight_bytes = 0;
        size_t mlp_gate_up_scale_count = 0;
        size_t mlp_down_weight_bytes = 0;
        size_t mlp_down_scale_count = 0;
        size_t query_state_count = 0;
        size_t value_state_count = 0;
        size_t recurrent_state_count = 0;
        if (!checked_product(input_rows, hidden_size, &gdn_input_weight_bytes) ||
            !checked_product(input_rows, hidden_size / group_size,
                             &gdn_input_scale_count) ||
            !checked_product(hidden_size, value_width, &gdn_output_weight_bytes) ||
            !checked_product(hidden_size, value_width / gdn_output_group_size,
                             &gdn_output_scale_count) ||
            !checked_product(qkv_width, conv_kernel_size, &conv_count) ||
            !checked_product(gate_up_rows, hidden_size, &mlp_gate_up_weight_bytes) ||
            !checked_product(gate_up_rows, hidden_size / group_size,
                             &mlp_gate_up_scale_count) ||
            !checked_product(hidden_size, intermediate_size, &mlp_down_weight_bytes) ||
            !checked_product(hidden_size, intermediate_size / group_size,
                             &mlp_down_scale_count) ||
            !checked_product(key_width, conv_kernel_size, &query_state_count) ||
            !checked_product(value_width, conv_kernel_size, &value_state_count) ||
            !checked_product(value_heads, key_dim, &recurrent_state_count) ||
            !checked_product(recurrent_state_count, value_dim,
                             &recurrent_state_count)) {
            write_error(error_output, error_capacity,
                        "Metal W8 linear-layer buffer dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8LinearLayerHandle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal W8 linear-layer handle failed");
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
        NSString *source = [NSString stringWithUTF8String:kMetalLinearLayerSource];
        id<MTLLibrary> library =
            [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }

        handle->layer_rms_pipeline =
            make_pipeline(handle->device, library, @"linear_layer_rms_norm", &error);
        handle->residual_pipeline =
            make_pipeline(handle->device, library, @"linear_layer_residual_add", &error);
        handle->gdn_input_pipeline =
            make_pipeline(handle->device, library, @"gdn_w8_input_projection", &error);
        handle->gdn_depthwise_pipeline =
            make_pipeline(handle->device, library, @"gdn_depthwise_preprocess", &error);
        handle->gdn_normalize_pipeline =
            make_pipeline(handle->device, library, @"gdn_normalize_qk", &error);
        handle->gdn_recurrent_pipeline =
            make_pipeline(handle->device, library, @"gdn_recurrent_update", &error);
        handle->gdn_norm_gate_pipeline =
            make_pipeline(handle->device, library, @"gdn_norm_gate", &error);
        handle->gdn_output_pipeline =
            make_pipeline(handle->device, library, gdn_output_kernel, &error);
        handle->mlp_gate_up_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_gate_up", &error);
        handle->mlp_activation_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_silu_mul", &error);
        handle->mlp_down_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_down", &error);
        if (handle->layer_rms_pipeline == nil || handle->residual_pipeline == nil ||
            handle->gdn_input_pipeline == nil || handle->gdn_depthwise_pipeline == nil ||
            handle->gdn_normalize_pipeline == nil ||
            handle->gdn_recurrent_pipeline == nil ||
            handle->gdn_norm_gate_pipeline == nil || handle->gdn_output_pipeline == nil ||
            handle->mlp_gate_up_pipeline == nil ||
            handle->mlp_activation_pipeline == nil || handle->mlp_down_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        handle->gdn_input_weights = [handle->device newBufferWithBytes:gdn_input_weights
                                                               length:gdn_input_weight_bytes
                                                              options:shared];
        handle->gdn_input_scales = [handle->device newBufferWithBytes:gdn_input_scales
                                                              length:gdn_input_scale_count * sizeof(float)
                                                             options:shared];
        handle->gdn_output_weights = [handle->device newBufferWithBytes:gdn_output_weights
                                                                length:gdn_output_weight_bytes
                                                               options:shared];
        handle->gdn_output_scales = [handle->device newBufferWithBytes:gdn_output_scales
                                                               length:gdn_output_scale_count * sizeof(float)
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
        handle->gdn_norm_weight = [handle->device newBufferWithBytes:gdn_norm_weight
                                                            length:value_dim * sizeof(float)
                                                           options:shared];
        handle->mlp_gate_up_weights = [handle->device newBufferWithBytes:mlp_gate_up_weights
                                                                 length:mlp_gate_up_weight_bytes
                                                                options:shared];
        handle->mlp_gate_up_scales = [handle->device newBufferWithBytes:mlp_gate_up_scales
                                                                length:mlp_gate_up_scale_count * sizeof(float)
                                                               options:shared];
        handle->mlp_down_weights = [handle->device newBufferWithBytes:mlp_down_weights
                                                              length:mlp_down_weight_bytes
                                                             options:shared];
        handle->mlp_down_scales = [handle->device newBufferWithBytes:mlp_down_scales
                                                             length:mlp_down_scale_count * sizeof(float)
                                                            options:shared];
        handle->input_rms_weight = [handle->device newBufferWithBytes:input_rms_weight
                                                               length:hidden_size * sizeof(float)
                                                              options:shared];
        handle->post_attention_rms_weight =
            [handle->device newBufferWithBytes:post_attention_rms_weight
                                         length:hidden_size * sizeof(float)
                                        options:shared];

        handle->input = make_shared_f32(handle->device, hidden_size);
        handle->output = make_shared_f32(handle->device, hidden_size);
        handle->normalized = make_private_f32(handle->device, hidden_size);
        handle->projected = make_private_f32(handle->device, input_rows);
        handle->processed = make_private_f32(handle->device, qkv_width);
        handle->core = make_private_f32(handle->device, value_width);
        handle->gated = make_private_f32(handle->device, value_width);
        handle->branch_output = make_private_f32(handle->device, hidden_size);
        handle->mlp_gate_up = make_private_f32(handle->device, gate_up_rows);
        handle->mlp_activated = make_private_f32(handle->device, intermediate_size);

        handle->query_state = make_shared_f32(handle->device, query_state_count);
        handle->key_state = make_shared_f32(handle->device, query_state_count);
        handle->value_state = make_shared_f32(handle->device, value_state_count);
        handle->recurrent_state = make_shared_f32(handle->device, recurrent_state_count);
        handle->query_scratch = make_shared_f32(handle->device, query_state_count);
        handle->key_scratch = make_shared_f32(handle->device, query_state_count);
        handle->value_scratch = make_shared_f32(handle->device, value_state_count);
        handle->recurrent_scratch = make_shared_f32(handle->device, recurrent_state_count);

        if (handle->queue == nil || handle->gdn_input_weights == nil ||
            handle->gdn_input_scales == nil || handle->gdn_output_weights == nil ||
            handle->gdn_output_scales == nil || handle->conv_weight == nil ||
            handle->a_log == nil || handle->dt_bias == nil ||
            handle->gdn_norm_weight == nil || handle->mlp_gate_up_weights == nil ||
            handle->mlp_gate_up_scales == nil || handle->mlp_down_weights == nil ||
            handle->mlp_down_scales == nil || handle->input_rms_weight == nil ||
            handle->post_attention_rms_weight == nil || handle->input == nil ||
            handle->output == nil || handle->normalized == nil ||
            handle->projected == nil || handle->processed == nil || handle->core == nil ||
            handle->gated == nil || handle->branch_output == nil ||
            handle->mlp_gate_up == nil || handle->mlp_activated == nil ||
            handle->query_state == nil || handle->key_state == nil ||
            handle->value_state == nil || handle->recurrent_state == nil ||
            handle->query_scratch == nil || handle->key_scratch == nil ||
            handle->value_scratch == nil || handle->recurrent_scratch == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal W8 linear-layer buffers failed");
            return 1;
        }

        handle->gdn_params = GdnParams{
            hidden_size,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel_size,
            key_width,
            value_width,
            qkv_width,
            input_rows,
            hidden_size / group_size,
            value_width / gdn_output_group_size,
            gdn_rms_norm_eps,
        };
        handle->mlp_params = MlpParams{
            hidden_size,
            intermediate_size,
            hidden_size / group_size,
            intermediate_size / group_size,
        };
        handle->layer_params = LinearLayerParams{hidden_size, layer_rms_norm_eps};
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_linear_layer_create(
    const int8_t *gdn_input_weights, const float *gdn_input_scales,
    const int8_t *gdn_output_weights, const float *gdn_output_scales,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *gdn_norm_weight, const int8_t *mlp_gate_up_weights,
    const float *mlp_gate_up_scales, const int8_t *mlp_down_weights,
    const float *mlp_down_scales, const float *input_rms_weight,
    const float *post_attention_rms_weight, uint32_t hidden_size,
    uint32_t key_heads, uint32_t value_heads, uint32_t key_dim,
    uint32_t value_dim, uint32_t conv_kernel_size, float gdn_rms_norm_eps,
    uint32_t intermediate_size, float layer_rms_norm_eps, uint32_t group_size,
    void **output, char *error_output, size_t error_capacity) {
    return create_linear_layer_impl(
        gdn_input_weights, gdn_input_scales, gdn_output_weights,
        gdn_output_scales, conv_weight, a_log, dt_bias, gdn_norm_weight,
        mlp_gate_up_weights, mlp_gate_up_scales, mlp_down_weights,
        mlp_down_scales, input_rms_weight, post_attention_rms_weight,
        hidden_size, key_heads, value_heads, key_dim, value_dim,
        conv_kernel_size, gdn_rms_norm_eps, intermediate_size,
        layer_rms_norm_eps, group_size, group_size,
        @"gdn_w8_output_projection", output, error_output, error_capacity);
}

extern "C" int apxinf_metal_w8_linear_layer_gdn_out_g32_create(
    const int8_t *gdn_input_weights, const float *gdn_input_scales,
    const int8_t *gdn_output_weights, const float *gdn_output_scales,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *gdn_norm_weight, const int8_t *mlp_gate_up_weights,
    const float *mlp_gate_up_scales, const int8_t *mlp_down_weights,
    const float *mlp_down_scales, const float *input_rms_weight,
    const float *post_attention_rms_weight, uint32_t hidden_size,
    uint32_t key_heads, uint32_t value_heads, uint32_t key_dim,
    uint32_t value_dim, uint32_t conv_kernel_size, float gdn_rms_norm_eps,
    uint32_t intermediate_size, float layer_rms_norm_eps,
    void **output, char *error_output, size_t error_capacity) {
    return create_linear_layer_impl(
        gdn_input_weights, gdn_input_scales, gdn_output_weights,
        gdn_output_scales, conv_weight, a_log, dt_bias, gdn_norm_weight,
        mlp_gate_up_weights, mlp_gate_up_scales, mlp_down_weights,
        mlp_down_scales, input_rms_weight, post_attention_rms_weight,
        hidden_size, key_heads, value_heads, key_dim, value_dim,
        conv_kernel_size, gdn_rms_norm_eps, intermediate_size,
        layer_rms_norm_eps, 64, 32, @"gdn_w8_output_projection_g32",
        output, error_output, error_capacity);
}

extern "C" int apxinf_metal_w8_linear_layer_seed_state(
    void *opaque_handle, const float *query_state, uint32_t query_count,
    const float *key_state, uint32_t key_count, const float *value_state,
    uint32_t value_count, const float *recurrent_state, uint32_t recurrent_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8LinearLayerHandle *>(opaque_handle);
        if (query_state == nullptr || key_state == nullptr || value_state == nullptr ||
            recurrent_state == nullptr ||
            !valid_state_counts(handle, query_count, key_count, value_count,
                                recurrent_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 linear-layer seed state");
            return 1;
        }
        const size_t query_bytes = static_cast<size_t>(query_count) * sizeof(float);
        const size_t value_bytes = static_cast<size_t>(value_count) * sizeof(float);
        const size_t recurrent_bytes = static_cast<size_t>(recurrent_count) * sizeof(float);
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

extern "C" int apxinf_metal_w8_linear_layer_decode(
    void *opaque_handle, const float *input, uint32_t input_count, float *output,
    uint32_t output_count, uint8_t inject_failure_after_execution,
    LinearLayerExecutionReceipt *receipt, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 linear-layer execution receipt is null");
            return 1;
        }
        *receipt = LinearLayerExecutionReceipt{};
        auto handle = static_cast<ApxinfMetalW8LinearLayerHandle *>(opaque_handle);
        if (handle == nullptr || input == nullptr || output == nullptr ||
            input_count != handle->gdn_params.hidden_size ||
            output_count != handle->gdn_params.hidden_size) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 linear-layer input or output");
            return 1;
        }

        const size_t hidden_bytes = static_cast<size_t>(input_count) * sizeof(float);
        std::memcpy(handle->input.contents, input, hidden_bytes);
        receipt->host_to_device_bytes = hidden_bytes;

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal W8 linear-layer command buffer failed");
            return 1;
        }
        receipt->command_buffers = 1;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal W8 linear-layer compute encoder failed");
            return 1;
        }
        receipt->compute_encoders = 1;

        [encoder setComputePipelineState:handle->layer_rms_pipeline];
        [encoder setBuffer:handle->input offset:0 atIndex:0];
        [encoder setBuffer:handle->input_rms_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBytes:&handle->layer_params length:sizeof(handle->layer_params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_input_pipeline];
        [encoder setBuffer:handle->gdn_input_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->gdn_input_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBuffer:handle->projected offset:0 atIndex:3];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (handle->gdn_params.input_rows + kRowsPerThreadgroup - 1) /
                        kRowsPerThreadgroup,
                    1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_depthwise_pipeline];
        [encoder setBuffer:handle->projected offset:0 atIndex:0];
        [encoder setBuffer:handle->conv_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->query_state offset:0 atIndex:2];
        [encoder setBuffer:handle->key_state offset:0 atIndex:3];
        [encoder setBuffer:handle->value_state offset:0 atIndex:4];
        [encoder setBuffer:handle->query_scratch offset:0 atIndex:5];
        [encoder setBuffer:handle->key_scratch offset:0 atIndex:6];
        [encoder setBuffer:handle->value_scratch offset:0 atIndex:7];
        [encoder setBuffer:handle->processed offset:0 atIndex:8];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:9];
        [encoder dispatchThreads:MTLSizeMake(handle->gdn_params.qkv_width, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_normalize_pipeline];
        [encoder setBuffer:handle->processed offset:0 atIndex:0];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(2 * handle->gdn_params.key_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, 2 * handle->gdn_params.key_heads), 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_recurrent_pipeline];
        [encoder setBuffer:handle->processed offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:handle->a_log offset:0 atIndex:2];
        [encoder setBuffer:handle->dt_bias offset:0 atIndex:3];
        [encoder setBuffer:handle->recurrent_state offset:0 atIndex:4];
        [encoder setBuffer:handle->recurrent_scratch offset:0 atIndex:5];
        [encoder setBuffer:handle->core offset:0 atIndex:6];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(handle->gdn_params.value_heads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_norm_gate_pipeline];
        [encoder setBuffer:handle->core offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:handle->gdn_norm_weight offset:0 atIndex:2];
        [encoder setBuffer:handle->gated offset:0 atIndex:3];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(handle->gdn_params.value_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, handle->gdn_params.value_heads), 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->gdn_output_pipeline];
        [encoder setBuffer:handle->gdn_output_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->gdn_output_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->gated offset:0 atIndex:2];
        [encoder setBuffer:handle->branch_output offset:0 atIndex:3];
        [encoder setBytes:&handle->gdn_params length:sizeof(handle->gdn_params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (handle->gdn_params.hidden_size + kRowsPerThreadgroup - 1) /
                        kRowsPerThreadgroup,
                    1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->residual_pipeline];
        [encoder setBuffer:handle->input offset:0 atIndex:0];
        [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
        [encoder setBuffer:handle->output offset:0 atIndex:2];
        [encoder setBytes:&handle->layer_params length:sizeof(handle->layer_params) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(handle->layer_params.hidden_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->layer_rms_pipeline];
        [encoder setBuffer:handle->output offset:0 atIndex:0];
        [encoder setBuffer:handle->post_attention_rms_weight offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBytes:&handle->layer_params length:sizeof(handle->layer_params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->mlp_gate_up_pipeline];
        [encoder setBuffer:handle->mlp_gate_up_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->mlp_gate_up_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->normalized offset:0 atIndex:2];
        [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:3];
        [encoder setBytes:&handle->mlp_params length:sizeof(handle->mlp_params) atIndex:4];
        const uint32_t gate_up_rows = handle->mlp_params.intermediate_size * 2;
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->mlp_activation_pipeline];
        [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:0];
        [encoder setBuffer:handle->mlp_activated offset:0 atIndex:1];
        [encoder setBytes:&handle->mlp_params length:sizeof(handle->mlp_params) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(handle->mlp_params.intermediate_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
        buffer_barrier(encoder);

        [encoder setComputePipelineState:handle->mlp_down_pipeline];
        [encoder setBuffer:handle->mlp_down_weights offset:0 atIndex:0];
        [encoder setBuffer:handle->mlp_down_scales offset:0 atIndex:1];
        [encoder setBuffer:handle->mlp_activated offset:0 atIndex:2];
        [encoder setBuffer:handle->branch_output offset:0 atIndex:3];
        [encoder setBytes:&handle->mlp_params length:sizeof(handle->mlp_params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                    (handle->mlp_params.hidden_size + kRowsPerThreadgroup - 1) /
                        kRowsPerThreadgroup,
                    1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        buffer_barrier(encoder);

        // In-place residual is safe because each invocation owns one H index.
        [encoder setComputePipelineState:handle->residual_pipeline];
        [encoder setBuffer:handle->output offset:0 atIndex:0];
        [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
        [encoder setBuffer:handle->output offset:0 atIndex:2];
        [encoder setBytes:&handle->layer_params length:sizeof(handle->layer_params) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(handle->layer_params.hidden_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];

        [encoder endEncoding];
        [command commit];
        receipt->commits = 1;
        [command waitUntilCompleted];
        receipt->waits = 1;
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (inject_failure_after_execution) {
            write_error(error_output, error_capacity,
                        "injected Metal W8 linear-layer failure after scratch execution");
            return 1;
        }

        std::swap(handle->query_state, handle->query_scratch);
        std::swap(handle->key_state, handle->key_scratch);
        std::swap(handle->value_state, handle->value_scratch);
        std::swap(handle->recurrent_state, handle->recurrent_scratch);
        receipt->state_commits = 1;
        std::memcpy(output, handle->output.contents, hidden_bytes);
        receipt->device_to_host_bytes = hidden_bytes;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_linear_layer_snapshot_state(
    void *opaque_handle, float *query_state, uint32_t query_count,
    float *key_state, uint32_t key_count, float *value_state,
    uint32_t value_count, float *recurrent_state, uint32_t recurrent_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8LinearLayerHandle *>(opaque_handle);
        if (query_state == nullptr || key_state == nullptr || value_state == nullptr ||
            recurrent_state == nullptr ||
            !valid_state_counts(handle, query_count, key_count, value_count,
                                recurrent_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 linear-layer state snapshot");
            return 1;
        }
        std::memcpy(query_state, handle->query_state.contents,
                    static_cast<size_t>(query_count) * sizeof(float));
        std::memcpy(key_state, handle->key_state.contents,
                    static_cast<size_t>(key_count) * sizeof(float));
        std::memcpy(value_state, handle->value_state.contents,
                    static_cast<size_t>(value_count) * sizeof(float));
        std::memcpy(recurrent_state, handle->recurrent_state.contents,
                    static_cast<size_t>(recurrent_count) * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_linear_layer_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8LinearLayerHandle *>(opaque_handle);
    delete handle;
}
