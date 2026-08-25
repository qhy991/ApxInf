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

constexpr uint32_t kStackDepth = 3;
constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kMatVecThreads = kRowsPerThreadgroup * 32;
constexpr uint32_t kElementThreads = 256;
constexpr uint32_t kAllSeededMask = (1u << kStackDepth) - 1;
constexpr uint32_t kBodyInputStagingLegacyDevice = 0;
constexpr uint32_t kBodyInputStagingThreadgroupShared = 1;
constexpr uint32_t kUnknownBodyInputStaging = UINT32_MAX;
constexpr uint32_t kExpectedThreadExecutionWidth = 32;
constexpr size_t kBodyInputFunctionNameCapacity = 64;

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

struct BoundaryMlpDescriptorV1 {
    const int8_t *gate_up_weights;
    const float *gate_up_scales;
    const int8_t *down_weights;
    const float *down_scales;
    const float *post_attention_rms_weight;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    float rms_norm_eps;
};

struct BoundaryStackLayerDescriptorV1 {
    const int8_t *gdn_input_weights;
    const float *gdn_input_scales;
    const int8_t *gdn_output_weights;
    const float *gdn_output_scales;
    const float *conv_weight;
    const float *a_log;
    const float *dt_bias;
    const float *gdn_norm_weight;
    const int8_t *mlp_gate_up_weights;
    const float *mlp_gate_up_scales;
    const int8_t *mlp_down_weights;
    const float *mlp_down_scales;
    const float *input_rms_weight;
    const float *post_attention_rms_weight;
    uint32_t hidden_size;
    uint32_t key_heads;
    uint32_t value_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t conv_kernel_size;
    float gdn_rms_norm_eps;
    uint32_t intermediate_size;
    float layer_rms_norm_eps;
};

struct BoundaryStateDescriptorV1 {
    const float *query_state;
    uint32_t query_count;
    const float *key_state;
    uint32_t key_count;
    const float *value_state;
    uint32_t value_count;
    const float *recurrent_state;
    uint32_t recurrent_count;
};

struct BoundaryMutableStateDescriptorV1 {
    float *query_state;
    uint32_t query_count;
    float *key_state;
    uint32_t key_count;
    float *value_state;
    uint32_t value_count;
    float *recurrent_state;
    uint32_t recurrent_count;
};

struct BoundaryExecutionReceiptV1 {
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t commits;
    uint32_t waits;
    uint32_t state_commits;
    uint32_t state_commit_mask;
};

struct BodyInputStagingRuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t threads_per_threadgroup;
    uint32_t gdn_input_dynamic_threadgroup_memory_bytes;
    uint32_t gdn_output_dynamic_threadgroup_memory_bytes;
    uint32_t mlp_gate_up_dynamic_threadgroup_memory_bytes;
    uint32_t mlp_down_dynamic_threadgroup_memory_bytes;
    uint32_t thread_execution_width;
    uint32_t threadgroup_barriers_per_projection;
    char gdn_input_function_name[kBodyInputFunctionNameCapacity];
    char gdn_output_function_name[kBodyInputFunctionNameCapacity];
    char mlp_gate_up_function_name[kBodyInputFunctionNameCapacity];
    char mlp_down_function_name[kBodyInputFunctionNameCapacity];
};

static_assert(sizeof(GdnParams) == 52, "Metal W8 GDN parameter ABI changed");
static_assert(sizeof(MlpParams) == 16, "Metal W8 MLP parameter ABI changed");
static_assert(sizeof(LinearLayerParams) == 8,
              "Metal W8 linear-layer parameter ABI changed");
static_assert(sizeof(BoundaryMlpDescriptorV1) == 56,
              "Metal W8 MLP-to-Stack3 boundary descriptor ABI changed");
static_assert(sizeof(BoundaryStackLayerDescriptorV1) == 152,
              "Metal W8 stack3 layer descriptor ABI changed");
static_assert(sizeof(BoundaryStateDescriptorV1) == 64,
              "Metal W8 stack3 state descriptor ABI changed");
static_assert(sizeof(BoundaryMutableStateDescriptorV1) == 64,
              "Metal W8 stack3 mutable state descriptor ABI changed");
static_assert(sizeof(BoundaryExecutionReceiptV1) == 40,
              "Metal W8 stack3 execution receipt ABI changed");
static_assert(sizeof(BodyInputStagingRuntimeReceiptV1) == 292,
              "Metal W8 body-input-staging runtime receipt ABI changed");

struct BoundaryStackShape {
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
    uint32_t intermediate_size;
    size_t gate_up_rows;
    size_t gdn_input_weight_bytes;
    size_t gdn_input_scale_count;
    size_t gdn_output_weight_bytes;
    size_t gdn_output_scale_count;
    size_t conv_count;
    size_t mlp_gate_up_weight_bytes;
    size_t mlp_gate_up_scale_count;
    size_t mlp_down_weight_bytes;
    size_t mlp_down_scale_count;
    size_t query_state_count;
    size_t value_state_count;
    size_t recurrent_state_count;
};

struct BoundaryMlpShape {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    size_t gate_up_rows;
    size_t gate_up_weight_bytes;
    size_t gate_up_scale_count;
    size_t down_weight_bytes;
    size_t down_scale_count;
};

struct BoundaryStackLayer {
    // Fourteen immutable packed-weight/parameter buffers.
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

    // Four active and four scratch state buffers. Scratch is never published
    // until all three encoders and the final output check succeed.
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

struct ApxinfMetalW8MlpStack3BoundaryHandleV1 {
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
    id<MTLFunction> gdn_input_function;
    id<MTLFunction> gdn_output_function;
    id<MTLFunction> mlp_gate_up_function;
    id<MTLFunction> mlp_down_function;

    // Five immutable full-attention boundary MLP/RMS buffers. They extend the
    // Stack3 resident set without allocating another hidden or scratch row.
    id<MTLBuffer> boundary_gate_up_weights;
    id<MTLBuffer> boundary_gate_up_scales;
    id<MTLBuffer> boundary_down_weights;
    id<MTLBuffer> boundary_down_scales;
    id<MTLBuffer> boundary_post_attention_rms_weight;
    MlpParams boundary_mlp_params;
    LinearLayerParams boundary_layer_params;

    BoundaryStackLayer layers[kStackDepth];

    // Exactly two shared hidden rows ping-pong across the three encoders.
    id<MTLBuffer> hidden_a;
    id<MTLBuffer> hidden_b;

    // Exactly eight private activation buffers, reused serially by each layer.
    id<MTLBuffer> normalized;
    id<MTLBuffer> projected;
    id<MTLBuffer> processed;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;
    id<MTLBuffer> branch_output;
    id<MTLBuffer> mlp_gate_up;
    id<MTLBuffer> mlp_activated;

    uint32_t requested_body_input_staging;
    uint32_t gdn_input_dynamic_threadgroup_memory_bytes;
    uint32_t gdn_output_dynamic_threadgroup_memory_bytes;
    uint32_t mlp_gate_up_dynamic_threadgroup_memory_bytes;
    uint32_t mlp_down_dynamic_threadgroup_memory_bytes;
    uint32_t boundary_mlp_down_dynamic_threadgroup_memory_bytes;
    uint32_t stack_mlp_down_dynamic_threadgroup_memory_bytes;
    uint32_t seeded_mask;
    bool terminal_error;
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

bool valid_body_input_staging_selector(uint32_t profile) {
    return profile == kBodyInputStagingLegacyDevice ||
           profile == kBodyInputStagingThreadgroupShared;
}

bool function_has_name(id<MTLFunction> function, NSString *name) {
    return function != nil && function.name != nil &&
           [function.name isEqualToString:name];
}

uint32_t observed_body_input_staging(
    const ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle) {
    if (handle == nullptr) {
        return kUnknownBodyInputStaging;
    }
    if (function_has_name(handle->gdn_input_function,
                          @"gdn_w8_input_projection") &&
        function_has_name(handle->gdn_output_function,
                          @"gdn_w8_output_projection_g32") &&
        function_has_name(handle->mlp_gate_up_function,
                          @"w8_mlp_gate_up") &&
        function_has_name(handle->mlp_down_function, @"w8_mlp_down")) {
        return kBodyInputStagingLegacyDevice;
    }
    if (function_has_name(handle->gdn_input_function,
                          @"gdn_w8_input_projection_tg_shared") &&
        function_has_name(handle->gdn_output_function,
                          @"gdn_w8_output_projection_g32_tg_shared") &&
        function_has_name(handle->mlp_gate_up_function,
                          @"w8_mlp_gate_up_tg_shared") &&
        function_has_name(handle->mlp_down_function,
                          @"w8_mlp_down_tg_shared")) {
        return kBodyInputStagingThreadgroupShared;
    }
    return kUnknownBodyInputStaging;
}

bool checked_f32_threadgroup_bytes(uint32_t columns, uint32_t *output) {
    size_t bytes = 0;
    if (output == nullptr ||
        !checked_product(static_cast<size_t>(columns), sizeof(float), &bytes) ||
        bytes > UINT32_MAX) {
        return false;
    }
    *output = static_cast<uint32_t>(bytes);
    return true;
}

bool pipeline_supports_threadgroup_input(
    id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
    uint32_t dynamic_bytes) {
    if (device == nil || pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kMatVecThreads ||
        pipeline.threadExecutionWidth != kExpectedThreadExecutionWidth) {
        return false;
    }
    const NSUInteger maximum = device.maxThreadgroupMemoryLength;
    const NSUInteger static_bytes = pipeline.staticThreadgroupMemoryLength;
    return static_bytes <= maximum &&
           static_cast<NSUInteger>(dynamic_bytes) <= maximum - static_bytes;
}

bool live_body_input_staging_matches(
    const ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle) {
    if (handle == nullptr ||
        observed_body_input_staging(handle) !=
            handle->requested_body_input_staging) {
        return false;
    }
    const bool staged = handle->requested_body_input_staging ==
                        kBodyInputStagingThreadgroupShared;
    return pipeline_supports_threadgroup_input(
               handle->device, handle->gdn_input_pipeline,
               staged ? handle->gdn_input_dynamic_threadgroup_memory_bytes : 0) &&
           pipeline_supports_threadgroup_input(
               handle->device, handle->gdn_output_pipeline,
               staged ? handle->gdn_output_dynamic_threadgroup_memory_bytes : 0) &&
           pipeline_supports_threadgroup_input(
               handle->device, handle->mlp_gate_up_pipeline,
               staged ? handle->mlp_gate_up_dynamic_threadgroup_memory_bytes : 0) &&
           pipeline_supports_threadgroup_input(
               handle->device, handle->mlp_down_pipeline,
               staged ? handle->mlp_down_dynamic_threadgroup_memory_bytes : 0);
}

bool copy_function_name(id<MTLFunction> function, char *output,
                        size_t capacity) {
    const char *name = function == nil || function.name == nil
                           ? nullptr
                           : function.name.UTF8String;
    if (name == nullptr || output == nullptr || capacity == 0 ||
        std::strlen(name) >= capacity) {
        return false;
    }
    std::snprintf(output, capacity, "%s", name);
    return true;
}

bool write_body_input_staging_receipt(
    const ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
    BodyInputStagingRuntimeReceiptV1 *receipt) {
    if (handle == nullptr || receipt == nullptr ||
        !live_body_input_staging_matches(handle)) {
        return false;
    }
    *receipt = BodyInputStagingRuntimeReceiptV1{};
    receipt->requested_profile = handle->requested_body_input_staging;
    receipt->observed_profile = observed_body_input_staging(handle);
    receipt->threads_per_threadgroup = kMatVecThreads;
    const bool staged = receipt->observed_profile ==
                        kBodyInputStagingThreadgroupShared;
    if (staged) {
        receipt->gdn_input_dynamic_threadgroup_memory_bytes =
            handle->gdn_input_dynamic_threadgroup_memory_bytes;
        receipt->gdn_output_dynamic_threadgroup_memory_bytes =
            handle->gdn_output_dynamic_threadgroup_memory_bytes;
        receipt->mlp_gate_up_dynamic_threadgroup_memory_bytes =
            handle->mlp_gate_up_dynamic_threadgroup_memory_bytes;
        receipt->mlp_down_dynamic_threadgroup_memory_bytes =
            handle->mlp_down_dynamic_threadgroup_memory_bytes;
        receipt->threadgroup_barriers_per_projection = 1;
    }
    receipt->thread_execution_width = static_cast<uint32_t>(
        handle->gdn_input_pipeline.threadExecutionWidth);
    return copy_function_name(handle->gdn_input_function,
                              receipt->gdn_input_function_name,
                              sizeof(receipt->gdn_input_function_name)) &&
           copy_function_name(handle->gdn_output_function,
                              receipt->gdn_output_function_name,
                              sizeof(receipt->gdn_output_function_name)) &&
           copy_function_name(handle->mlp_gate_up_function,
                              receipt->mlp_gate_up_function_name,
                              sizeof(receipt->mlp_gate_up_function_name)) &&
           copy_function_name(handle->mlp_down_function,
                              receipt->mlp_down_function_name,
                              sizeof(receipt->mlp_down_function_name));
}

void set_body_input_threadgroup_memory(
    const ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
    id<MTLComputeCommandEncoder> encoder, uint32_t bytes) {
    if (handle->requested_body_input_staging ==
        kBodyInputStagingThreadgroupShared) {
        [encoder setThreadgroupMemoryLength:bytes atIndex:0];
    }
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

bool derive_boundary_mlp_shape(const BoundaryMlpDescriptorV1 &boundary,
                               uint32_t expected_hidden_size,
                               BoundaryMlpShape *shape) {
    if (shape == nullptr || boundary.gate_up_weights == nullptr ||
        boundary.gate_up_scales == nullptr || boundary.down_weights == nullptr ||
        boundary.down_scales == nullptr ||
        boundary.post_attention_rms_weight == nullptr ||
        boundary.hidden_size == 0 ||
        boundary.hidden_size != expected_hidden_size ||
        boundary.hidden_size % 64 != 0 || boundary.hidden_size % 4 != 0 ||
        boundary.intermediate_size == 0 || boundary.intermediate_size % 64 != 0 ||
        boundary.intermediate_size % 4 != 0 ||
        boundary.intermediate_size > UINT32_MAX / 2 ||
        !std::isfinite(boundary.rms_norm_eps) || boundary.rms_norm_eps < 0.0f) {
        return false;
    }
    shape->hidden_size = boundary.hidden_size;
    shape->intermediate_size = boundary.intermediate_size;
    shape->gate_up_rows = static_cast<size_t>(boundary.intermediate_size) * 2;
    return checked_product(shape->gate_up_rows, shape->hidden_size,
                           &shape->gate_up_weight_bytes) &&
           checked_product(shape->gate_up_rows, shape->hidden_size / 64,
                           &shape->gate_up_scale_count) &&
           checked_product(shape->hidden_size, shape->intermediate_size,
                           &shape->down_weight_bytes) &&
           checked_product(shape->hidden_size, shape->intermediate_size / 64,
                           &shape->down_scale_count);
}

bool same_shape(const BoundaryStackLayerDescriptorV1 &left,
                const BoundaryStackLayerDescriptorV1 &right) {
    return left.hidden_size == right.hidden_size &&
           left.key_heads == right.key_heads &&
           left.value_heads == right.value_heads && left.key_dim == right.key_dim &&
           left.value_dim == right.value_dim &&
           left.conv_kernel_size == right.conv_kernel_size &&
           left.gdn_rms_norm_eps == right.gdn_rms_norm_eps &&
           left.intermediate_size == right.intermediate_size &&
           left.layer_rms_norm_eps == right.layer_rms_norm_eps;
}

bool descriptor_pointers_valid(const BoundaryStackLayerDescriptorV1 &layer) {
    return layer.gdn_input_weights != nullptr && layer.gdn_input_scales != nullptr &&
           layer.gdn_output_weights != nullptr && layer.gdn_output_scales != nullptr &&
           layer.conv_weight != nullptr && layer.a_log != nullptr &&
           layer.dt_bias != nullptr && layer.gdn_norm_weight != nullptr &&
           layer.mlp_gate_up_weights != nullptr &&
           layer.mlp_gate_up_scales != nullptr &&
           layer.mlp_down_weights != nullptr && layer.mlp_down_scales != nullptr &&
           layer.input_rms_weight != nullptr &&
           layer.post_attention_rms_weight != nullptr;
}

bool derive_shape(const BoundaryStackLayerDescriptorV1 &layer, BoundaryStackShape *shape) {
    if (shape == nullptr || !descriptor_pointers_valid(layer) ||
        layer.hidden_size == 0 || layer.key_heads == 0 || layer.value_heads == 0 ||
        layer.key_dim == 0 || layer.value_dim == 0 ||
        layer.conv_kernel_size == 0 || layer.intermediate_size == 0 ||
        layer.value_heads % layer.key_heads != 0 || layer.hidden_size % 64 != 0 ||
        layer.intermediate_size % 64 != 0 || layer.hidden_size % 4 != 0 ||
        layer.intermediate_size % 4 != 0 ||
        layer.intermediate_size > UINT32_MAX / 2 ||
        !std::isfinite(layer.gdn_rms_norm_eps) || layer.gdn_rms_norm_eps < 0.0f ||
        !std::isfinite(layer.layer_rms_norm_eps) ||
        layer.layer_rms_norm_eps < 0.0f) {
        return false;
    }
    const uint64_t key_width64 =
        static_cast<uint64_t>(layer.key_heads) * layer.key_dim;
    const uint64_t value_width64 =
        static_cast<uint64_t>(layer.value_heads) * layer.value_dim;
    const uint64_t qkv_width64 = 2 * key_width64 + value_width64;
    const uint64_t input_rows64 = qkv_width64 + value_width64 + 2 * layer.value_heads;
    if (key_width64 > UINT32_MAX || value_width64 > UINT32_MAX ||
        qkv_width64 > UINT32_MAX || input_rows64 > UINT32_MAX ||
        value_width64 % 32 != 0 || value_width64 % 4 != 0) {
        return false;
    }
    shape->hidden_size = layer.hidden_size;
    shape->key_heads = layer.key_heads;
    shape->value_heads = layer.value_heads;
    shape->key_dim = layer.key_dim;
    shape->value_dim = layer.value_dim;
    shape->conv_kernel_size = layer.conv_kernel_size;
    shape->key_width = static_cast<uint32_t>(key_width64);
    shape->value_width = static_cast<uint32_t>(value_width64);
    shape->qkv_width = static_cast<uint32_t>(qkv_width64);
    shape->input_rows = static_cast<uint32_t>(input_rows64);
    shape->intermediate_size = layer.intermediate_size;
    shape->gate_up_rows = static_cast<size_t>(layer.intermediate_size) * 2;
    return checked_product(shape->input_rows, shape->hidden_size,
                           &shape->gdn_input_weight_bytes) &&
           checked_product(shape->input_rows, shape->hidden_size / 64,
                           &shape->gdn_input_scale_count) &&
           checked_product(shape->hidden_size, shape->value_width,
                           &shape->gdn_output_weight_bytes) &&
           checked_product(shape->hidden_size, shape->value_width / 32,
                           &shape->gdn_output_scale_count) &&
           checked_product(shape->qkv_width, shape->conv_kernel_size,
                           &shape->conv_count) &&
           checked_product(shape->gate_up_rows, shape->hidden_size,
                           &shape->mlp_gate_up_weight_bytes) &&
           checked_product(shape->gate_up_rows, shape->hidden_size / 64,
                           &shape->mlp_gate_up_scale_count) &&
           checked_product(shape->hidden_size, shape->intermediate_size,
                           &shape->mlp_down_weight_bytes) &&
           checked_product(shape->hidden_size, shape->intermediate_size / 64,
                           &shape->mlp_down_scale_count) &&
           checked_product(shape->key_width, shape->conv_kernel_size,
                           &shape->query_state_count) &&
           checked_product(shape->value_width, shape->conv_kernel_size,
                           &shape->value_state_count) &&
           checked_product(shape->value_heads, shape->key_dim,
                           &shape->recurrent_state_count) &&
           checked_product(shape->recurrent_state_count, shape->value_dim,
                           &shape->recurrent_state_count);
}

void allocate_layer(id<MTLDevice> device, BoundaryStackLayer *output,
                    const BoundaryStackLayerDescriptorV1 &source,
                    const BoundaryStackShape &shape) {
    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    output->gdn_input_weights =
        [device newBufferWithBytes:source.gdn_input_weights
                           length:shape.gdn_input_weight_bytes
                          options:shared];
    output->gdn_input_scales =
        [device newBufferWithBytes:source.gdn_input_scales
                           length:shape.gdn_input_scale_count * sizeof(float)
                          options:shared];
    output->gdn_output_weights =
        [device newBufferWithBytes:source.gdn_output_weights
                           length:shape.gdn_output_weight_bytes
                          options:shared];
    output->gdn_output_scales =
        [device newBufferWithBytes:source.gdn_output_scales
                           length:shape.gdn_output_scale_count * sizeof(float)
                          options:shared];
    output->conv_weight = [device newBufferWithBytes:source.conv_weight
                                              length:shape.conv_count * sizeof(float)
                                             options:shared];
    output->a_log = [device newBufferWithBytes:source.a_log
                                       length:shape.value_heads * sizeof(float)
                                      options:shared];
    output->dt_bias = [device newBufferWithBytes:source.dt_bias
                                         length:shape.value_heads * sizeof(float)
                                        options:shared];
    output->gdn_norm_weight =
        [device newBufferWithBytes:source.gdn_norm_weight
                           length:shape.value_dim * sizeof(float)
                          options:shared];
    output->mlp_gate_up_weights =
        [device newBufferWithBytes:source.mlp_gate_up_weights
                           length:shape.mlp_gate_up_weight_bytes
                          options:shared];
    output->mlp_gate_up_scales =
        [device newBufferWithBytes:source.mlp_gate_up_scales
                           length:shape.mlp_gate_up_scale_count * sizeof(float)
                          options:shared];
    output->mlp_down_weights =
        [device newBufferWithBytes:source.mlp_down_weights
                           length:shape.mlp_down_weight_bytes
                          options:shared];
    output->mlp_down_scales =
        [device newBufferWithBytes:source.mlp_down_scales
                           length:shape.mlp_down_scale_count * sizeof(float)
                          options:shared];
    output->input_rms_weight =
        [device newBufferWithBytes:source.input_rms_weight
                           length:shape.hidden_size * sizeof(float)
                          options:shared];
    output->post_attention_rms_weight =
        [device newBufferWithBytes:source.post_attention_rms_weight
                           length:shape.hidden_size * sizeof(float)
                          options:shared];

    output->query_state = make_shared_f32(device, shape.query_state_count);
    output->key_state = make_shared_f32(device, shape.query_state_count);
    output->value_state = make_shared_f32(device, shape.value_state_count);
    output->recurrent_state = make_shared_f32(device, shape.recurrent_state_count);
    output->query_scratch = make_shared_f32(device, shape.query_state_count);
    output->key_scratch = make_shared_f32(device, shape.query_state_count);
    output->value_scratch = make_shared_f32(device, shape.value_state_count);
    output->recurrent_scratch = make_shared_f32(device, shape.recurrent_state_count);

    output->gdn_params = GdnParams{
        shape.hidden_size,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        shape.conv_kernel_size,
        shape.key_width,
        shape.value_width,
        shape.qkv_width,
        shape.input_rows,
        shape.hidden_size / 64,
        shape.value_width / 32,
        source.gdn_rms_norm_eps,
    };
    output->mlp_params = MlpParams{
        shape.hidden_size,
        shape.intermediate_size,
        shape.hidden_size / 64,
        shape.intermediate_size / 64,
    };
    output->layer_params =
        LinearLayerParams{shape.hidden_size, source.layer_rms_norm_eps};
}

bool layer_buffers_valid(const BoundaryStackLayer &layer) {
    return layer.gdn_input_weights != nil && layer.gdn_input_scales != nil &&
           layer.gdn_output_weights != nil && layer.gdn_output_scales != nil &&
           layer.conv_weight != nil && layer.a_log != nil && layer.dt_bias != nil &&
           layer.gdn_norm_weight != nil && layer.mlp_gate_up_weights != nil &&
           layer.mlp_gate_up_scales != nil && layer.mlp_down_weights != nil &&
           layer.mlp_down_scales != nil && layer.input_rms_weight != nil &&
           layer.post_attention_rms_weight != nil && layer.query_state != nil &&
           layer.key_state != nil && layer.value_state != nil &&
           layer.recurrent_state != nil && layer.query_scratch != nil &&
           layer.key_scratch != nil && layer.value_scratch != nil &&
           layer.recurrent_scratch != nil;
}

bool valid_state_counts(const BoundaryStackLayer &layer, uint32_t query_count,
                        uint32_t key_count, uint32_t value_count,
                        uint32_t recurrent_count) {
    const size_t expected_query =
        static_cast<size_t>(layer.gdn_params.key_width) *
        layer.gdn_params.conv_kernel_size;
    const size_t expected_value =
        static_cast<size_t>(layer.gdn_params.value_width) *
        layer.gdn_params.conv_kernel_size;
    const size_t expected_recurrent =
        static_cast<size_t>(layer.gdn_params.value_heads) *
        layer.gdn_params.key_dim * layer.gdn_params.value_dim;
    return query_count == expected_query && key_count == expected_query &&
           value_count == expected_value && recurrent_count == expected_recurrent;
}

void encode_boundary_mlp(ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
                         id<MTLBuffer> input, id<MTLBuffer> output,
                         id<MTLComputeCommandEncoder> encoder) {
    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:handle->boundary_post_attention_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&handle->boundary_layer_params
                length:sizeof(handle->boundary_layer_params)
               atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_gate_up_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder, handle->mlp_gate_up_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:handle->boundary_gate_up_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->boundary_gate_up_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:3];
    [encoder setBytes:&handle->boundary_mlp_params
                length:sizeof(handle->boundary_mlp_params)
               atIndex:4];
    const uint32_t gate_up_rows =
        handle->boundary_mlp_params.intermediate_size * 2;
    [encoder dispatchThreadgroups:MTLSizeMake(
                (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_activation_pipeline];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:1];
    [encoder setBytes:&handle->boundary_mlp_params
                length:sizeof(handle->boundary_mlp_params)
               atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(
                handle->boundary_mlp_params.intermediate_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_down_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder,
        handle->boundary_mlp_down_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:handle->boundary_down_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->boundary_down_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:2];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:3];
    [encoder setBytes:&handle->boundary_mlp_params
                length:sizeof(handle->boundary_mlp_params)
               atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                (handle->boundary_mlp_params.hidden_size + kRowsPerThreadgroup - 1) /
                    kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&handle->boundary_layer_params
                length:sizeof(handle->boundary_layer_params)
               atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(
                handle->boundary_layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
}

void encode_layer(ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
                  BoundaryStackLayer &layer, id<MTLBuffer> input,
                  id<MTLBuffer> output,
                  id<MTLComputeCommandEncoder> encoder) {
    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:layer.input_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_input_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder, handle->gdn_input_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:layer.gdn_input_weights offset:0 atIndex:0];
    [encoder setBuffer:layer.gdn_input_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBuffer:handle->projected offset:0 atIndex:3];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                (layer.gdn_params.input_rows + kRowsPerThreadgroup - 1) /
                    kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_depthwise_pipeline];
    [encoder setBuffer:handle->projected offset:0 atIndex:0];
    [encoder setBuffer:layer.conv_weight offset:0 atIndex:1];
    [encoder setBuffer:layer.query_state offset:0 atIndex:2];
    [encoder setBuffer:layer.key_state offset:0 atIndex:3];
    [encoder setBuffer:layer.value_state offset:0 atIndex:4];
    [encoder setBuffer:layer.query_scratch offset:0 atIndex:5];
    [encoder setBuffer:layer.key_scratch offset:0 atIndex:6];
    [encoder setBuffer:layer.value_scratch offset:0 atIndex:7];
    [encoder setBuffer:handle->processed offset:0 atIndex:8];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:9];
    [encoder dispatchThreads:MTLSizeMake(layer.gdn_params.qkv_width, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_normalize_pipeline];
    [encoder setBuffer:handle->processed offset:0 atIndex:0];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(2 * layer.gdn_params.key_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
              std::min(kElementThreads, 2 * layer.gdn_params.key_heads), 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_recurrent_pipeline];
    [encoder setBuffer:handle->processed offset:0 atIndex:0];
    [encoder setBuffer:handle->projected offset:0 atIndex:1];
    [encoder setBuffer:layer.a_log offset:0 atIndex:2];
    [encoder setBuffer:layer.dt_bias offset:0 atIndex:3];
    [encoder setBuffer:layer.recurrent_state offset:0 atIndex:4];
    [encoder setBuffer:layer.recurrent_scratch offset:0 atIndex:5];
    [encoder setBuffer:handle->core offset:0 atIndex:6];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(layer.gdn_params.value_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_norm_gate_pipeline];
    [encoder setBuffer:handle->core offset:0 atIndex:0];
    [encoder setBuffer:handle->projected offset:0 atIndex:1];
    [encoder setBuffer:layer.gdn_norm_weight offset:0 atIndex:2];
    [encoder setBuffer:handle->gated offset:0 atIndex:3];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(layer.gdn_params.value_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
              std::min(kElementThreads, layer.gdn_params.value_heads), 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->gdn_output_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder, handle->gdn_output_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:layer.gdn_output_weights offset:0 atIndex:0];
    [encoder setBuffer:layer.gdn_output_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->gated offset:0 atIndex:2];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:3];
    [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                (layer.gdn_params.hidden_size + kRowsPerThreadgroup - 1) /
                    kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(layer.layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:layer.post_attention_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_gate_up_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder, handle->mlp_gate_up_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:layer.mlp_gate_up_weights offset:0 atIndex:0];
    [encoder setBuffer:layer.mlp_gate_up_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:3];
    [encoder setBytes:&layer.mlp_params length:sizeof(layer.mlp_params) atIndex:4];
    const uint32_t gate_up_rows = layer.mlp_params.intermediate_size * 2;
    [encoder dispatchThreadgroups:MTLSizeMake(
                (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_activation_pipeline];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:1];
    [encoder setBytes:&layer.mlp_params length:sizeof(layer.mlp_params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(layer.mlp_params.intermediate_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->mlp_down_pipeline];
    set_body_input_threadgroup_memory(
        handle, encoder, handle->stack_mlp_down_dynamic_threadgroup_memory_bytes);
    [encoder setBuffer:layer.mlp_down_weights offset:0 atIndex:0];
    [encoder setBuffer:layer.mlp_down_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:2];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:3];
    [encoder setBytes:&layer.mlp_params length:sizeof(layer.mlp_params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                (layer.mlp_params.hidden_size + kRowsPerThreadgroup - 1) /
                    kRowsPerThreadgroup,
                1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    buffer_barrier(encoder);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(layer.layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
}

}  // namespace

int create_mlp_stack3_boundary_gdn_out_g32_impl(
    const BoundaryMlpDescriptorV1 *boundary,
    const BoundaryStackLayerDescriptorV1 *layers, uint32_t layer_count,
    uint32_t body_input_staging, void **output, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 output handle is null");
            return 1;
        }
        *output = nullptr;
        if (!valid_body_input_staging_selector(body_input_staging)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 body-input-staging selector");
            return 1;
        }
        if (boundary == nullptr || layers == nullptr ||
            layer_count != kStackDepth) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 requires exactly three layer descriptors");
            return 1;
        }
        BoundaryStackShape shape{};
        if (!derive_shape(layers[0], &shape)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 packed contract");
            return 1;
        }
        for (uint32_t index = 1; index < kStackDepth; ++index) {
            BoundaryStackShape candidate{};
            if (!same_shape(layers[0], layers[index]) ||
                !derive_shape(layers[index], &candidate)) {
                write_error(error_output, error_capacity,
                            "Metal W8 MLP-to-Stack3 boundary v1 layer shapes or RMS epsilons differ");
                return 1;
            }
        }
        BoundaryMlpShape boundary_shape{};
        if (!derive_boundary_mlp_shape(*boundary, shape.hidden_size,
                                       &boundary_shape)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 prefix contract");
            return 1;
        }
        uint32_t gdn_input_dynamic_bytes = 0;
        uint32_t gdn_output_dynamic_bytes = 0;
        uint32_t mlp_gate_up_dynamic_bytes = 0;
        uint32_t boundary_mlp_down_dynamic_bytes = 0;
        uint32_t stack_mlp_down_dynamic_bytes = 0;
        if (!checked_f32_threadgroup_bytes(shape.hidden_size,
                                           &gdn_input_dynamic_bytes) ||
            !checked_f32_threadgroup_bytes(shape.value_width,
                                           &gdn_output_dynamic_bytes) ||
            !checked_f32_threadgroup_bytes(shape.hidden_size,
                                           &mlp_gate_up_dynamic_bytes) ||
            !checked_f32_threadgroup_bytes(
                boundary_shape.intermediate_size,
                &boundary_mlp_down_dynamic_bytes) ||
            !checked_f32_threadgroup_bytes(shape.intermediate_size,
                                           &stack_mlp_down_dynamic_bytes)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 dynamic threadgroup byte count overflows");
            return 1;
        }
        const uint32_t maximum_mlp_down_dynamic_bytes =
            std::max(boundary_mlp_down_dynamic_bytes,
                     stack_mlp_down_dynamic_bytes);

        auto handle =
            new (std::nothrow) ApxinfMetalW8MlpStack3BoundaryHandleV1{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal W8 MLP-to-Stack3 boundary v1 handle failed");
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
        const bool staged = body_input_staging ==
                            kBodyInputStagingThreadgroupShared;
        handle->gdn_input_function = [library
            newFunctionWithName:staged
                                    ? @"gdn_w8_input_projection_tg_shared"
                                    : @"gdn_w8_input_projection"];
        handle->gdn_input_pipeline =
            handle->gdn_input_function == nil
                ? nil
                : [handle->device
                      newComputePipelineStateWithFunction:handle->gdn_input_function
                                                   error:&error];
        handle->gdn_depthwise_pipeline =
            make_pipeline(handle->device, library, @"gdn_depthwise_preprocess", &error);
        handle->gdn_normalize_pipeline =
            make_pipeline(handle->device, library, @"gdn_normalize_qk", &error);
        handle->gdn_recurrent_pipeline =
            make_pipeline(handle->device, library, @"gdn_recurrent_update", &error);
        handle->gdn_norm_gate_pipeline =
            make_pipeline(handle->device, library, @"gdn_norm_gate", &error);
        handle->gdn_output_function = [library
            newFunctionWithName:staged
                                    ? @"gdn_w8_output_projection_g32_tg_shared"
                                    : @"gdn_w8_output_projection_g32"];
        handle->gdn_output_pipeline =
            handle->gdn_output_function == nil
                ? nil
                : [handle->device
                      newComputePipelineStateWithFunction:handle->gdn_output_function
                                                   error:&error];
        handle->mlp_gate_up_function =
            [library newFunctionWithName:staged ? @"w8_mlp_gate_up_tg_shared"
                                                 : @"w8_mlp_gate_up"];
        handle->mlp_gate_up_pipeline =
            handle->mlp_gate_up_function == nil
                ? nil
                : [handle->device
                      newComputePipelineStateWithFunction:handle->mlp_gate_up_function
                                                   error:&error];
        handle->mlp_activation_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_silu_mul", &error);
        handle->mlp_down_function =
            [library newFunctionWithName:staged ? @"w8_mlp_down_tg_shared"
                                                 : @"w8_mlp_down"];
        handle->mlp_down_pipeline =
            handle->mlp_down_function == nil
                ? nil
                : [handle->device
                      newComputePipelineStateWithFunction:handle->mlp_down_function
                                                   error:&error];
        handle->requested_body_input_staging = body_input_staging;
        handle->gdn_input_dynamic_threadgroup_memory_bytes =
            gdn_input_dynamic_bytes;
        handle->gdn_output_dynamic_threadgroup_memory_bytes =
            gdn_output_dynamic_bytes;
        handle->mlp_gate_up_dynamic_threadgroup_memory_bytes =
            mlp_gate_up_dynamic_bytes;
        handle->mlp_down_dynamic_threadgroup_memory_bytes =
            maximum_mlp_down_dynamic_bytes;
        handle->boundary_mlp_down_dynamic_threadgroup_memory_bytes =
            boundary_mlp_down_dynamic_bytes;
        handle->stack_mlp_down_dynamic_threadgroup_memory_bytes =
            stack_mlp_down_dynamic_bytes;
        if (handle->layer_rms_pipeline == nil || handle->residual_pipeline == nil ||
            handle->gdn_input_pipeline == nil ||
            handle->gdn_depthwise_pipeline == nil ||
            handle->gdn_normalize_pipeline == nil ||
            handle->gdn_recurrent_pipeline == nil ||
            handle->gdn_norm_gate_pipeline == nil ||
            handle->gdn_output_pipeline == nil ||
            handle->mlp_gate_up_pipeline == nil ||
            handle->mlp_activation_pipeline == nil ||
            handle->mlp_down_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        if (!live_body_input_staging_matches(handle)) {
            delete handle;
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 live body-input-staging pipelines or threadgroup resources do not match the requested selector");
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        handle->boundary_gate_up_weights =
            [handle->device newBufferWithBytes:boundary->gate_up_weights
                                        length:boundary_shape.gate_up_weight_bytes
                                       options:shared];
        handle->boundary_gate_up_scales =
            [handle->device newBufferWithBytes:boundary->gate_up_scales
                                        length:boundary_shape.gate_up_scale_count *
                                               sizeof(float)
                                       options:shared];
        handle->boundary_down_weights =
            [handle->device newBufferWithBytes:boundary->down_weights
                                        length:boundary_shape.down_weight_bytes
                                       options:shared];
        handle->boundary_down_scales =
            [handle->device newBufferWithBytes:boundary->down_scales
                                        length:boundary_shape.down_scale_count *
                                               sizeof(float)
                                       options:shared];
        handle->boundary_post_attention_rms_weight =
            [handle->device newBufferWithBytes:boundary->post_attention_rms_weight
                                        length:boundary_shape.hidden_size * sizeof(float)
                                       options:shared];
        handle->boundary_mlp_params = MlpParams{
            boundary_shape.hidden_size,
            boundary_shape.intermediate_size,
            boundary_shape.hidden_size / 64,
            boundary_shape.intermediate_size / 64,
        };
        handle->boundary_layer_params = LinearLayerParams{
            boundary_shape.hidden_size,
            boundary->rms_norm_eps,
        };
        for (uint32_t index = 0; index < kStackDepth; ++index) {
            allocate_layer(handle->device, &handle->layers[index], layers[index], shape);
        }
        handle->hidden_a = make_shared_f32(handle->device, shape.hidden_size);
        handle->hidden_b = make_shared_f32(handle->device, shape.hidden_size);
        handle->normalized = make_private_f32(handle->device, shape.hidden_size);
        handle->projected = make_private_f32(handle->device, shape.input_rows);
        handle->processed = make_private_f32(handle->device, shape.qkv_width);
        handle->core = make_private_f32(handle->device, shape.value_width);
        handle->gated = make_private_f32(handle->device, shape.value_width);
        handle->branch_output = make_private_f32(handle->device, shape.hidden_size);
        const size_t maximum_gate_up_rows =
            std::max(shape.gate_up_rows, boundary_shape.gate_up_rows);
        const size_t maximum_intermediate_size =
            std::max(static_cast<size_t>(shape.intermediate_size),
                     static_cast<size_t>(boundary_shape.intermediate_size));
        handle->mlp_gate_up =
            make_private_f32(handle->device, maximum_gate_up_rows);
        handle->mlp_activated =
            make_private_f32(handle->device, maximum_intermediate_size);
        bool layers_valid = true;
        for (const BoundaryStackLayer &layer : handle->layers) {
            layers_valid = layers_valid && layer_buffers_valid(layer);
        }
        if (handle->queue == nil ||
            handle->boundary_gate_up_weights == nil ||
            handle->boundary_gate_up_scales == nil ||
            handle->boundary_down_weights == nil ||
            handle->boundary_down_scales == nil ||
            handle->boundary_post_attention_rms_weight == nil ||
            !layers_valid || handle->hidden_a == nil ||
            handle->hidden_b == nil || handle->normalized == nil ||
            handle->projected == nil || handle->processed == nil ||
            handle->core == nil || handle->gated == nil ||
            handle->branch_output == nil || handle->mlp_gate_up == nil ||
            handle->mlp_activated == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal W8 MLP-to-Stack3 boundary v1 buffers failed");
            return 1;
        }
        handle->seeded_mask = 0;
        handle->terminal_error = false;
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_v1(
    const BoundaryMlpDescriptorV1 *boundary,
    const BoundaryStackLayerDescriptorV1 *layers, uint32_t layer_count,
    void **output, char *error_output, size_t error_capacity) {
    return create_mlp_stack3_boundary_gdn_out_g32_impl(
        boundary, layers, layer_count, kBodyInputStagingLegacyDevice, output,
        error_output, error_capacity);
}

extern "C" int
apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_with_body_input_staging_v1(
    const BoundaryMlpDescriptorV1 *boundary,
    const BoundaryStackLayerDescriptorV1 *layers, uint32_t layer_count,
    uint32_t body_input_staging, void **output, char *error_output,
    size_t error_capacity) {
    return create_mlp_stack3_boundary_gdn_out_g32_impl(
        boundary, layers, layer_count, body_input_staging, output,
        error_output, error_capacity);
}

extern "C" int
apxinf_metal_w8_mlp_stack3_boundary_body_input_staging_receipt_v1(
    void *opaque_handle, BodyInputStagingRuntimeReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (receipt != nullptr) {
            *receipt = BodyInputStagingRuntimeReceiptV1{};
        }
        auto handle = static_cast<ApxinfMetalW8MlpStack3BoundaryHandleV1 *>(
            opaque_handle);
        if (!write_body_input_staging_receipt(handle, receipt)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 live body-input-staging receipt is invalid");
            return 1;
        }
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_stack3_boundary_seed_states_v1(
    void *opaque_handle, const BoundaryStateDescriptorV1 *states,
    uint32_t state_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8MlpStack3BoundaryHandleV1 *>(opaque_handle);
        if (handle == nullptr || states == nullptr || state_count != kStackDepth) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 requires exactly three seed states");
            return 1;
        }
        // Validate all slots before copying any slot, so the seed mask can only
        // transition from 000 to 111.
        for (uint32_t slot = 0; slot < kStackDepth; ++slot) {
            const BoundaryStateDescriptorV1 &state = states[slot];
            if (state.query_state == nullptr || state.key_state == nullptr ||
                state.value_state == nullptr || state.recurrent_state == nullptr ||
                !valid_state_counts(handle->layers[slot], state.query_count,
                                    state.key_count, state.value_count,
                                    state.recurrent_count) ||
                !all_finite(state.query_state, state.query_count) ||
                !all_finite(state.key_state, state.key_count) ||
                !all_finite(state.value_state, state.value_count) ||
                !all_finite(state.recurrent_state, state.recurrent_count)) {
                write_error(error_output, error_capacity,
                            "invalid Metal W8 MLP-to-Stack3 boundary v1 seed state");
                return 1;
            }
        }
        for (uint32_t slot = 0; slot < kStackDepth; ++slot) {
            BoundaryStackLayer &layer = handle->layers[slot];
            const BoundaryStateDescriptorV1 &state = states[slot];
            const size_t query_bytes =
                static_cast<size_t>(state.query_count) * sizeof(float);
            const size_t value_bytes =
                static_cast<size_t>(state.value_count) * sizeof(float);
            const size_t recurrent_bytes =
                static_cast<size_t>(state.recurrent_count) * sizeof(float);
            std::memcpy(layer.query_state.contents, state.query_state, query_bytes);
            std::memcpy(layer.query_scratch.contents, state.query_state, query_bytes);
            std::memcpy(layer.key_state.contents, state.key_state, query_bytes);
            std::memcpy(layer.key_scratch.contents, state.key_state, query_bytes);
            std::memcpy(layer.value_state.contents, state.value_state, value_bytes);
            std::memcpy(layer.value_scratch.contents, state.value_state, value_bytes);
            std::memcpy(layer.recurrent_state.contents, state.recurrent_state,
                        recurrent_bytes);
            std::memcpy(layer.recurrent_scratch.contents, state.recurrent_state,
                        recurrent_bytes);
        }
        handle->seeded_mask = kAllSeededMask;
        handle->terminal_error = false;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_stack3_boundary_decode_v1(
    void *opaque_handle, const float *input, uint32_t input_count, float *output,
    uint32_t output_count, uint8_t inject_failure_after_execution,
    BoundaryExecutionReceiptV1 *receipt, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        if (receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 execution receipt is null");
            return 1;
        }
        *receipt = BoundaryExecutionReceiptV1{};
        auto handle =
            static_cast<ApxinfMetalW8MlpStack3BoundaryHandleV1 *>(opaque_handle);
        if (handle == nullptr || input == nullptr || output == nullptr ||
            input_count != output_count ||
            (handle != nullptr &&
             input_count != handle->layers[0].gdn_params.hidden_size)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 input or output");
            return 1;
        }
        if (handle->terminal_error) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 is terminal until reset");
            return 1;
        }
        if (!live_body_input_staging_matches(handle)) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 live body-input-staging selection changed after create");
            return 1;
        }
        if (handle->seeded_mask != kAllSeededMask) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 seed mask is not 111");
            return 1;
        }
        if (!all_finite(input, input_count)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 input is non-finite");
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
                        "create Metal W8 MLP-to-Stack3 boundary v1 command buffer failed");
            return 1;
        }
        receipt->command_buffers = 1;
        id<MTLComputeCommandEncoder> boundary_encoder =
            [command computeCommandEncoder];
        if (boundary_encoder == nil) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "create Metal W8 MLP-to-Stack3 boundary v1 prefix encoder failed");
            return 1;
        }
        receipt->compute_encoders = 1;
        encode_boundary_mlp(handle, handle->hidden_a, handle->hidden_b,
                            boundary_encoder);
        [boundary_encoder endEncoding];
        for (uint32_t slot = 0; slot < kStackDepth; ++slot) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil) {
                handle->terminal_error = true;
                write_error(error_output, error_capacity,
                            "create Metal W8 MLP-to-Stack3 boundary v1 compute encoder failed");
                return 1;
            }
            receipt->compute_encoders += 1;
            id<MTLBuffer> layer_input =
                slot % 2 == 0 ? handle->hidden_b : handle->hidden_a;
            id<MTLBuffer> layer_output =
                slot % 2 == 0 ? handle->hidden_a : handle->hidden_b;
            encode_layer(handle, handle->layers[slot], layer_input, layer_output,
                         encoder);
            [encoder endEncoding];
        }
        [command commit];
        receipt->commits = 1;
        [command waitUntilCompleted];
        receipt->waits = 1;
        if (command.status != MTLCommandBufferStatusCompleted) {
            handle->terminal_error = true;
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (inject_failure_after_execution) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "injected Metal W8 MLP-to-Stack3 boundary v1 failure after scratch execution");
            return 1;
        }
        // This version intentionally has no host checks after the boundary MLP
        // or stack layers 0 and 1. Only final hidden_a is checked before the
        // all-or-nothing state swap and host output publication.
        const float *final_values =
            static_cast<const float *>(handle->hidden_a.contents);
        if (!all_finite(final_values, input_count)) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 final output is non-finite");
            return 1;
        }
        for (BoundaryStackLayer &layer : handle->layers) {
            std::swap(layer.query_state, layer.query_scratch);
            std::swap(layer.key_state, layer.key_scratch);
            std::swap(layer.value_state, layer.value_scratch);
            std::swap(layer.recurrent_state, layer.recurrent_scratch);
        }
        receipt->state_commits = kStackDepth;
        receipt->state_commit_mask = kAllSeededMask;
        std::memcpy(output, final_values, hidden_bytes);
        receipt->device_to_host_bytes = hidden_bytes;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_stack3_boundary_snapshot_state_v1(
    void *opaque_handle, uint32_t slot, BoundaryMutableStateDescriptorV1 *state,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalW8MlpStack3BoundaryHandleV1 *>(opaque_handle);
        if (handle == nullptr || state == nullptr || slot >= kStackDepth ||
            handle->seeded_mask != kAllSeededMask || state->query_state == nullptr ||
            state->key_state == nullptr || state->value_state == nullptr ||
            state->recurrent_state == nullptr ||
            !valid_state_counts(handle->layers[slot], state->query_count,
                                state->key_count, state->value_count,
                                state->recurrent_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 state snapshot");
            return 1;
        }
        BoundaryStackLayer &layer = handle->layers[slot];
        std::memcpy(state->query_state, layer.query_state.contents,
                    static_cast<size_t>(state->query_count) * sizeof(float));
        std::memcpy(state->key_state, layer.key_state.contents,
                    static_cast<size_t>(state->key_count) * sizeof(float));
        std::memcpy(state->value_state, layer.value_state.contents,
                    static_cast<size_t>(state->value_count) * sizeof(float));
        std::memcpy(state->recurrent_state, layer.recurrent_state.contents,
                    static_cast<size_t>(state->recurrent_count) * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_mlp_stack3_boundary_destroy_v1(
    void *opaque_handle) {
    auto handle =
        static_cast<ApxinfMetalW8MlpStack3BoundaryHandleV1 *>(opaque_handle);
    delete handle;
}
