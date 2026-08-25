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
constexpr uint32_t kLegacyProfile = 0;
constexpr uint32_t kFusedProfile = 2;
constexpr uint32_t kFusedThreads = 128;
constexpr uint32_t kRequiredThreadExecutionWidth = 32;
constexpr uint32_t kFusedSourceThreadgroupBytes = (4 * 128 + 3) * sizeof(float);
constexpr uint32_t kFusedStaticThreadgroupBytes = 2064;
constexpr size_t kFunctionChainCapacity = 256;
constexpr float kFixedGdnRmsNormEps = 1.0e-6f;

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
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
    uint32_t gdn_core_seams;
    uint32_t gdn_core_kernel_dispatches;
    uint32_t gdn_core_explicit_buffer_barriers;
    uint32_t threads_per_threadgroup;
    uint32_t gdn_core_threadgroups;
    uint32_t gdn_core_launched_threads;
    uint32_t pipeline_thread_execution_width;
    uint32_t source_declared_threadgroup_memory_bytes;
    uint32_t pipeline_static_threadgroup_memory_bytes;
    uint32_t internal_threadgroup_barrier_sites_per_threadgroup;
    uint32_t fixed_shape_validated;
    uint32_t rms_norm_eps_bits;
    uint32_t persistent_output_groups_per_row;
    uint32_t core_kernel_output_groups_per_row;
    char observed_function_chain[kFunctionChainCapacity];
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
static_assert(sizeof(BoundaryExecutionReceiptV1) == 368,
              "Metal W8 stack3 execution receipt ABI changed");

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
    id<MTLComputePipelineState> gdn_core_fused_pipeline;
    id<MTLComputePipelineState> gdn_output_pipeline;
    id<MTLComputePipelineState> mlp_gate_up_pipeline;
    id<MTLComputePipelineState> mlp_activation_pipeline;
    id<MTLComputePipelineState> mlp_down_pipeline;

    id<MTLFunction> gdn_depthwise_function;
    id<MTLFunction> gdn_normalize_function;
    id<MTLFunction> gdn_recurrent_function;
    id<MTLFunction> gdn_norm_gate_function;
    id<MTLFunction> gdn_core_fused_function;

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

    uint32_t seeded_mask;
    uint32_t gdn_core_profile;
    bool production_receipt_enabled;
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

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLFunction> function, NSError **error) {
    return function == nil ? nil
                           : [device newComputePipelineStateWithFunction:function error:error];
}

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "F32 bit custody changed");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool valid_production_profile(uint32_t profile) {
    return profile == kLegacyProfile || profile == kFusedProfile;
}

bool fixed_fused_shape(const BoundaryStackLayerDescriptorV1 &layer) {
    return layer.hidden_size == 1024 && layer.key_heads == 16 &&
           layer.value_heads == 16 && layer.value_heads == layer.key_heads &&
           layer.key_dim == 128 && layer.value_dim == 128 &&
           layer.conv_kernel_size == 4 &&
           f32_bits(layer.gdn_rms_norm_eps) == f32_bits(kFixedGdnRmsNormEps);
}

bool live_fixed_shape(const GdnParams &params) {
    return params.hidden_size == 1024 && params.key_heads == 16 &&
           params.value_heads == 16 && params.value_heads == params.key_heads &&
           params.key_dim == 128 && params.value_dim == 128 &&
           params.conv_kernel_size == 4 && params.key_width == 2048 &&
           params.value_width == 2048 && params.qkv_width == 6144 &&
           params.input_rows == 8224 && params.input_groups_per_row == 16 &&
           params.output_groups_per_row == 64 &&
           f32_bits(params.rms_norm_eps) == f32_bits(kFixedGdnRmsNormEps);
}

bool live_pipeline_matches(id<MTLFunction> function,
                           id<MTLComputePipelineState> pipeline,
                           NSString *expected_name, uint32_t required_threads,
                           uint32_t expected_static_threadgroup_bytes) {
    return function != nil && pipeline != nil && expected_name != nil &&
           [function.name isEqualToString:expected_name] &&
           pipeline.threadExecutionWidth == kRequiredThreadExecutionWidth &&
           pipeline.maxTotalThreadsPerThreadgroup >= required_threads &&
           pipeline.staticThreadgroupMemoryLength ==
               expected_static_threadgroup_bytes;
}

bool live_profile_matches(ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle) {
    if (handle == nullptr || !valid_production_profile(handle->gdn_core_profile)) {
        return false;
    }
    if (handle->gdn_core_profile == kFusedProfile) {
        return live_pipeline_matches(handle->gdn_core_fused_function,
                                     handle->gdn_core_fused_pipeline,
                                     @"gdn_core_fused_v1", kFusedThreads,
                                     kFusedStaticThreadgroupBytes);
    }
    if (!handle->production_receipt_enabled) {
        return handle->gdn_depthwise_function != nil &&
               handle->gdn_depthwise_pipeline != nil &&
               handle->gdn_normalize_function != nil &&
               handle->gdn_normalize_pipeline != nil &&
               handle->gdn_recurrent_function != nil &&
               handle->gdn_recurrent_pipeline != nil &&
               handle->gdn_norm_gate_function != nil &&
               handle->gdn_norm_gate_pipeline != nil;
    }
    return live_pipeline_matches(handle->gdn_depthwise_function,
                                 handle->gdn_depthwise_pipeline,
                                 @"gdn_depthwise_preprocess", kElementThreads, 0) &&
           live_pipeline_matches(handle->gdn_normalize_function,
                                 handle->gdn_normalize_pipeline,
                                 @"gdn_normalize_qk", 32, 0) &&
           live_pipeline_matches(handle->gdn_recurrent_function,
                                 handle->gdn_recurrent_pipeline,
                                 @"gdn_recurrent_update", kElementThreads, 0) &&
           live_pipeline_matches(handle->gdn_norm_gate_function,
                                 handle->gdn_norm_gate_pipeline,
                                 @"gdn_norm_gate", 16, 0);
}

id<MTLComputePipelineState> selected_gdn_core_pipeline(
    ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle) {
    return handle->gdn_core_profile == kFusedProfile
               ? handle->gdn_core_fused_pipeline
               : handle->gdn_recurrent_pipeline;
}

void write_observed_function_chain(
    ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
    BoundaryExecutionReceiptV1 *receipt) {
    if (handle->gdn_core_profile == kFusedProfile) {
        std::snprintf(receipt->observed_function_chain, kFunctionChainCapacity,
                      "%s", handle->gdn_core_fused_function.name.UTF8String);
        return;
    }
    std::snprintf(receipt->observed_function_chain, kFunctionChainCapacity,
                  "%s|%s|%s|%s",
                  handle->gdn_depthwise_function.name.UTF8String,
                  handle->gdn_normalize_function.name.UTF8String,
                  handle->gdn_recurrent_function.name.UTF8String,
                  handle->gdn_norm_gate_function.name.UTF8String);
}

uint32_t expected_transaction_dispatches(uint32_t profile) {
    return profile == kFusedProfile ? 35 : 44;
}

uint32_t expected_transaction_barriers(uint32_t profile) {
    return profile == kFusedProfile ? 31 : 40;
}

uint32_t expected_gdn_core_dispatches(uint32_t profile) {
    return profile == kFusedProfile ? 3 : 12;
}

uint32_t expected_gdn_core_threadgroups(uint32_t profile) {
    return profile == kFusedProfile ? 48 : 126;
}

uint32_t expected_gdn_core_launched_threads(uint32_t profile) {
    return profile == kFusedProfile ? 6144 : 30864;
}

void record_dispatch(BoundaryExecutionReceiptV1 *receipt, bool gdn_core,
                     uint32_t threadgroups, uint32_t launched_threads) {
    ++receipt->kernel_dispatches;
    if (gdn_core) {
        ++receipt->gdn_core_kernel_dispatches;
        receipt->gdn_core_threadgroups += threadgroups;
        receipt->gdn_core_launched_threads += launched_threads;
    }
}

void observe_consistent_group_count(uint32_t *field, uint32_t observed) {
    if (*field == 0) {
        *field = observed;
    } else if (*field != observed) {
        *field = UINT32_MAX;
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

void buffer_barrier(id<MTLComputeCommandEncoder> encoder,
                    BoundaryExecutionReceiptV1 *receipt, bool gdn_core) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++receipt->explicit_buffer_barriers;
    if (gdn_core) {
        ++receipt->gdn_core_explicit_buffer_barriers;
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
                         id<MTLComputeCommandEncoder> encoder,
                         BoundaryExecutionReceiptV1 *receipt) {
    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:handle->boundary_post_attention_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&handle->boundary_layer_params
                length:sizeof(handle->boundary_layer_params)
               atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_gate_up_pipeline];
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
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_activation_pipeline];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:1];
    [encoder setBytes:&handle->boundary_mlp_params
                length:sizeof(handle->boundary_mlp_params)
               atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(
                handle->boundary_mlp_params.intermediate_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_down_pipeline];
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
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

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
    record_dispatch(receipt, false, 0, 0);
}

void encode_layer(ApxinfMetalW8MlpStack3BoundaryHandleV1 *handle,
                  BoundaryStackLayer &layer, id<MTLBuffer> input,
                  id<MTLBuffer> output,
                  id<MTLComputeCommandEncoder> encoder,
                  BoundaryExecutionReceiptV1 *receipt) {
    ++receipt->gdn_core_seams;
    observe_consistent_group_count(
        &receipt->persistent_output_groups_per_row,
        layer.gdn_params.output_groups_per_row);
    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:layer.input_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->gdn_input_pipeline];
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
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    if (handle->gdn_core_profile == kFusedProfile) {
        [encoder setComputePipelineState:handle->gdn_core_fused_pipeline];
        [encoder setBuffer:handle->projected offset:0 atIndex:0];
        [encoder setBuffer:layer.conv_weight offset:0 atIndex:1];
        [encoder setBuffer:layer.query_state offset:0 atIndex:2];
        [encoder setBuffer:layer.key_state offset:0 atIndex:3];
        [encoder setBuffer:layer.value_state offset:0 atIndex:4];
        [encoder setBuffer:layer.query_scratch offset:0 atIndex:5];
        [encoder setBuffer:layer.key_scratch offset:0 atIndex:6];
        [encoder setBuffer:layer.value_scratch offset:0 atIndex:7];
        [encoder setBuffer:layer.a_log offset:0 atIndex:8];
        [encoder setBuffer:layer.dt_bias offset:0 atIndex:9];
        [encoder setBuffer:layer.recurrent_state offset:0 atIndex:10];
        [encoder setBuffer:layer.recurrent_scratch offset:0 atIndex:11];
        [encoder setBuffer:layer.gdn_norm_weight offset:0 atIndex:12];
        [encoder setBuffer:handle->gated offset:0 atIndex:13];
        // The accepted fused primitive's fixed-shape guard requires the
        // core-local accepted guard count (32), derived from its legacy G64
        // packing contract. The core never indexes output scales, so custody
        // uses a by-value core-only parameter copy while the production G32
        // output projection retains the persistent value-width-derived count
        // in layer.gdn_params (64 groups).
        GdnParams fused_core_params = layer.gdn_params;
        fused_core_params.output_groups_per_row = 32;
        observe_consistent_group_count(
            &receipt->core_kernel_output_groups_per_row,
            fused_core_params.output_groups_per_row);
        [encoder setBytes:&fused_core_params
                    length:sizeof(fused_core_params)
                   atIndex:14];
        [encoder dispatchThreadgroups:MTLSizeMake(layer.gdn_params.value_heads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kFusedThreads, 1, 1)];
        record_dispatch(receipt, true, layer.gdn_params.value_heads,
                        layer.gdn_params.value_heads * kFusedThreads);
        buffer_barrier(encoder, receipt, true);
    } else {
        observe_consistent_group_count(
            &receipt->core_kernel_output_groups_per_row,
            layer.gdn_params.output_groups_per_row);
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
        record_dispatch(receipt, true,
                        (layer.gdn_params.qkv_width + kElementThreads - 1) /
                            kElementThreads,
                        layer.gdn_params.qkv_width);
        buffer_barrier(encoder, receipt, true);

        [encoder setComputePipelineState:handle->gdn_normalize_pipeline];
        [encoder setBuffer:handle->processed offset:0 atIndex:0];
        [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:1];
        const uint32_t normalize_threads = 2 * layer.gdn_params.key_heads;
        [encoder dispatchThreads:MTLSizeMake(normalize_threads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, normalize_threads), 1, 1)];
        record_dispatch(receipt, true, 1, normalize_threads);
        buffer_barrier(encoder, receipt, true);

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
        record_dispatch(receipt, true, layer.gdn_params.value_heads,
                        layer.gdn_params.value_heads * kElementThreads);
        buffer_barrier(encoder, receipt, true);

        [encoder setComputePipelineState:handle->gdn_norm_gate_pipeline];
        [encoder setBuffer:handle->core offset:0 atIndex:0];
        [encoder setBuffer:handle->projected offset:0 atIndex:1];
        [encoder setBuffer:layer.gdn_norm_weight offset:0 atIndex:2];
        [encoder setBuffer:handle->gated offset:0 atIndex:3];
        [encoder setBytes:&layer.gdn_params length:sizeof(layer.gdn_params) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(layer.gdn_params.value_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  std::min(kElementThreads, layer.gdn_params.value_heads), 1, 1)];
        record_dispatch(receipt, true, 1, layer.gdn_params.value_heads);
        buffer_barrier(encoder, receipt, true);
    }

    [encoder setComputePipelineState:handle->gdn_output_pipeline];
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
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(layer.layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->layer_rms_pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:layer.post_attention_rms_weight offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_gate_up_pipeline];
    [encoder setBuffer:layer.mlp_gate_up_weights offset:0 atIndex:0];
    [encoder setBuffer:layer.mlp_gate_up_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->normalized offset:0 atIndex:2];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:3];
    [encoder setBytes:&layer.mlp_params length:sizeof(layer.mlp_params) atIndex:4];
    const uint32_t gate_up_rows = layer.mlp_params.intermediate_size * 2;
    [encoder dispatchThreadgroups:MTLSizeMake(
                (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_activation_pipeline];
    [encoder setBuffer:handle->mlp_gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->mlp_activated offset:0 atIndex:1];
    [encoder setBytes:&layer.mlp_params length:sizeof(layer.mlp_params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(layer.mlp_params.intermediate_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->mlp_down_pipeline];
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
    record_dispatch(receipt, false, 0, 0);
    buffer_barrier(encoder, receipt, false);

    [encoder setComputePipelineState:handle->residual_pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:handle->branch_output offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&layer.layer_params length:sizeof(layer.layer_params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(layer.layer_params.hidden_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kElementThreads, 1, 1)];
    record_dispatch(receipt, false, 0, 0);
}

}  // namespace

extern "C" int apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_v1(
    const BoundaryMlpDescriptorV1 *boundary,
    const BoundaryStackLayerDescriptorV1 *layers, uint32_t layer_count,
    uint32_t gdn_core_profile, uint8_t require_fixed_shape, void **output,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 output handle is null");
            return 1;
        }
        *output = nullptr;
        if (boundary == nullptr || layers == nullptr ||
            layer_count != kStackDepth ||
            !valid_production_profile(gdn_core_profile) ||
            require_fixed_shape > 1 ||
            (gdn_core_profile == kFusedProfile && require_fixed_shape != 1)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 requires three layers and production profile 0 or 2; fused requires the strict fixed-shape contract");
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
        if (require_fixed_shape != 0) {
            for (uint32_t index = 0; index < kStackDepth; ++index) {
                if (!fixed_fused_shape(layers[index])) {
                    write_error(error_output, error_capacity,
                                "Metal W8 MLP-to-Stack3 production profile requires H=1024, KH=VH=16, KD=VD=128, conv=4, persistent output_groups=64, a fused core-local guard value of 32, and rms_norm_eps bits equal to 1e-6f32");
                    return 1;
                }
            }
        }
        BoundaryMlpShape boundary_shape{};
        if (!derive_boundary_mlp_shape(*boundary, shape.hidden_size,
                                       &boundary_shape)) {
            write_error(error_output, error_capacity,
                        "invalid Metal W8 MLP-to-Stack3 boundary v1 prefix contract");
            return 1;
        }

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
        handle->gdn_input_pipeline =
            make_pipeline(handle->device, library, @"gdn_w8_input_projection", &error);
        if (gdn_core_profile == kLegacyProfile) {
            handle->gdn_depthwise_function =
                [library newFunctionWithName:@"gdn_depthwise_preprocess"];
            handle->gdn_normalize_function =
                [library newFunctionWithName:@"gdn_normalize_qk"];
            handle->gdn_recurrent_function =
                [library newFunctionWithName:@"gdn_recurrent_update"];
            handle->gdn_norm_gate_function =
                [library newFunctionWithName:@"gdn_norm_gate"];
            handle->gdn_depthwise_pipeline = make_pipeline(
                handle->device, handle->gdn_depthwise_function, &error);
            handle->gdn_normalize_pipeline = make_pipeline(
                handle->device, handle->gdn_normalize_function, &error);
            handle->gdn_recurrent_pipeline = make_pipeline(
                handle->device, handle->gdn_recurrent_function, &error);
            handle->gdn_norm_gate_pipeline = make_pipeline(
                handle->device, handle->gdn_norm_gate_function, &error);
        } else {
            handle->gdn_core_fused_function =
                [library newFunctionWithName:@"gdn_core_fused_v1"];
            handle->gdn_core_fused_pipeline = make_pipeline(
                handle->device, handle->gdn_core_fused_function, &error);
        }
        handle->gdn_output_pipeline =
            make_pipeline(handle->device, library, @"gdn_w8_output_projection_g32", &error);
        handle->mlp_gate_up_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_gate_up", &error);
        handle->mlp_activation_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_silu_mul", &error);
        handle->mlp_down_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_down", &error);
        handle->gdn_core_profile = gdn_core_profile;
        handle->production_receipt_enabled = require_fixed_shape != 0;
        if (handle->layer_rms_pipeline == nil || handle->residual_pipeline == nil ||
            handle->gdn_input_pipeline == nil || !live_profile_matches(handle) ||
            handle->gdn_output_pipeline == nil ||
            handle->mlp_gate_up_pipeline == nil ||
            handle->mlp_activation_pipeline == nil ||
            handle->mlp_down_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
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
        if (!live_profile_matches(handle)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary live GDN profile identity changed");
            return 1;
        }
        if (handle->production_receipt_enabled) {
            for (const BoundaryStackLayer &layer : handle->layers) {
                if (!live_fixed_shape(layer.gdn_params)) {
                    write_error(error_output, error_capacity,
                                "Metal W8 MLP-to-Stack3 production fixed-shape contract changed before dispatch");
                    return 1;
                }
            }
        }
        if (handle->terminal_error) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary v1 is terminal until reset");
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
        receipt->requested_profile = handle->gdn_core_profile;

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
                            boundary_encoder, receipt);
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
                         encoder, receipt);
            [encoder endEncoding];
        }
        if (receipt->kernel_dispatches !=
                expected_transaction_dispatches(handle->gdn_core_profile) ||
            receipt->explicit_buffer_barriers !=
                expected_transaction_barriers(handle->gdn_core_profile) ||
            receipt->gdn_core_kernel_dispatches !=
                expected_gdn_core_dispatches(handle->gdn_core_profile) ||
            receipt->gdn_core_explicit_buffer_barriers !=
                expected_gdn_core_dispatches(handle->gdn_core_profile) ||
            receipt->gdn_core_seams != kStackDepth ||
            (handle->production_receipt_enabled &&
             (receipt->gdn_core_threadgroups !=
                  expected_gdn_core_threadgroups(handle->gdn_core_profile) ||
              receipt->gdn_core_launched_threads !=
                  expected_gdn_core_launched_threads(handle->gdn_core_profile) ||
              receipt->persistent_output_groups_per_row != 64 ||
              receipt->core_kernel_output_groups_per_row !=
                  (handle->gdn_core_profile == kFusedProfile ? 32 : 64)))) {
            handle->terminal_error = true;
            write_error(error_output, error_capacity,
                        "Metal W8 MLP-to-Stack3 boundary pre-commit topology mismatch");
            return 1;
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
        receipt->observed_profile = handle->gdn_core_profile;
        receipt->threads_per_threadgroup =
            handle->gdn_core_profile == kFusedProfile ? kFusedThreads
                                                      : kElementThreads;
        id<MTLComputePipelineState> selected_pipeline =
            selected_gdn_core_pipeline(handle);
        receipt->pipeline_thread_execution_width =
            static_cast<uint32_t>(selected_pipeline.threadExecutionWidth);
        receipt->source_declared_threadgroup_memory_bytes =
            handle->gdn_core_profile == kFusedProfile
                ? kFusedSourceThreadgroupBytes
                : 0;
        receipt->pipeline_static_threadgroup_memory_bytes =
            static_cast<uint32_t>(selected_pipeline.staticThreadgroupMemoryLength);
        receipt->internal_threadgroup_barrier_sites_per_threadgroup =
            handle->gdn_core_profile == kFusedProfile ? 4 : 0;
        receipt->fixed_shape_validated =
            handle->production_receipt_enabled ? 1 : 0;
        receipt->rms_norm_eps_bits =
            f32_bits(handle->layers[0].gdn_params.rms_norm_eps);
        write_observed_function_chain(handle, receipt);
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
