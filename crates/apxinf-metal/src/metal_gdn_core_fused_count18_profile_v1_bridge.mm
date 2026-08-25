#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kSeamsPerRun = 18;
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kKeyHeads = 16;
constexpr uint32_t kValueHeads = 16;
constexpr uint32_t kKeyDim = 128;
constexpr uint32_t kValueDim = 128;
constexpr uint32_t kConvKernelSize = 4;
constexpr uint32_t kKeyWidth = kKeyHeads * kKeyDim;
constexpr uint32_t kValueWidth = kValueHeads * kValueDim;
constexpr uint32_t kQkvWidth = 2 * kKeyWidth + kValueWidth;
constexpr uint32_t kProjectedPerSeam =
    kQkvWidth + kValueWidth + 2 * kValueHeads;
constexpr uint32_t kConvWeightPerSeam = kQkvWidth * kConvKernelSize;
constexpr uint32_t kQueryStatePerSeam = kKeyWidth * kConvKernelSize;
constexpr uint32_t kKeyStatePerSeam = kKeyWidth * kConvKernelSize;
constexpr uint32_t kValueStatePerSeam = kValueWidth * kConvKernelSize;
constexpr uint32_t kHeadScalarPerSeam = kValueHeads;
constexpr uint32_t kRecurrentPerSeam =
    kValueHeads * kKeyDim * kValueDim;
constexpr uint32_t kNormWeightPerSeam = kValueDim;
constexpr uint32_t kProcessedPerSeam = kQkvWidth;
constexpr uint32_t kCorePerSeam = kValueWidth;
constexpr uint32_t kGatedPerSeam = kValueWidth;

constexpr uint32_t kProjectedCount = kSeamsPerRun * kProjectedPerSeam;
constexpr uint32_t kConvWeightCount = kSeamsPerRun * kConvWeightPerSeam;
constexpr uint32_t kQueryStateCount = kSeamsPerRun * kQueryStatePerSeam;
constexpr uint32_t kKeyStateCount = kSeamsPerRun * kKeyStatePerSeam;
constexpr uint32_t kValueStateCount = kSeamsPerRun * kValueStatePerSeam;
constexpr uint32_t kHeadScalarCount = kSeamsPerRun * kHeadScalarPerSeam;
constexpr uint32_t kRecurrentCount = kSeamsPerRun * kRecurrentPerSeam;
constexpr uint32_t kNormWeightCount = kSeamsPerRun * kNormWeightPerSeam;
constexpr uint32_t kProcessedCount = kSeamsPerRun * kProcessedPerSeam;
constexpr uint32_t kCoreCount = kSeamsPerRun * kCorePerSeam;
constexpr uint32_t kGatedCount = kSeamsPerRun * kGatedPerSeam;

constexpr uint32_t kLegacyProfile = 0;
constexpr uint32_t kQkStagedProfile = 1;
constexpr uint32_t kFusedProfile = 2;
constexpr uint32_t kLegacyThreads = 256;
constexpr uint32_t kCandidateThreads = 128;
constexpr uint32_t kRequiredThreadExecutionWidth = 32;
constexpr uint32_t kQkStagedSourceThreadgroupBytes =
    (2 + kKeyDim + kKeyDim) * sizeof(float);
constexpr uint32_t kFusedSourceThreadgroupBytes =
    (4 * kValueDim + 3) * sizeof(float);
constexpr size_t kFunctionChainCapacity = 256;

constexpr uint32_t kDepthwiseSlot = 0;
constexpr uint32_t kNormalizeSlot = 1;
constexpr uint32_t kLegacyRecurrentSlot = 2;
constexpr uint32_t kQkStagedRecurrentSlot = 3;
constexpr uint32_t kNormGateSlot = 4;
constexpr uint32_t kFusedSlot = 5;
constexpr uint32_t kPipelineSlotCount = 6;

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

struct GdnCoreCount18ObservedTopologyV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
    uint32_t launched_threads;
    uint32_t threadgroups;
    uint32_t commits;
    uint32_t waits;
};

// This layout is mirrored verbatim by the Rust primitive. Keep the 26 u32
// fields consecutive so successful_runs stays naturally 8-byte aligned.
struct GdnCoreCount18RuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t seams_per_run;
    uint32_t kernel_dispatches_per_run;
    uint32_t explicit_buffer_barriers_per_run;
    uint32_t launched_threads_per_run;
    uint32_t threadgroups_per_run;
    uint32_t recurrent_threads_per_threadgroup;
    uint32_t pipeline_thread_execution_width;
    uint32_t pipeline_static_threadgroup_memory_bytes;
    uint32_t source_declared_threadgroup_memory_bytes;
    uint32_t internal_threadgroup_barrier_sites_per_threadgroup;
    uint32_t fixed_shape_host_validated;
    uint32_t input_output_buffers_non_overlapping;
    uint32_t command_buffers_per_run;
    uint32_t compute_encoders_per_run;
    uint32_t commits_per_run;
    uint32_t waits_per_run;
    uint32_t last_observed_kernel_dispatches;
    uint32_t last_observed_explicit_buffer_barriers;
    uint32_t last_observed_launched_threads;
    uint32_t last_observed_threadgroups;
    uint32_t last_observed_command_buffers;
    uint32_t last_observed_compute_encoders;
    uint32_t last_observed_commits;
    uint32_t last_observed_waits;
    uint64_t successful_runs;
    char observed_function_chain[kFunctionChainCapacity];
};

static_assert(sizeof(GdnParams) == 52, "Metal GDN parameter ABI changed");
static_assert(sizeof(GdnCoreCount18ObservedTopologyV1) == 32,
              "Metal GDN core observed-topology ABI changed");
static_assert(sizeof(GdnCoreCount18RuntimeReceiptV1) == 368,
              "Metal GDN core runtime-receipt ABI changed");

struct ApxinfMetalGdnCoreFusedCount18ProfileV1Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFunction> functions[kPipelineSlotCount];
    id<MTLComputePipelineState> pipelines[kPipelineSlotCount];

    // Every seam owns a disjoint slice, and every tensor owns a distinct
    // shared allocation. In particular, no input is ever reused as output.
    id<MTLBuffer> projected;
    id<MTLBuffer> conv_weight;
    id<MTLBuffer> query_state;
    id<MTLBuffer> key_state;
    id<MTLBuffer> value_state;
    id<MTLBuffer> a_log;
    id<MTLBuffer> dt_bias;
    id<MTLBuffer> recurrent_state;
    id<MTLBuffer> norm_weight;

    id<MTLBuffer> next_query_state;
    id<MTLBuffer> next_key_state;
    id<MTLBuffer> next_value_state;
    id<MTLBuffer> processed;
    id<MTLBuffer> next_recurrent_state;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;

    GdnParams params;
    uint64_t successful_runs[3];
    GdnCoreCount18ObservedTopologyV1 last_observed[3];
    bool has_staged_fixture;
    bool has_snapshot;
};

#include "metal_w8_gdn_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal GDN core error"
                                         : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal GDN core error"
                             : error.localizedDescription.UTF8String);
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

bool valid_profile(uint32_t profile) { return profile <= kFusedProfile; }

NSString *slot_function_name(uint32_t slot) {
    switch (slot) {
        case kDepthwiseSlot:
            return @"gdn_depthwise_preprocess";
        case kNormalizeSlot:
            return @"gdn_normalize_qk";
        case kLegacyRecurrentSlot:
            return @"gdn_recurrent_update";
        case kQkStagedRecurrentSlot:
            return @"gdn_recurrent_update_qk_staged_v1";
        case kNormGateSlot:
            return @"gdn_norm_gate";
        case kFusedSlot:
            return @"gdn_core_fused_v1";
        default:
            return nil;
    }
}

uint32_t slot_required_threads(uint32_t slot) {
    switch (slot) {
        case kDepthwiseSlot:
        case kLegacyRecurrentSlot:
            return kLegacyThreads;
        case kNormalizeSlot:
            return 2 * kKeyHeads;
        case kQkStagedRecurrentSlot:
        case kFusedSlot:
            return kCandidateThreads;
        case kNormGateSlot:
            return kValueHeads;
        default:
            return 0;
    }
}

uint32_t slot_source_threadgroup_bytes(uint32_t slot) {
    if (slot == kQkStagedRecurrentSlot) {
        return kQkStagedSourceThreadgroupBytes;
    }
    if (slot == kFusedSlot) {
        return kFusedSourceThreadgroupBytes;
    }
    return 0;
}

uint32_t profile_recurrent_threads(uint32_t profile) {
    return profile == kLegacyProfile ? kLegacyThreads : kCandidateThreads;
}

uint32_t profile_selected_slot(uint32_t profile) {
    if (profile == kLegacyProfile) {
        return kLegacyRecurrentSlot;
    }
    if (profile == kQkStagedProfile) {
        return kQkStagedRecurrentSlot;
    }
    return kFusedSlot;
}

uint32_t profile_dispatches(uint32_t profile) {
    return kSeamsPerRun * (profile == kFusedProfile ? 1 : 4);
}

uint32_t profile_launched_threads(uint32_t profile) {
    if (profile == kFusedProfile) {
        return kSeamsPerRun * kValueHeads * kCandidateThreads;
    }
    const uint32_t per_seam =
        kQkvWidth + 2 * kKeyHeads +
        kValueHeads * profile_recurrent_threads(profile) + kValueHeads;
    return kSeamsPerRun * per_seam;
}

uint32_t profile_threadgroups(uint32_t profile) {
    if (profile == kFusedProfile) {
        return kSeamsPerRun * kValueHeads;
    }
    const uint32_t depthwise_groups =
        (kQkvWidth + kLegacyThreads - 1) / kLegacyThreads;
    return kSeamsPerRun *
           (depthwise_groups + 1 + kValueHeads + 1);
}

uint32_t profile_source_threadgroup_bytes(uint32_t profile) {
    return slot_source_threadgroup_bytes(profile_selected_slot(profile));
}

uint32_t profile_internal_barrier_sites(uint32_t profile) {
    if (profile == kQkStagedProfile) {
        return 1;
    }
    if (profile == kFusedProfile) {
        return 4;
    }
    return 0;
}

bool live_shape_matches(
    const ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle) {
    if (handle == nullptr) {
        return false;
    }
    const GdnParams &params = handle->params;
    return params.hidden_size == kHiddenSize &&
           params.key_heads == kKeyHeads &&
           params.value_heads == kValueHeads &&
           params.value_heads == params.key_heads &&
           params.key_dim == kKeyDim && params.value_dim == kValueDim &&
           params.conv_kernel_size == kConvKernelSize &&
           params.key_width == kKeyWidth && params.value_width == kValueWidth &&
           params.qkv_width == kQkvWidth &&
           params.input_rows == kProjectedPerSeam &&
           params.input_groups_per_row == 16 &&
           params.output_groups_per_row == 32 &&
           params.rms_norm_eps == 1.0e-6f;
}

bool buffers_are_non_overlapping(
    const ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle) {
    if (handle == nullptr) {
        return false;
    }
    const id<MTLBuffer> buffers[] = {
        handle->projected,
        handle->conv_weight,
        handle->query_state,
        handle->key_state,
        handle->value_state,
        handle->a_log,
        handle->dt_bias,
        handle->recurrent_state,
        handle->norm_weight,
        handle->next_query_state,
        handle->next_key_state,
        handle->next_value_state,
        handle->processed,
        handle->next_recurrent_state,
        handle->core,
        handle->gated,
    };
    constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
    for (size_t left = 0; left < kBufferCount; ++left) {
        if (buffers[left] == nil || buffers[left].contents == nullptr) {
            return false;
        }
        const uintptr_t left_begin =
            reinterpret_cast<uintptr_t>(buffers[left].contents);
        if (buffers[left].length > UINTPTR_MAX - left_begin) {
            return false;
        }
        const uintptr_t left_end = left_begin + buffers[left].length;
        for (size_t right = left + 1; right < kBufferCount; ++right) {
            if (buffers[right] == nil || buffers[right].contents == nullptr) {
                return false;
            }
            const uintptr_t right_begin =
                reinterpret_cast<uintptr_t>(buffers[right].contents);
            if (buffers[right].length > UINTPTR_MAX - right_begin) {
                return false;
            }
            const uintptr_t right_end = right_begin + buffers[right].length;
            if (left_begin < right_end && right_begin < left_end) {
                return false;
            }
        }
    }
    return true;
}

bool live_slot_matches(
    ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle, uint32_t slot) {
    if (handle == nullptr || slot >= kPipelineSlotCount) {
        return false;
    }
    id<MTLFunction> function = handle->functions[slot];
    id<MTLComputePipelineState> pipeline = handle->pipelines[slot];
    NSString *expected = slot_function_name(slot);
    if (function == nil || pipeline == nil || expected == nil ||
        ![function.name isEqualToString:expected] ||
        pipeline.threadExecutionWidth != kRequiredThreadExecutionWidth ||
        pipeline.maxTotalThreadsPerThreadgroup < slot_required_threads(slot)) {
        return false;
    }
    const NSUInteger observed_static = pipeline.staticThreadgroupMemoryLength;
    const NSUInteger source_static = slot_source_threadgroup_bytes(slot);
    return source_static == 0 ? observed_static == 0
                              : observed_static >= source_static;
}

bool live_profile_matches(
    ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle, uint32_t profile) {
    if (handle == nullptr || !valid_profile(profile)) {
        return false;
    }
    if (profile == kFusedProfile) {
        return live_slot_matches(handle, kFusedSlot);
    }
    return live_slot_matches(handle, kDepthwiseSlot) &&
           live_slot_matches(handle, kNormalizeSlot) &&
           live_slot_matches(handle, profile_selected_slot(profile)) &&
           live_slot_matches(handle, kNormGateSlot);
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLFunction> function,
                                          NSError **error) {
    return function == nil
               ? nil
               : [device newComputePipelineStateWithFunction:function
                                                        error:error];
}

id<MTLBuffer> make_shared_f32(id<MTLDevice> device, uint32_t count) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer != nil) {
        std::memset(buffer.contents, 0, bytes);
    }
    return buffer;
}

void zero_f32(id<MTLBuffer> buffer, uint32_t count) {
    std::memset(buffer.contents, 0,
                static_cast<size_t>(count) * sizeof(float));
}

void poison_f32(id<MTLBuffer> buffer, uint32_t count) {
    const float poison = std::numeric_limits<float>::quiet_NaN();
    auto values = static_cast<float *>(buffer.contents);
    for (uint32_t index = 0; index < count; ++index) {
        values[index] = poison;
    }
}

size_t seam_offset(uint32_t seam, uint32_t elements_per_seam) {
    return static_cast<size_t>(seam) * elements_per_seam * sizeof(float);
}

void append_function_chain(
    const ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle,
    uint32_t profile, char output[kFunctionChainCapacity]) {
    std::memset(output, 0, kFunctionChainCapacity);
    if (profile == kFusedProfile) {
        std::snprintf(output, kFunctionChainCapacity, "%s",
                      handle->functions[kFusedSlot].name.UTF8String);
        return;
    }
    std::snprintf(
        output, kFunctionChainCapacity, "%s|%s|%s|%s",
        handle->functions[kDepthwiseSlot].name.UTF8String,
        handle->functions[kNormalizeSlot].name.UTF8String,
        handle->functions[profile_selected_slot(profile)].name.UTF8String,
        handle->functions[kNormGateSlot].name.UTF8String);
}

void encode_broad_buffer_barrier(
    id<MTLComputeCommandEncoder> encoder,
    GdnCoreCount18ObservedTopologyV1 *actual) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++actual->explicit_buffer_barriers;
}

void encode_four_stage_seam(
    ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle,
    id<MTLComputeCommandEncoder> encoder, uint32_t profile, uint32_t seam,
    GdnCoreCount18ObservedTopologyV1 *actual) {
    [encoder setComputePipelineState:handle->pipelines[kDepthwiseSlot]];
    [encoder setBuffer:handle->projected
                  offset:seam_offset(seam, kProjectedPerSeam)
                 atIndex:0];
    [encoder setBuffer:handle->conv_weight
                  offset:seam_offset(seam, kConvWeightPerSeam)
                 atIndex:1];
    [encoder setBuffer:handle->query_state
                  offset:seam_offset(seam, kQueryStatePerSeam)
                 atIndex:2];
    [encoder setBuffer:handle->key_state
                  offset:seam_offset(seam, kKeyStatePerSeam)
                 atIndex:3];
    [encoder setBuffer:handle->value_state
                  offset:seam_offset(seam, kValueStatePerSeam)
                 atIndex:4];
    [encoder setBuffer:handle->next_query_state
                  offset:seam_offset(seam, kQueryStatePerSeam)
                 atIndex:5];
    [encoder setBuffer:handle->next_key_state
                  offset:seam_offset(seam, kKeyStatePerSeam)
                 atIndex:6];
    [encoder setBuffer:handle->next_value_state
                  offset:seam_offset(seam, kValueStatePerSeam)
                 atIndex:7];
    [encoder setBuffer:handle->processed
                  offset:seam_offset(seam, kProcessedPerSeam)
                 atIndex:8];
    [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:9];
    [encoder dispatchThreads:MTLSizeMake(kQkvWidth, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kLegacyThreads, 1, 1)];
    ++actual->kernel_dispatches;
    actual->launched_threads += kQkvWidth;
    actual->threadgroups +=
        (kQkvWidth + kLegacyThreads - 1) / kLegacyThreads;
    encode_broad_buffer_barrier(encoder, actual);

    [encoder setComputePipelineState:handle->pipelines[kNormalizeSlot]];
    [encoder setBuffer:handle->processed
                  offset:seam_offset(seam, kProcessedPerSeam)
                 atIndex:0];
    [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(2 * kKeyHeads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(2 * kKeyHeads, 1, 1)];
    ++actual->kernel_dispatches;
    actual->launched_threads += 2 * kKeyHeads;
    ++actual->threadgroups;
    encode_broad_buffer_barrier(encoder, actual);

    const uint32_t recurrent_slot = profile_selected_slot(profile);
    const uint32_t recurrent_threads = profile_recurrent_threads(profile);
    [encoder setComputePipelineState:handle->pipelines[recurrent_slot]];
    [encoder setBuffer:handle->processed
                  offset:seam_offset(seam, kProcessedPerSeam)
                 atIndex:0];
    [encoder setBuffer:handle->projected
                  offset:seam_offset(seam, kProjectedPerSeam)
                 atIndex:1];
    [encoder setBuffer:handle->a_log
                  offset:seam_offset(seam, kHeadScalarPerSeam)
                 atIndex:2];
    [encoder setBuffer:handle->dt_bias
                  offset:seam_offset(seam, kHeadScalarPerSeam)
                 atIndex:3];
    [encoder setBuffer:handle->recurrent_state
                  offset:seam_offset(seam, kRecurrentPerSeam)
                 atIndex:4];
    [encoder setBuffer:handle->next_recurrent_state
                  offset:seam_offset(seam, kRecurrentPerSeam)
                 atIndex:5];
    [encoder setBuffer:handle->core
                  offset:seam_offset(seam, kCorePerSeam)
                 atIndex:6];
    [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(kValueHeads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(recurrent_threads, 1, 1)];
    ++actual->kernel_dispatches;
    actual->launched_threads += kValueHeads * recurrent_threads;
    actual->threadgroups += kValueHeads;
    encode_broad_buffer_barrier(encoder, actual);

    [encoder setComputePipelineState:handle->pipelines[kNormGateSlot]];
    [encoder setBuffer:handle->core
                  offset:seam_offset(seam, kCorePerSeam)
                 atIndex:0];
    [encoder setBuffer:handle->projected
                  offset:seam_offset(seam, kProjectedPerSeam)
                 atIndex:1];
    [encoder setBuffer:handle->norm_weight
                  offset:seam_offset(seam, kNormWeightPerSeam)
                 atIndex:2];
    [encoder setBuffer:handle->gated
                  offset:seam_offset(seam, kGatedPerSeam)
                 atIndex:3];
    [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(kValueHeads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kValueHeads, 1, 1)];
    ++actual->kernel_dispatches;
    actual->launched_threads += kValueHeads;
    ++actual->threadgroups;
    encode_broad_buffer_barrier(encoder, actual);
}

void encode_fused_seam(
    ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *handle,
    id<MTLComputeCommandEncoder> encoder, uint32_t seam,
    GdnCoreCount18ObservedTopologyV1 *actual) {
    [encoder setComputePipelineState:handle->pipelines[kFusedSlot]];
    // ABI 0..14 deliberately has no device processed/core intermediates.
    [encoder setBuffer:handle->projected
                  offset:seam_offset(seam, kProjectedPerSeam)
                 atIndex:0];
    [encoder setBuffer:handle->conv_weight
                  offset:seam_offset(seam, kConvWeightPerSeam)
                 atIndex:1];
    [encoder setBuffer:handle->query_state
                  offset:seam_offset(seam, kQueryStatePerSeam)
                 atIndex:2];
    [encoder setBuffer:handle->key_state
                  offset:seam_offset(seam, kKeyStatePerSeam)
                 atIndex:3];
    [encoder setBuffer:handle->value_state
                  offset:seam_offset(seam, kValueStatePerSeam)
                 atIndex:4];
    [encoder setBuffer:handle->next_query_state
                  offset:seam_offset(seam, kQueryStatePerSeam)
                 atIndex:5];
    [encoder setBuffer:handle->next_key_state
                  offset:seam_offset(seam, kKeyStatePerSeam)
                 atIndex:6];
    [encoder setBuffer:handle->next_value_state
                  offset:seam_offset(seam, kValueStatePerSeam)
                 atIndex:7];
    [encoder setBuffer:handle->a_log
                  offset:seam_offset(seam, kHeadScalarPerSeam)
                 atIndex:8];
    [encoder setBuffer:handle->dt_bias
                  offset:seam_offset(seam, kHeadScalarPerSeam)
                 atIndex:9];
    [encoder setBuffer:handle->recurrent_state
                  offset:seam_offset(seam, kRecurrentPerSeam)
                 atIndex:10];
    [encoder setBuffer:handle->next_recurrent_state
                  offset:seam_offset(seam, kRecurrentPerSeam)
                 atIndex:11];
    [encoder setBuffer:handle->norm_weight
                  offset:seam_offset(seam, kNormWeightPerSeam)
                 atIndex:12];
    [encoder setBuffer:handle->gated
                  offset:seam_offset(seam, kGatedPerSeam)
                 atIndex:13];
    [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:14];
    [encoder dispatchThreadgroups:MTLSizeMake(kValueHeads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kCandidateThreads, 1, 1)];
    ++actual->kernel_dispatches;
    actual->launched_threads += kValueHeads * kCandidateThreads;
    actual->threadgroups += kValueHeads;
    encode_broad_buffer_barrier(encoder, actual);
}

}  // namespace

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_create(
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal GDN core output handle is null");
            return 1;
        }
        *output = nullptr;
        auto handle = new (std::nothrow)
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal GDN core handle failed");
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
        NSString *source = [NSString stringWithUTF8String:kMetalGdnSource];
        id<MTLLibrary> library =
            [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        for (uint32_t slot = 0; slot < kPipelineSlotCount; ++slot) {
            handle->functions[slot] =
                [library newFunctionWithName:slot_function_name(slot)];
            handle->pipelines[slot] = make_pipeline(
                handle->device, handle->functions[slot], &error);
            if (!live_slot_matches(handle, slot)) {
                delete handle;
                if (error != nil) {
                    write_nserror(error_output, error_capacity, error);
                } else {
                    write_error(
                        error_output, error_capacity,
                        "Metal GDN core function identity or pipeline contract failed");
                }
                return 1;
            }
        }

        handle->queue = [handle->device newCommandQueue];
        handle->projected = make_shared_f32(handle->device, kProjectedCount);
        handle->conv_weight =
            make_shared_f32(handle->device, kConvWeightCount);
        handle->query_state =
            make_shared_f32(handle->device, kQueryStateCount);
        handle->key_state = make_shared_f32(handle->device, kKeyStateCount);
        handle->value_state =
            make_shared_f32(handle->device, kValueStateCount);
        handle->a_log = make_shared_f32(handle->device, kHeadScalarCount);
        handle->dt_bias = make_shared_f32(handle->device, kHeadScalarCount);
        handle->recurrent_state =
            make_shared_f32(handle->device, kRecurrentCount);
        handle->norm_weight =
            make_shared_f32(handle->device, kNormWeightCount);
        handle->next_query_state =
            make_shared_f32(handle->device, kQueryStateCount);
        handle->next_key_state =
            make_shared_f32(handle->device, kKeyStateCount);
        handle->next_value_state =
            make_shared_f32(handle->device, kValueStateCount);
        handle->processed = make_shared_f32(handle->device, kProcessedCount);
        handle->next_recurrent_state =
            make_shared_f32(handle->device, kRecurrentCount);
        handle->core = make_shared_f32(handle->device, kCoreCount);
        handle->gated = make_shared_f32(handle->device, kGatedCount);
        if (handle->queue == nil || handle->projected == nil ||
            handle->conv_weight == nil || handle->query_state == nil ||
            handle->key_state == nil || handle->value_state == nil ||
            handle->a_log == nil || handle->dt_bias == nil ||
            handle->recurrent_state == nil || handle->norm_weight == nil ||
            handle->next_query_state == nil ||
            handle->next_key_state == nil ||
            handle->next_value_state == nil || handle->processed == nil ||
            handle->next_recurrent_state == nil || handle->core == nil ||
            handle->gated == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal GDN core resources failed");
            return 1;
        }

        handle->params = GdnParams{
            kHiddenSize,
            kKeyHeads,
            kValueHeads,
            kKeyDim,
            kValueDim,
            kConvKernelSize,
            kKeyWidth,
            kValueWidth,
            kQkvWidth,
            kProjectedPerSeam,
            16,
            32,
            1.0e-6f,
        };
        if (!live_shape_matches(handle) || !buffers_are_non_overlapping(handle)) {
            delete handle;
            write_error(error_output, error_capacity,
                        "fixed-shape Metal GDN core resource contract failed");
            return 1;
        }
        handle->has_staged_fixture = false;
        handle->has_snapshot = false;
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_stage_fixture(
    void *opaque_handle, const float *projected, uint32_t projected_count,
    const float *conv_weight, uint32_t conv_weight_count,
    const float *query_state, uint32_t query_state_count,
    const float *key_state, uint32_t key_state_count,
    const float *value_state, uint32_t value_state_count, const float *a_log,
    uint32_t a_log_count, const float *dt_bias, uint32_t dt_bias_count,
    const float *recurrent_state, uint32_t recurrent_state_count,
    const float *norm_weight, uint32_t norm_weight_count, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || projected_count != kProjectedCount ||
            conv_weight_count != kConvWeightCount ||
            query_state_count != kQueryStateCount ||
            key_state_count != kKeyStateCount ||
            value_state_count != kValueStateCount ||
            a_log_count != kHeadScalarCount ||
            dt_bias_count != kHeadScalarCount ||
            recurrent_state_count != kRecurrentCount ||
            norm_weight_count != kNormWeightCount ||
            !live_shape_matches(handle) || !buffers_are_non_overlapping(handle) ||
            !finite_f32(projected, projected_count) ||
            !finite_f32(conv_weight, conv_weight_count) ||
            !finite_f32(query_state, query_state_count) ||
            !finite_f32(key_state, key_state_count) ||
            !finite_f32(value_state, value_state_count) ||
            !finite_f32(a_log, a_log_count) ||
            !finite_f32(dt_bias, dt_bias_count) ||
            !finite_f32(recurrent_state, recurrent_state_count) ||
            !finite_f32(norm_weight, norm_weight_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core staged fixture");
            return 1;
        }

        std::memcpy(handle->projected.contents, projected,
                    static_cast<size_t>(projected_count) * sizeof(float));
        std::memcpy(handle->conv_weight.contents, conv_weight,
                    static_cast<size_t>(conv_weight_count) * sizeof(float));
        std::memcpy(handle->query_state.contents, query_state,
                    static_cast<size_t>(query_state_count) * sizeof(float));
        std::memcpy(handle->key_state.contents, key_state,
                    static_cast<size_t>(key_state_count) * sizeof(float));
        std::memcpy(handle->value_state.contents, value_state,
                    static_cast<size_t>(value_state_count) * sizeof(float));
        std::memcpy(handle->a_log.contents, a_log,
                    static_cast<size_t>(a_log_count) * sizeof(float));
        std::memcpy(handle->dt_bias.contents, dt_bias,
                    static_cast<size_t>(dt_bias_count) * sizeof(float));
        std::memcpy(handle->recurrent_state.contents, recurrent_state,
                    static_cast<size_t>(recurrent_state_count) * sizeof(float));
        std::memcpy(handle->norm_weight.contents, norm_weight,
                    static_cast<size_t>(norm_weight_count) * sizeof(float));

        // Stage establishes a deterministic baseline. processed/core are
        // legacy-only intermediates and remain zero for the fused profile.
        zero_f32(handle->next_query_state, kQueryStateCount);
        zero_f32(handle->next_key_state, kKeyStateCount);
        zero_f32(handle->next_value_state, kValueStateCount);
        zero_f32(handle->processed, kProcessedCount);
        zero_f32(handle->next_recurrent_state, kRecurrentCount);
        zero_f32(handle->core, kCoreCount);
        zero_f32(handle->gated, kGatedCount);
        handle->has_staged_fixture = true;
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int
apxinf_metal_gdn_core_fused_count18_profile_v1_verify_fixture_unchanged(
    void *opaque_handle, const float *projected, uint32_t projected_count,
    const float *conv_weight, uint32_t conv_weight_count,
    const float *query_state, uint32_t query_state_count,
    const float *key_state, uint32_t key_state_count,
    const float *value_state, uint32_t value_state_count, const float *a_log,
    uint32_t a_log_count, const float *dt_bias, uint32_t dt_bias_count,
    const float *recurrent_state, uint32_t recurrent_state_count,
    const float *norm_weight, uint32_t norm_weight_count, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
        const bool valid =
            handle != nullptr && handle->has_staged_fixture &&
            projected != nullptr && projected_count == kProjectedCount &&
            conv_weight != nullptr && conv_weight_count == kConvWeightCount &&
            query_state != nullptr && query_state_count == kQueryStateCount &&
            key_state != nullptr && key_state_count == kKeyStateCount &&
            value_state != nullptr && value_state_count == kValueStateCount &&
            a_log != nullptr && a_log_count == kHeadScalarCount &&
            dt_bias != nullptr && dt_bias_count == kHeadScalarCount &&
            recurrent_state != nullptr &&
            recurrent_state_count == kRecurrentCount &&
            norm_weight != nullptr && norm_weight_count == kNormWeightCount &&
            live_shape_matches(handle) && buffers_are_non_overlapping(handle) &&
            std::memcmp(handle->projected.contents, projected,
                        static_cast<size_t>(projected_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->conv_weight.contents, conv_weight,
                        static_cast<size_t>(conv_weight_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->query_state.contents, query_state,
                        static_cast<size_t>(query_state_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->key_state.contents, key_state,
                        static_cast<size_t>(key_state_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->value_state.contents, value_state,
                        static_cast<size_t>(value_state_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->a_log.contents, a_log,
                        static_cast<size_t>(a_log_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->dt_bias.contents, dt_bias,
                        static_cast<size_t>(dt_bias_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->recurrent_state.contents, recurrent_state,
                        static_cast<size_t>(recurrent_state_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->norm_weight.contents, norm_weight,
                        static_cast<size_t>(norm_weight_count) * sizeof(float)) == 0;
        if (!valid) {
            write_error(error_output, error_capacity,
                        "staged Metal GDN core fixture changed");
            return 1;
        }
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_poison_outputs(
    void *opaque_handle, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !handle->has_staged_fixture ||
            !live_shape_matches(handle) || !buffers_are_non_overlapping(handle)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core poison contract");
            return 1;
        }
        // Only externally compared outputs are poisoned. processed/core remain
        // fixture-stage zeros until an A/B run writes them.
        poison_f32(handle->next_query_state, kQueryStateCount);
        poison_f32(handle->next_key_state, kKeyStateCount);
        poison_f32(handle->next_value_state, kValueStateCount);
        poison_f32(handle->next_recurrent_state, kRecurrentCount);
        poison_f32(handle->gated, kGatedCount);
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_run(
    void *opaque_handle, uint32_t profile, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);

        // Validate the selector before clearing snapshot state or touching an
        // encoder. An invalid selector therefore cannot mutate bridge state.
        if (handle == nullptr || !valid_profile(profile)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core profile selector");
            return 1;
        }
        if (!handle->has_staged_fixture || !live_shape_matches(handle) ||
            !buffers_are_non_overlapping(handle) ||
            !live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core run contract");
            return 1;
        }

        handle->has_snapshot = false;
        GdnCoreCount18ObservedTopologyV1 actual{};
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal GDN core command buffer failed");
            return 1;
        }
        ++actual.command_buffers;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal GDN core compute encoder failed");
            return 1;
        }
        ++actual.compute_encoders;

        for (uint32_t seam = 0; seam < kSeamsPerRun; ++seam) {
            if (profile == kFusedProfile) {
                encode_fused_seam(handle, encoder, seam, &actual);
            } else {
                encode_four_stage_seam(handle, encoder, profile, seam, &actual);
            }
        }
        [encoder endEncoding];

        const uint32_t expected_dispatches = profile_dispatches(profile);
        const uint32_t expected_launched = profile_launched_threads(profile);
        const uint32_t expected_threadgroups = profile_threadgroups(profile);
        if (actual.command_buffers != 1 || actual.compute_encoders != 1 ||
            actual.kernel_dispatches != expected_dispatches ||
            actual.explicit_buffer_barriers != expected_dispatches ||
            actual.launched_threads != expected_launched ||
            actual.threadgroups != expected_threadgroups) {
            write_error(error_output, error_capacity,
                        "Metal GDN core pre-commit topology mismatch");
            return 1;
        }

        [command commit];
        ++actual.commits;
        [command waitUntilCompleted];
        ++actual.waits;
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        if (actual.commits != 1 || actual.waits != 1) {
            write_error(error_output, error_capacity,
                        "Metal GDN core completion topology mismatch");
            return 1;
        }

        ++handle->successful_runs[profile];
        handle->last_observed[profile] = actual;
        handle->has_snapshot = true;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_snapshot(
    void *opaque_handle, float *next_query_state_output,
    uint32_t next_query_state_count, float *next_key_state_output,
    uint32_t next_key_state_count, float *next_value_state_output,
    uint32_t next_value_state_count, float *next_recurrent_state_output,
    uint32_t next_recurrent_state_count, float *gated_output,
    uint32_t gated_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !handle->has_snapshot ||
            next_query_state_output == nullptr ||
            next_query_state_count != kQueryStateCount ||
            next_key_state_output == nullptr ||
            next_key_state_count != kKeyStateCount ||
            next_value_state_output == nullptr ||
            next_value_state_count != kValueStateCount ||
            next_recurrent_state_output == nullptr ||
            next_recurrent_state_count != kRecurrentCount ||
            gated_output == nullptr || gated_count != kGatedCount) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core snapshot contract");
            return 1;
        }
        std::memcpy(next_query_state_output, handle->next_query_state.contents,
                    static_cast<size_t>(next_query_state_count) * sizeof(float));
        std::memcpy(next_key_state_output, handle->next_key_state.contents,
                    static_cast<size_t>(next_key_state_count) * sizeof(float));
        std::memcpy(next_value_state_output, handle->next_value_state.contents,
                    static_cast<size_t>(next_value_state_count) * sizeof(float));
        std::memcpy(next_recurrent_state_output,
                    handle->next_recurrent_state.contents,
                    static_cast<size_t>(next_recurrent_state_count) * sizeof(float));
        std::memcpy(gated_output, handle->gated.contents,
                    static_cast<size_t>(gated_count) * sizeof(float));
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_core_fused_count18_profile_v1_receipt(
    void *opaque_handle, uint32_t profile,
    GdnCoreCount18RuntimeReceiptV1 *receipt, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr || !valid_profile(profile) ||
            !live_shape_matches(handle) || !buffers_are_non_overlapping(handle) ||
            !live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN core receipt contract");
            return 1;
        }

        std::memset(receipt, 0, sizeof(*receipt));
        receipt->requested_profile = profile;
        receipt->observed_profile = profile;
        receipt->seams_per_run = kSeamsPerRun;
        receipt->kernel_dispatches_per_run = profile_dispatches(profile);
        receipt->explicit_buffer_barriers_per_run = profile_dispatches(profile);
        receipt->launched_threads_per_run = profile_launched_threads(profile);
        receipt->threadgroups_per_run = profile_threadgroups(profile);
        // For profile C this legacy-named field records the fused core width.
        receipt->recurrent_threads_per_threadgroup =
            profile_recurrent_threads(profile);
        id<MTLComputePipelineState> selected_pipeline =
            handle->pipelines[profile_selected_slot(profile)];
        receipt->pipeline_thread_execution_width =
            static_cast<uint32_t>(selected_pipeline.threadExecutionWidth);
        receipt->pipeline_static_threadgroup_memory_bytes =
            static_cast<uint32_t>(
                selected_pipeline.staticThreadgroupMemoryLength);
        receipt->source_declared_threadgroup_memory_bytes =
            profile_source_threadgroup_bytes(profile);
        receipt->internal_threadgroup_barrier_sites_per_threadgroup =
            profile_internal_barrier_sites(profile);
        receipt->fixed_shape_host_validated = 1;
        receipt->input_output_buffers_non_overlapping = 1;
        receipt->command_buffers_per_run = 1;
        receipt->compute_encoders_per_run = 1;
        receipt->commits_per_run = 1;
        receipt->waits_per_run = 1;

        const auto &last = handle->last_observed[profile];
        receipt->last_observed_kernel_dispatches = last.kernel_dispatches;
        receipt->last_observed_explicit_buffer_barriers =
            last.explicit_buffer_barriers;
        receipt->last_observed_launched_threads = last.launched_threads;
        receipt->last_observed_threadgroups = last.threadgroups;
        receipt->last_observed_command_buffers = last.command_buffers;
        receipt->last_observed_compute_encoders = last.compute_encoders;
        receipt->last_observed_commits = last.commits;
        receipt->last_observed_waits = last.waits;
        receipt->successful_runs = handle->successful_runs[profile];
        append_function_chain(handle, profile, receipt->observed_function_chain);
        return 0;
    }
}

extern "C" void apxinf_metal_gdn_core_fused_count18_profile_v1_destroy(
    void *opaque_handle) {
    auto handle = static_cast<
        ApxinfMetalGdnCoreFusedCount18ProfileV1Handle *>(opaque_handle);
    delete handle;
}
