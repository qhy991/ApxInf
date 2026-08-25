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
constexpr uint32_t kProjectedPerSeam = kQkvWidth + kValueWidth + 2 * kValueHeads;
constexpr uint32_t kRecurrentPerSeam = kValueHeads * kKeyDim * kValueDim;
constexpr uint32_t kCorePerSeam = kValueHeads * kValueDim;
constexpr uint32_t kProcessedCount = kSeamsPerRun * kQkvWidth;
constexpr uint32_t kProjectedCount = kSeamsPerRun * kProjectedPerSeam;
constexpr uint32_t kHeadScalarCount = kSeamsPerRun * kValueHeads;
constexpr uint32_t kRecurrentCount = kSeamsPerRun * kRecurrentPerSeam;
constexpr uint32_t kCoreCount = kSeamsPerRun * kCorePerSeam;
constexpr uint32_t kRequiredThreadExecutionWidth = 32;
constexpr uint32_t kLegacyProfile = 0;
constexpr uint32_t kLeaderBroadcastProfile = 1;
constexpr uint32_t kQkStagedProfile = 2;
constexpr uint32_t kLegacyThreads = 256;
constexpr uint32_t kCandidateThreads = 128;
constexpr uint32_t kLeaderSourceThreadgroupBytes = 2 * sizeof(float);
constexpr uint32_t kQkStagedSourceThreadgroupBytes =
    (2 + kKeyDim + kKeyDim) * sizeof(float);
constexpr size_t kFunctionNameCapacity = 64;

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

struct GdnRecurrentObservedTopologyV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t threadgroups;
    uint32_t explicit_buffer_barriers;
    uint32_t launched_threads;
    uint32_t active_value_threads;
    uint32_t idle_threads;
    uint32_t commits;
    uint32_t waits;
};

struct GdnRecurrentCount18RuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t seams_per_run;
    uint32_t key_heads;
    uint32_t value_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t processed_elements_per_seam;
    uint32_t projected_elements_per_seam;
    uint32_t recurrent_elements_per_seam;
    uint32_t core_elements_per_seam;
    uint32_t threads_per_threadgroup;
    uint32_t simdgroups_per_threadgroup;
    uint32_t pipeline_max_total_threads_per_threadgroup;
    uint32_t pipeline_thread_execution_width;
    uint32_t pipeline_static_threadgroup_memory_bytes;
    uint32_t source_declared_threadgroup_memory_bytes;
    uint32_t dynamic_threadgroup_memory_bytes;
    uint32_t internal_threadgroup_barrier_sites_per_threadgroup;
    uint32_t source_derived_internal_barrier_executions_per_run;
    uint32_t launched_threads_per_run;
    uint32_t active_value_threads_per_run;
    uint32_t idle_threads_per_run;
    uint32_t command_buffers_per_run;
    uint32_t compute_encoders_per_run;
    uint32_t kernel_dispatches_per_run;
    uint32_t threadgroups_per_run;
    uint32_t explicit_buffer_barriers_per_run;
    uint32_t commits_per_run;
    uint32_t waits_per_run;
    uint32_t fixed_shape_host_validated;
    uint32_t input_output_buffers_non_overlapping;
    uint64_t host_to_device_bytes_per_run;
    uint64_t device_to_host_bytes_per_run;
    uint64_t processed_buffer_bytes;
    uint64_t projected_buffer_bytes;
    uint64_t a_log_buffer_bytes;
    uint64_t dt_bias_buffer_bytes;
    uint64_t state_buffer_bytes;
    uint64_t next_state_buffer_bytes;
    uint64_t core_buffer_bytes;
    uint64_t persistent_buffer_bytes_total;
    uint64_t successful_runs;
    uint32_t last_observed_command_buffers;
    uint32_t last_observed_compute_encoders;
    uint32_t last_observed_kernel_dispatches;
    uint32_t last_observed_threadgroups;
    uint32_t last_observed_explicit_buffer_barriers;
    uint32_t last_observed_launched_threads;
    uint32_t last_observed_active_value_threads;
    uint32_t last_observed_idle_threads;
    uint32_t last_observed_commits;
    uint32_t last_observed_waits;
    char requested_function_name[kFunctionNameCapacity];
    char observed_function_name[kFunctionNameCapacity];
};

static_assert(sizeof(GdnParams) == 52, "Metal GDN parameter ABI changed");
static_assert(sizeof(GdnRecurrentObservedTopologyV1) == 40,
              "Metal GDN recurrent observed-topology ABI changed");
static_assert(sizeof(GdnRecurrentCount18RuntimeReceiptV1) == 384,
              "Metal GDN recurrent runtime-receipt ABI changed");

struct ApxinfMetalGdnRecurrentCount18ProfileV1Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFunction> functions[3];
    id<MTLComputePipelineState> pipelines[3];
    id<MTLBuffer> processed;
    id<MTLBuffer> projected;
    id<MTLBuffer> a_log;
    id<MTLBuffer> dt_bias;
    id<MTLBuffer> state;
    id<MTLBuffer> next_state;
    id<MTLBuffer> core;
    GdnParams params;
    uint64_t successful_runs[3];
    GdnRecurrentObservedTopologyV1 last_observed[3];
    bool has_staged_fixture;
    bool has_snapshot;
};

#include "metal_w8_gdn_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal GDN recurrent error"
                                         : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal GDN recurrent error"
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

bool valid_profile(uint32_t profile) { return profile <= kQkStagedProfile; }

bool live_shape_matches(
    const ApxinfMetalGdnRecurrentCount18ProfileV1Handle *handle) {
    if (handle == nullptr) {
        return false;
    }
    const GdnParams &params = handle->params;
    return params.hidden_size == kHiddenSize &&
           params.key_heads == kKeyHeads &&
           params.value_heads == kValueHeads && params.key_dim == kKeyDim &&
           params.value_dim == kValueDim &&
           params.conv_kernel_size == kConvKernelSize &&
           params.key_width == kKeyWidth && params.value_width == kValueWidth &&
           params.qkv_width == kQkvWidth &&
           params.input_rows == kProjectedPerSeam &&
           params.input_groups_per_row == kHiddenSize / 64 &&
           params.output_groups_per_row == kValueWidth / 64 &&
           params.rms_norm_eps == 1.0e-6f;
}

bool buffers_are_non_overlapping(
    const ApxinfMetalGdnRecurrentCount18ProfileV1Handle *handle) {
    if (handle == nullptr) {
        return false;
    }
    const id<MTLBuffer> buffers[] = {
        handle->processed, handle->projected, handle->a_log, handle->dt_bias,
        handle->state,     handle->next_state, handle->core,
    };
    constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
    for (size_t left = 0; left < kBufferCount; ++left) {
        if (buffers[left] == nil || buffers[left].contents == nullptr) {
            return false;
        }
        const uintptr_t left_begin =
            reinterpret_cast<uintptr_t>(buffers[left].contents);
        const uintptr_t left_end = left_begin + buffers[left].length;
        if (left_end < left_begin) {
            return false;
        }
        for (size_t right = left + 1; right < kBufferCount; ++right) {
            if (buffers[right] == nil || buffers[right].contents == nullptr) {
                return false;
            }
            const uintptr_t right_begin =
                reinterpret_cast<uintptr_t>(buffers[right].contents);
            const uintptr_t right_end = right_begin + buffers[right].length;
            if (right_end < right_begin ||
                (left_begin < right_end && right_begin < left_end)) {
                return false;
            }
        }
    }
    return true;
}

uint32_t profile_threads(uint32_t profile) {
    return profile == kLegacyProfile ? kLegacyThreads : kCandidateThreads;
}

uint32_t profile_source_threadgroup_bytes(uint32_t profile) {
    if (profile == kLeaderBroadcastProfile) {
        return kLeaderSourceThreadgroupBytes;
    }
    if (profile == kQkStagedProfile) {
        return kQkStagedSourceThreadgroupBytes;
    }
    return 0;
}

uint32_t profile_internal_barrier_sites(uint32_t profile) {
    return profile == kLegacyProfile ? 0 : 1;
}

NSString *profile_function_name(uint32_t profile) {
    switch (profile) {
        case kLegacyProfile:
            return @"gdn_recurrent_update";
        case kLeaderBroadcastProfile:
            return @"gdn_recurrent_update_leader_broadcast_v1";
        case kQkStagedProfile:
            return @"gdn_recurrent_update_qk_staged_v1";
        default:
            return nil;
    }
}

void copy_function_name(char output[kFunctionNameCapacity], NSString *name) {
    std::memset(output, 0, kFunctionNameCapacity);
    if (name != nil) {
        std::snprintf(output, kFunctionNameCapacity, "%s", name.UTF8String);
    }
}

bool live_profile_matches(
    ApxinfMetalGdnRecurrentCount18ProfileV1Handle *handle, uint32_t profile) {
    if (handle == nullptr || !valid_profile(profile)) {
        return false;
    }
    id<MTLFunction> function = handle->functions[profile];
    id<MTLComputePipelineState> pipeline = handle->pipelines[profile];
    NSString *expected = profile_function_name(profile);
    if (function == nil || pipeline == nil || expected == nil ||
        ![function.name isEqualToString:expected] ||
        pipeline.threadExecutionWidth != kRequiredThreadExecutionWidth ||
        pipeline.maxTotalThreadsPerThreadgroup < profile_threads(profile)) {
        return false;
    }
    const NSUInteger observed_static = pipeline.staticThreadgroupMemoryLength;
    const NSUInteger source_static = profile_source_threadgroup_bytes(profile);
    return profile == kLegacyProfile ? observed_static == 0
                                     : observed_static >= source_static;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLFunction> function,
                                          NSError **error) {
    return function == nil
               ? nil
               : [device newComputePipelineStateWithFunction:function error:error];
}

id<MTLBuffer> make_shared_f32(id<MTLDevice> device, uint32_t count) {
    return [device newBufferWithLength:static_cast<size_t>(count) * sizeof(float)
                               options:MTLResourceStorageModeShared];
}

}  // namespace

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_create(
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal GDN recurrent output handle is null");
            return 1;
        }
        *output = nullptr;
        auto handle =
            new (std::nothrow) ApxinfMetalGdnRecurrentCount18ProfileV1Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal GDN recurrent handle failed");
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
        for (uint32_t profile = kLegacyProfile;
             profile <= kQkStagedProfile; ++profile) {
            handle->functions[profile] =
                [library newFunctionWithName:profile_function_name(profile)];
            handle->pipelines[profile] = make_pipeline(
                handle->device, handle->functions[profile], &error);
            if (!live_profile_matches(handle, profile)) {
                delete handle;
                if (error != nil) {
                    write_nserror(error_output, error_capacity, error);
                } else {
                    write_error(
                        error_output, error_capacity,
                        "Metal GDN recurrent function identity or pipeline contract failed");
                }
                return 1;
            }
        }

        handle->queue = [handle->device newCommandQueue];
        handle->processed = make_shared_f32(handle->device, kProcessedCount);
        handle->projected = make_shared_f32(handle->device, kProjectedCount);
        handle->a_log = make_shared_f32(handle->device, kHeadScalarCount);
        handle->dt_bias = make_shared_f32(handle->device, kHeadScalarCount);
        handle->state = make_shared_f32(handle->device, kRecurrentCount);
        handle->next_state = make_shared_f32(handle->device, kRecurrentCount);
        handle->core = make_shared_f32(handle->device, kCoreCount);
        if (handle->queue == nil || handle->processed == nil ||
            handle->projected == nil || handle->a_log == nil ||
            handle->dt_bias == nil || handle->state == nil ||
            handle->next_state == nil || handle->core == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal GDN recurrent resources failed");
            return 1;
        }
        std::memset(handle->processed.contents, 0,
                    static_cast<size_t>(kProcessedCount) * sizeof(float));
        std::memset(handle->projected.contents, 0,
                    static_cast<size_t>(kProjectedCount) * sizeof(float));
        std::memset(handle->a_log.contents, 0,
                    static_cast<size_t>(kHeadScalarCount) * sizeof(float));
        std::memset(handle->dt_bias.contents, 0,
                    static_cast<size_t>(kHeadScalarCount) * sizeof(float));
        std::memset(handle->state.contents, 0,
                    static_cast<size_t>(kRecurrentCount) * sizeof(float));
        std::memset(handle->next_state.contents, 0,
                    static_cast<size_t>(kRecurrentCount) * sizeof(float));
        std::memset(handle->core.contents, 0,
                    static_cast<size_t>(kCoreCount) * sizeof(float));
        handle->params = GdnParams{
            kHiddenSize, kKeyHeads, kValueHeads, kKeyDim, kValueDim,
            kConvKernelSize, kKeyWidth, kValueWidth, kQkvWidth,
            kProjectedPerSeam, kHiddenSize / 64, kValueWidth / 64, 1.0e-6f};
        if (!live_shape_matches(handle) || !buffers_are_non_overlapping(handle)) {
            delete handle;
            write_error(error_output, error_capacity,
                        "fixed-shape Metal GDN recurrent resource contract failed");
            return 1;
        }
        handle->has_staged_fixture = false;
        handle->has_snapshot = false;
        *output = handle;
        return 0;
    }
}

extern "C" int
apxinf_metal_gdn_recurrent_count18_profile_v1_verify_fixture_unchanged(
    void *opaque_handle, const float *processed, uint32_t processed_count,
    const float *projected, uint32_t projected_count, const float *a_log,
    uint32_t a_log_count, const float *dt_bias, uint32_t dt_bias_count,
    const float *state, uint32_t state_count, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        const bool valid =
            handle != nullptr && handle->has_staged_fixture &&
            processed != nullptr && processed_count == kProcessedCount &&
            projected != nullptr && projected_count == kProjectedCount &&
            a_log != nullptr && a_log_count == kHeadScalarCount &&
            dt_bias != nullptr && dt_bias_count == kHeadScalarCount &&
            state != nullptr && state_count == kRecurrentCount &&
            std::memcmp(handle->processed.contents, processed,
                        static_cast<size_t>(processed_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->projected.contents, projected,
                        static_cast<size_t>(projected_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->a_log.contents, a_log,
                        static_cast<size_t>(a_log_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->dt_bias.contents, dt_bias,
                        static_cast<size_t>(dt_bias_count) * sizeof(float)) == 0 &&
            std::memcmp(handle->state.contents, state,
                        static_cast<size_t>(state_count) * sizeof(float)) == 0;
        if (!valid) {
            write_error(error_output, error_capacity,
                        "staged Metal GDN recurrent fixture changed");
            return 1;
        }
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_stage_fixture(
    void *opaque_handle, const float *processed, uint32_t processed_count,
    const float *projected, uint32_t projected_count, const float *a_log,
    uint32_t a_log_count, const float *dt_bias, uint32_t dt_bias_count,
    const float *state, uint32_t state_count, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || processed_count != kProcessedCount ||
            projected_count != kProjectedCount ||
            a_log_count != kHeadScalarCount ||
            dt_bias_count != kHeadScalarCount ||
            state_count != kRecurrentCount ||
            !finite_f32(processed, processed_count) ||
            !finite_f32(projected, projected_count) ||
            !finite_f32(a_log, a_log_count) ||
            !finite_f32(dt_bias, dt_bias_count) ||
            !finite_f32(state, state_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN recurrent staged fixture");
            return 1;
        }
        std::memcpy(handle->processed.contents, processed,
                    static_cast<size_t>(processed_count) * sizeof(float));
        std::memcpy(handle->projected.contents, projected,
                    static_cast<size_t>(projected_count) * sizeof(float));
        std::memcpy(handle->a_log.contents, a_log,
                    static_cast<size_t>(a_log_count) * sizeof(float));
        std::memcpy(handle->dt_bias.contents, dt_bias,
                    static_cast<size_t>(dt_bias_count) * sizeof(float));
        std::memcpy(handle->state.contents, state,
                    static_cast<size_t>(state_count) * sizeof(float));
        handle->has_staged_fixture = true;
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_poison_outputs(
    void *opaque_handle, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !handle->has_staged_fixture) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN recurrent poison contract");
            return 1;
        }
        const float poison = std::numeric_limits<float>::quiet_NaN();
        auto *next_state = static_cast<float *>(handle->next_state.contents);
        auto *core = static_cast<float *>(handle->core.contents);
        for (uint32_t index = 0; index < kRecurrentCount; ++index) {
            next_state[index] = poison;
        }
        for (uint32_t index = 0; index < kCoreCount; ++index) {
            core[index] = poison;
        }
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_run(
    void *opaque_handle, uint32_t profile, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !valid_profile(profile) ||
            !handle->has_staged_fixture || !live_shape_matches(handle) ||
            !buffers_are_non_overlapping(handle)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN recurrent run contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal GDN recurrent function or pipeline identity changed");
            return 1;
        }

        handle->has_snapshot = false;
        GdnRecurrentObservedTopologyV1 actual{};
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal GDN recurrent command buffer failed");
            return 1;
        }
        ++actual.command_buffers;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal GDN recurrent compute encoder failed");
            return 1;
        }
        ++actual.compute_encoders;
        [encoder setComputePipelineState:handle->pipelines[profile]];
        const size_t f32_bytes = sizeof(float);
        for (uint32_t seam = 0; seam < kSeamsPerRun; ++seam) {
            [encoder setBuffer:handle->processed
                          offset:static_cast<size_t>(seam) * kQkvWidth * f32_bytes
                         atIndex:0];
            [encoder setBuffer:handle->projected
                          offset:static_cast<size_t>(seam) * kProjectedPerSeam * f32_bytes
                         atIndex:1];
            [encoder setBuffer:handle->a_log
                          offset:static_cast<size_t>(seam) * kValueHeads * f32_bytes
                         atIndex:2];
            [encoder setBuffer:handle->dt_bias
                          offset:static_cast<size_t>(seam) * kValueHeads * f32_bytes
                         atIndex:3];
            [encoder setBuffer:handle->state
                          offset:static_cast<size_t>(seam) * kRecurrentPerSeam * f32_bytes
                         atIndex:4];
            [encoder setBuffer:handle->next_state
                          offset:static_cast<size_t>(seam) * kRecurrentPerSeam * f32_bytes
                         atIndex:5];
            [encoder setBuffer:handle->core
                          offset:static_cast<size_t>(seam) * kCorePerSeam * f32_bytes
                         atIndex:6];
            [encoder setBytes:&handle->params
                       length:sizeof(handle->params)
                      atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(kValueHeads, 1, 1)
                     threadsPerThreadgroup:
                         MTLSizeMake(profile_threads(profile), 1, 1)];
            ++actual.kernel_dispatches;
            actual.threadgroups += kValueHeads;
            actual.launched_threads += kValueHeads * profile_threads(profile);
            actual.active_value_threads += kValueHeads * kValueDim;
            actual.idle_threads +=
                kValueHeads * (profile_threads(profile) - kValueDim);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ++actual.explicit_buffer_barriers;
        }
        [encoder endEncoding];
        if (actual.command_buffers != 1 || actual.compute_encoders != 1 ||
            actual.kernel_dispatches != kSeamsPerRun ||
            actual.threadgroups != kSeamsPerRun * kValueHeads ||
            actual.explicit_buffer_barriers != kSeamsPerRun ||
            actual.launched_threads !=
                kSeamsPerRun * kValueHeads * profile_threads(profile) ||
            actual.active_value_threads !=
                kSeamsPerRun * kValueHeads * kValueDim ||
            actual.idle_threads !=
                kSeamsPerRun * kValueHeads *
                    (profile_threads(profile) - kValueDim)) {
            write_error(error_output, error_capacity,
                        "Metal GDN recurrent pre-commit topology mismatch");
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
                        "Metal GDN recurrent completion topology mismatch");
            return 1;
        }
        ++handle->successful_runs[profile];
        handle->last_observed[profile] = actual;
        handle->has_snapshot = true;
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_snapshot(
    void *opaque_handle, float *next_state_output,
    uint32_t next_state_count, float *core_output, uint32_t core_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || next_state_output == nullptr ||
            core_output == nullptr || next_state_count != kRecurrentCount ||
            core_count != kCoreCount || !handle->has_snapshot) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN recurrent snapshot contract");
            return 1;
        }
        std::memcpy(next_state_output, handle->next_state.contents,
                    static_cast<size_t>(next_state_count) * sizeof(float));
        std::memcpy(core_output, handle->core.contents,
                    static_cast<size_t>(core_count) * sizeof(float));
        return 0;
    }
}

extern "C" int apxinf_metal_gdn_recurrent_count18_profile_v1_receipt(
    void *opaque_handle, uint32_t profile,
    GdnRecurrentCount18RuntimeReceiptV1 *receipt, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr || !valid_profile(profile) ||
            !live_shape_matches(handle) || !buffers_are_non_overlapping(handle)) {
            write_error(error_output, error_capacity,
                        "invalid Metal GDN recurrent receipt contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal GDN recurrent receipt identity changed");
            return 1;
        }
        std::memset(receipt, 0, sizeof(*receipt));
        receipt->requested_profile = profile;
        receipt->observed_profile = profile;
        receipt->seams_per_run = kSeamsPerRun;
        receipt->key_heads = kKeyHeads;
        receipt->value_heads = kValueHeads;
        receipt->key_dim = kKeyDim;
        receipt->value_dim = kValueDim;
        receipt->processed_elements_per_seam = kQkvWidth;
        receipt->projected_elements_per_seam = kProjectedPerSeam;
        receipt->recurrent_elements_per_seam = kRecurrentPerSeam;
        receipt->core_elements_per_seam = kCorePerSeam;
        receipt->threads_per_threadgroup = profile_threads(profile);
        receipt->simdgroups_per_threadgroup =
            profile_threads(profile) / kRequiredThreadExecutionWidth;
        id<MTLComputePipelineState> pipeline = handle->pipelines[profile];
        receipt->pipeline_max_total_threads_per_threadgroup =
            static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
        receipt->pipeline_thread_execution_width =
            static_cast<uint32_t>(pipeline.threadExecutionWidth);
        receipt->pipeline_static_threadgroup_memory_bytes =
            static_cast<uint32_t>(pipeline.staticThreadgroupMemoryLength);
        receipt->source_declared_threadgroup_memory_bytes =
            profile_source_threadgroup_bytes(profile);
        receipt->dynamic_threadgroup_memory_bytes = 0;
        receipt->internal_threadgroup_barrier_sites_per_threadgroup =
            profile_internal_barrier_sites(profile);
        receipt->source_derived_internal_barrier_executions_per_run =
            profile_internal_barrier_sites(profile) * kSeamsPerRun * kValueHeads;
        receipt->launched_threads_per_run =
            kSeamsPerRun * kValueHeads * profile_threads(profile);
        receipt->active_value_threads_per_run =
            kSeamsPerRun * kValueHeads * kValueDim;
        receipt->idle_threads_per_run =
            kSeamsPerRun * kValueHeads *
            (profile_threads(profile) - kValueDim);
        receipt->command_buffers_per_run = 1;
        receipt->compute_encoders_per_run = 1;
        receipt->kernel_dispatches_per_run = kSeamsPerRun;
        receipt->threadgroups_per_run = kSeamsPerRun * kValueHeads;
        receipt->explicit_buffer_barriers_per_run = kSeamsPerRun;
        receipt->commits_per_run = 1;
        receipt->waits_per_run = 1;
        receipt->fixed_shape_host_validated = 1;
        receipt->input_output_buffers_non_overlapping = 1;
        receipt->host_to_device_bytes_per_run = 0;
        receipt->device_to_host_bytes_per_run = 0;
        receipt->processed_buffer_bytes =
            static_cast<uint64_t>(kProcessedCount) * sizeof(float);
        receipt->projected_buffer_bytes =
            static_cast<uint64_t>(kProjectedCount) * sizeof(float);
        receipt->a_log_buffer_bytes =
            static_cast<uint64_t>(kHeadScalarCount) * sizeof(float);
        receipt->dt_bias_buffer_bytes =
            static_cast<uint64_t>(kHeadScalarCount) * sizeof(float);
        receipt->state_buffer_bytes =
            static_cast<uint64_t>(kRecurrentCount) * sizeof(float);
        receipt->next_state_buffer_bytes =
            static_cast<uint64_t>(kRecurrentCount) * sizeof(float);
        receipt->core_buffer_bytes =
            static_cast<uint64_t>(kCoreCount) * sizeof(float);
        receipt->persistent_buffer_bytes_total =
            receipt->processed_buffer_bytes + receipt->projected_buffer_bytes +
            receipt->a_log_buffer_bytes + receipt->dt_bias_buffer_bytes +
            receipt->state_buffer_bytes + receipt->next_state_buffer_bytes +
            receipt->core_buffer_bytes;
        receipt->successful_runs = handle->successful_runs[profile];
        const auto &last = handle->last_observed[profile];
        receipt->last_observed_command_buffers = last.command_buffers;
        receipt->last_observed_compute_encoders = last.compute_encoders;
        receipt->last_observed_kernel_dispatches = last.kernel_dispatches;
        receipt->last_observed_threadgroups = last.threadgroups;
        receipt->last_observed_explicit_buffer_barriers =
            last.explicit_buffer_barriers;
        receipt->last_observed_launched_threads = last.launched_threads;
        receipt->last_observed_active_value_threads =
            last.active_value_threads;
        receipt->last_observed_idle_threads = last.idle_threads;
        receipt->last_observed_commits = last.commits;
        receipt->last_observed_waits = last.waits;
        copy_function_name(receipt->requested_function_name,
                           profile_function_name(profile));
        copy_function_name(receipt->observed_function_name,
                           handle->functions[profile].name);
        return 0;
    }
}

extern "C" void apxinf_metal_gdn_recurrent_count18_profile_v1_destroy(
    void *opaque_handle) {
    auto handle = static_cast<
        ApxinfMetalGdnRecurrentCount18ProfileV1Handle *>(opaque_handle);
    delete handle;
}
