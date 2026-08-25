#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kRmsCallsPerRun = 43;
constexpr uint32_t kThreadsPerThreadgroup = 256;
constexpr uint32_t kRequiredThreadExecutionWidth = 32;
constexpr uint32_t kProfileCount = 2;
constexpr uint32_t kLegacyProfile = 0;
constexpr uint32_t kSimdTailProfile = 1;
constexpr size_t kFunctionNameCapacity = 64;

struct LinearLayerParams {
    uint32_t hidden_size;
    float rms_norm_eps;
};

struct RmsNormObservedTopologyV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
    uint32_t commits;
    uint32_t waits;
};

struct RmsNormCount43RuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t hidden_size;
    uint32_t rms_calls_per_run;
    uint32_t threads_per_threadgroup;
    uint32_t simdgroups_per_threadgroup;
    uint32_t pipeline_max_total_threads_per_threadgroup;
    uint32_t pipeline_thread_execution_width;
    uint32_t static_threadgroup_memory_bytes;
    uint32_t dynamic_threadgroup_memory_bytes;
    uint32_t internal_threadgroup_barriers_per_dispatch;
    uint32_t command_buffers_per_run;
    uint32_t compute_encoders_per_run;
    uint32_t kernel_dispatches_per_run;
    uint32_t explicit_buffer_barriers_per_run;
    uint32_t commits_per_run;
    uint32_t waits_per_run;
    uint32_t reserved_alignment;
    uint64_t host_to_device_bytes_per_run;
    uint64_t device_to_host_bytes_per_run;
    uint64_t successful_runs;
    uint32_t last_observed_command_buffers;
    uint32_t last_observed_compute_encoders;
    uint32_t last_observed_kernel_dispatches;
    uint32_t last_observed_explicit_buffer_barriers;
    uint32_t last_observed_commits;
    uint32_t last_observed_waits;
    char requested_function_name[kFunctionNameCapacity];
    char observed_function_name[kFunctionNameCapacity];
};

static_assert(sizeof(LinearLayerParams) == 8,
              "Metal RMSNorm parameter ABI changed");
static_assert(sizeof(RmsNormObservedTopologyV1) == 24,
              "Metal RMSNorm observed-topology ABI changed");
static_assert(sizeof(RmsNormCount43RuntimeReceiptV1) == 248,
              "Metal RMSNorm runtime-receipt ABI changed");

struct ApxinfMetalRmsNormCount43ProfileV1Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFunction> functions[kProfileCount];
    id<MTLComputePipelineState> pipelines[kProfileCount];
    id<MTLBuffer> weight;
    id<MTLBuffer> states;
    LinearLayerParams params;
    uint64_t successful_runs[kProfileCount];
    RmsNormObservedTopologyV1 last_observed[kProfileCount];
    bool has_staged_input;
    bool has_snapshot;
};

#include "metal_w8_linear_layer_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal RMSNorm error" : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal RMSNorm error"
                             : error.localizedDescription.UTF8String);
}

bool valid_profile(uint32_t profile) {
    return profile == kLegacyProfile || profile == kSimdTailProfile;
}

NSString *expected_function_name(uint32_t profile) {
    if (profile == kLegacyProfile) {
        return @"linear_layer_rms_norm";
    }
    if (profile == kSimdTailProfile) {
        return @"linear_layer_rms_norm_simd_tail_exact_v1";
    }
    return nil;
}

uint32_t internal_barriers_per_dispatch(uint32_t profile) {
    return profile == kLegacyProfile ? 9 : 4;
}

void copy_function_name(char output[kFunctionNameCapacity], NSString *name) {
    std::memset(output, 0, kFunctionNameCapacity);
    if (name != nil) {
        std::snprintf(output, kFunctionNameCapacity, "%s", name.UTF8String);
    }
}

bool live_profile_matches(ApxinfMetalRmsNormCount43ProfileV1Handle *handle,
                          uint32_t profile) {
    if (handle == nullptr || !valid_profile(profile) ||
        handle->functions[profile] == nil || handle->pipelines[profile] == nil) {
        return false;
    }
    NSString *expected = expected_function_name(profile);
    return expected != nil && [handle->functions[profile].name isEqualToString:expected] &&
           handle->pipelines[profile].threadExecutionWidth ==
               kRequiredThreadExecutionWidth &&
           handle->pipelines[profile].maxTotalThreadsPerThreadgroup >=
               kThreadsPerThreadgroup &&
           handle->pipelines[profile].staticThreadgroupMemoryLength ==
               kThreadsPerThreadgroup * sizeof(float);
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

}  // namespace

extern "C" int apxinf_metal_rms_norm_count43_profile_v1_create(
    const float *weight, uint32_t weight_count, float rms_norm_eps,
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal RMSNorm count43 output handle is null");
            return 1;
        }
        *output = nullptr;
        if (weight_count != kHiddenSize || !finite_f32(weight, weight_count) ||
            !std::isfinite(rms_norm_eps) || rms_norm_eps < 0.0f) {
            write_error(error_output, error_capacity,
                        "invalid Metal RMSNorm count43 weight or epsilon contract");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalRmsNormCount43ProfileV1Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal RMSNorm count43 handle failed");
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
        for (uint32_t profile = 0; profile < kProfileCount; ++profile) {
            NSString *name = expected_function_name(profile);
            handle->functions[profile] = [library newFunctionWithName:name];
            if (handle->functions[profile] != nil) {
                handle->pipelines[profile] =
                    [handle->device newComputePipelineStateWithFunction:handle->functions[profile]
                                                                   error:&error];
            }
            if (!live_profile_matches(handle, profile)) {
                delete handle;
                if (error != nil) {
                    write_nserror(error_output, error_capacity, error);
                } else {
                    write_error(error_output, error_capacity,
                                "Metal RMSNorm profile identity or 32-lane pipeline contract failed");
                }
                return 1;
            }
        }

        handle->queue = [handle->device newCommandQueue];
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        const size_t state_bytes =
            static_cast<size_t>(kRmsCallsPerRun + 1) * row_bytes;
        handle->weight = [handle->device newBufferWithBytes:weight
                                                     length:row_bytes
                                                    options:MTLResourceStorageModeShared];
        handle->states = [handle->device newBufferWithLength:state_bytes
                                                     options:MTLResourceStorageModeShared];
        if (handle->queue == nil || handle->weight == nil || handle->states == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal RMSNorm count43 resources failed");
            return 1;
        }
        std::memset(handle->states.contents, 0, state_bytes);
        handle->params = LinearLayerParams{kHiddenSize, rms_norm_eps};
        handle->has_staged_input = false;
        handle->has_snapshot = false;
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_rms_norm_count43_profile_v1_stage_input(
    void *opaque_handle, const float *input, uint32_t input_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalRmsNormCount43ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || input_count != kHiddenSize ||
            !finite_f32(input, input_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal RMSNorm count43 staged input");
            return 1;
        }
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        std::memcpy(handle->states.contents, input, row_bytes);
        handle->has_staged_input = true;
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_rms_norm_count43_profile_v1_run(
    void *opaque_handle, uint32_t profile,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalRmsNormCount43ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !valid_profile(profile) ||
            !handle->has_staged_input) {
            write_error(error_output, error_capacity,
                        "invalid Metal RMSNorm count43 run contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal RMSNorm function or pipeline identity changed");
            return 1;
        }

        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        handle->has_snapshot = false;
        RmsNormObservedTopologyV1 actual{};

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal RMSNorm count43 command buffer failed");
            return 1;
        }
        ++actual.command_buffers;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal RMSNorm count43 compute encoder failed");
            return 1;
        }
        ++actual.compute_encoders;
        [encoder setComputePipelineState:handle->pipelines[profile]];
        [encoder setBuffer:handle->weight offset:0 atIndex:1];
        [encoder setBytes:&handle->params length:sizeof(handle->params) atIndex:3];
        for (uint32_t call = 0; call < kRmsCallsPerRun; ++call) {
            const size_t input_offset = static_cast<size_t>(call) * row_bytes;
            const size_t output_offset = static_cast<size_t>(call + 1) * row_bytes;
            [encoder setBuffer:handle->states offset:input_offset atIndex:0];
            [encoder setBuffer:handle->states offset:output_offset atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
            ++actual.kernel_dispatches;
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ++actual.explicit_buffer_barriers;
        }
        [encoder endEncoding];
        if (actual.command_buffers != 1 || actual.compute_encoders != 1 ||
            actual.kernel_dispatches != kRmsCallsPerRun ||
            actual.explicit_buffer_barriers != kRmsCallsPerRun) {
            write_error(error_output, error_capacity,
                        "Metal RMSNorm count43 pre-commit topology mismatch");
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
                        "Metal RMSNorm count43 completion topology mismatch");
            return 1;
        }
        ++handle->successful_runs[profile];
        handle->last_observed[profile] = actual;
        handle->has_snapshot = true;
        return 0;
    }
}

extern "C" int apxinf_metal_rms_norm_count43_profile_v1_snapshot_chain(
    void *opaque_handle, float *output, uint32_t output_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalRmsNormCount43ProfileV1Handle *>(opaque_handle);
        constexpr uint32_t expected_count = kHiddenSize * kRmsCallsPerRun;
        if (handle == nullptr || output == nullptr || output_count != expected_count ||
            !handle->has_snapshot) {
            write_error(error_output, error_capacity,
                        "invalid Metal RMSNorm count43 snapshot contract");
            return 1;
        }
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        const auto *bytes = static_cast<const uint8_t *>(handle->states.contents);
        std::memcpy(output, bytes + row_bytes,
                    static_cast<size_t>(expected_count) * sizeof(float));
        return 0;
    }
}

extern "C" int apxinf_metal_rms_norm_count43_profile_v1_receipt(
    void *opaque_handle, uint32_t profile,
    RmsNormCount43RuntimeReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalRmsNormCount43ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr || !valid_profile(profile)) {
            write_error(error_output, error_capacity,
                        "invalid Metal RMSNorm count43 receipt contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal RMSNorm receipt identity changed");
            return 1;
        }
        std::memset(receipt, 0, sizeof(*receipt));
        receipt->requested_profile = profile;
        receipt->observed_profile = profile;
        receipt->hidden_size = kHiddenSize;
        receipt->rms_calls_per_run = kRmsCallsPerRun;
        receipt->threads_per_threadgroup = kThreadsPerThreadgroup;
        receipt->simdgroups_per_threadgroup =
            kThreadsPerThreadgroup / kRequiredThreadExecutionWidth;
        receipt->pipeline_max_total_threads_per_threadgroup = static_cast<uint32_t>(
            handle->pipelines[profile].maxTotalThreadsPerThreadgroup);
        receipt->pipeline_thread_execution_width = static_cast<uint32_t>(
            handle->pipelines[profile].threadExecutionWidth);
        receipt->static_threadgroup_memory_bytes = static_cast<uint32_t>(
            handle->pipelines[profile].staticThreadgroupMemoryLength);
        receipt->dynamic_threadgroup_memory_bytes = 0;
        receipt->internal_threadgroup_barriers_per_dispatch =
            internal_barriers_per_dispatch(profile);
        receipt->command_buffers_per_run = 1;
        receipt->compute_encoders_per_run = 1;
        receipt->kernel_dispatches_per_run = kRmsCallsPerRun;
        receipt->explicit_buffer_barriers_per_run = kRmsCallsPerRun;
        receipt->commits_per_run = 1;
        receipt->waits_per_run = 1;
        receipt->host_to_device_bytes_per_run = 0;
        receipt->device_to_host_bytes_per_run = 0;
        receipt->successful_runs = handle->successful_runs[profile];
        const auto &last = handle->last_observed[profile];
        receipt->last_observed_command_buffers = last.command_buffers;
        receipt->last_observed_compute_encoders = last.compute_encoders;
        receipt->last_observed_kernel_dispatches = last.kernel_dispatches;
        receipt->last_observed_explicit_buffer_barriers =
            last.explicit_buffer_barriers;
        receipt->last_observed_commits = last.commits;
        receipt->last_observed_waits = last.waits;
        NSString *expected = expected_function_name(profile);
        copy_function_name(receipt->requested_function_name, expected);
        copy_function_name(receipt->observed_function_name,
                           handle->functions[profile].name);
        return 0;
    }
}

extern "C" void apxinf_metal_rms_norm_count43_profile_v1_destroy(
    void *opaque_handle) {
    auto handle =
        static_cast<ApxinfMetalRmsNormCount43ProfileV1Handle *>(opaque_handle);
    delete handle;
}
