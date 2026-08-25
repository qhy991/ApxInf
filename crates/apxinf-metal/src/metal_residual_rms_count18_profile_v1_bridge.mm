#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kSeamsPerRun = 18;
constexpr uint32_t kThreadsPerThreadgroup = 256;
constexpr uint32_t kRequiredThreadExecutionWidth = 32;
constexpr uint32_t kLegacyProfile = 0;
constexpr uint32_t kFusedProfile = 1;
constexpr size_t kFunctionNameCapacity = 64;

struct LinearLayerParams {
    uint32_t hidden_size;
    float rms_norm_eps;
};

struct ResidualRmsObservedTopologyV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
    uint32_t pair_local_raw_barriers;
    uint32_t common_consumer_barriers;
    uint32_t commits;
    uint32_t waits;
};

struct ResidualRmsCount18RuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    uint32_t hidden_size;
    uint32_t seams_per_run;
    uint32_t threads_per_threadgroup;
    uint32_t simdgroups_per_threadgroup;
    uint32_t primary_pipeline_max_total_threads_per_threadgroup;
    uint32_t primary_pipeline_thread_execution_width;
    uint32_t primary_static_threadgroup_memory_bytes;
    uint32_t secondary_pipeline_max_total_threads_per_threadgroup;
    uint32_t secondary_pipeline_thread_execution_width;
    uint32_t secondary_static_threadgroup_memory_bytes;
    uint32_t dynamic_threadgroup_memory_bytes;
    uint32_t internal_threadgroup_barriers_per_seam;
    uint32_t internal_threadgroup_barriers_per_run;
    uint32_t command_buffers_per_run;
    uint32_t compute_encoders_per_run;
    uint32_t kernel_dispatches_per_run;
    uint32_t explicit_buffer_barriers_per_run;
    uint32_t pair_local_raw_barriers_per_run;
    uint32_t common_consumer_barriers_per_run;
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
    uint32_t last_observed_pair_local_raw_barriers;
    uint32_t last_observed_common_consumer_barriers;
    uint32_t last_observed_commits;
    uint32_t last_observed_waits;
    char requested_primary_function_name[kFunctionNameCapacity];
    char observed_primary_function_name[kFunctionNameCapacity];
    char requested_secondary_function_name[kFunctionNameCapacity];
    char observed_secondary_function_name[kFunctionNameCapacity];
};

static_assert(sizeof(LinearLayerParams) == 8,
              "Metal residual-RMS parameter ABI changed");
static_assert(sizeof(ResidualRmsObservedTopologyV1) == 32,
              "Metal residual-RMS observed-topology ABI changed");
static_assert(sizeof(ResidualRmsCount18RuntimeReceiptV1) == 408,
              "Metal residual-RMS runtime-receipt ABI changed");

struct ApxinfMetalResidualRmsCount18ProfileV1Handle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFunction> residual_function;
    id<MTLFunction> rms_function;
    id<MTLFunction> fused_function;
    id<MTLComputePipelineState> residual_pipeline;
    id<MTLComputePipelineState> rms_pipeline;
    id<MTLComputePipelineState> fused_pipeline;
    id<MTLBuffer> weight;
    id<MTLBuffer> update;
    id<MTLBuffer> seed;
    id<MTLBuffer> residual_trace;
    id<MTLBuffer> normalized_trace;
    LinearLayerParams params;
    uint64_t successful_runs[2];
    ResidualRmsObservedTopologyV1 last_observed[2];
    bool has_staged_seed;
    bool has_snapshot;
};

#include "metal_w8_linear_layer_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s",
                      message == nullptr ? "unknown Metal residual-RMS error"
                                         : message);
    }
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal residual-RMS error"
                             : error.localizedDescription.UTF8String);
}

bool valid_profile(uint32_t profile) {
    return profile == kLegacyProfile || profile == kFusedProfile;
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

void copy_function_name(char output[kFunctionNameCapacity], NSString *name) {
    std::memset(output, 0, kFunctionNameCapacity);
    if (name != nil) {
        std::snprintf(output, kFunctionNameCapacity, "%s", name.UTF8String);
    }
}

bool valid_pipeline(id<MTLComputePipelineState> pipeline,
                    NSUInteger expected_static_threadgroup_bytes) {
    return pipeline != nil &&
           pipeline.threadExecutionWidth == kRequiredThreadExecutionWidth &&
           pipeline.maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup &&
           pipeline.staticThreadgroupMemoryLength ==
               expected_static_threadgroup_bytes;
}

bool live_profile_matches(
    ApxinfMetalResidualRmsCount18ProfileV1Handle *handle, uint32_t profile) {
    if (handle == nullptr || !valid_profile(profile)) {
        return false;
    }
    if (profile == kLegacyProfile) {
        return handle->residual_function != nil && handle->rms_function != nil &&
               [handle->residual_function.name
                   isEqualToString:@"linear_layer_residual_add"] &&
               [handle->rms_function.name
                   isEqualToString:@"linear_layer_rms_norm"] &&
               valid_pipeline(handle->residual_pipeline, 0) &&
               valid_pipeline(handle->rms_pipeline,
                              kThreadsPerThreadgroup * sizeof(float));
    }
    return handle->fused_function != nil &&
           [handle->fused_function.name
               isEqualToString:
                   @"linear_layer_residual_rms_norm_fused_exact_v1"] &&
           valid_pipeline(handle->fused_pipeline,
                          kThreadsPerThreadgroup * sizeof(float));
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLFunction> function,
                                          NSError **error) {
    return function == nil
               ? nil
               : [device newComputePipelineStateWithFunction:function error:error];
}

uint32_t expected_dispatches(uint32_t profile) {
    return profile == kLegacyProfile ? 2 * kSeamsPerRun : kSeamsPerRun;
}

uint32_t expected_pair_local_barriers(uint32_t profile) {
    return profile == kLegacyProfile ? kSeamsPerRun : 0;
}

uint32_t expected_common_barriers() { return kSeamsPerRun; }

uint32_t expected_explicit_barriers(uint32_t profile) {
    return expected_pair_local_barriers(profile) + expected_common_barriers();
}

}  // namespace

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_create(
    const float *weight, uint32_t weight_count, float rms_norm_eps,
    void **output, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal residual-RMS count18 output handle is null");
            return 1;
        }
        *output = nullptr;
        constexpr uint32_t trace_count = kHiddenSize * kSeamsPerRun;
        if (weight_count != trace_count || !finite_f32(weight, weight_count) ||
            !std::isfinite(rms_norm_eps) || rms_norm_eps < 0.0f) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS count18 fixture contract");
            return 1;
        }

        auto handle =
            new (std::nothrow) ApxinfMetalResidualRmsCount18ProfileV1Handle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal residual-RMS count18 handle failed");
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
        handle->residual_function =
            [library newFunctionWithName:@"linear_layer_residual_add"];
        handle->rms_function =
            [library newFunctionWithName:@"linear_layer_rms_norm"];
        handle->fused_function = [library
            newFunctionWithName:
                @"linear_layer_residual_rms_norm_fused_exact_v1"];
        handle->residual_pipeline =
            make_pipeline(handle->device, handle->residual_function, &error);
        handle->rms_pipeline =
            make_pipeline(handle->device, handle->rms_function, &error);
        handle->fused_pipeline =
            make_pipeline(handle->device, handle->fused_function, &error);
        if (!live_profile_matches(handle, kLegacyProfile) ||
            !live_profile_matches(handle, kFusedProfile)) {
            delete handle;
            if (error != nil) {
                write_nserror(error_output, error_capacity, error);
            } else {
                write_error(
                    error_output, error_capacity,
                    "Metal residual-RMS function identity or pipeline contract failed");
            }
            return 1;
        }

        handle->queue = [handle->device newCommandQueue];
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        const size_t trace_bytes = static_cast<size_t>(kSeamsPerRun) * row_bytes;
        handle->weight = [handle->device newBufferWithBytes:weight
                                                     length:trace_bytes
                                                    options:MTLResourceStorageModeShared];
        handle->update = [handle->device newBufferWithLength:trace_bytes
                                                     options:MTLResourceStorageModeShared];
        handle->seed = [handle->device newBufferWithLength:row_bytes
                                                   options:MTLResourceStorageModeShared];
        handle->residual_trace =
            [handle->device newBufferWithLength:trace_bytes
                                         options:MTLResourceStorageModeShared];
        handle->normalized_trace =
            [handle->device newBufferWithLength:trace_bytes
                                         options:MTLResourceStorageModeShared];
        if (handle->queue == nil || handle->weight == nil ||
            handle->update == nil || handle->seed == nil ||
            handle->residual_trace == nil || handle->normalized_trace == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal residual-RMS resources failed");
            return 1;
        }
        std::memset(handle->seed.contents, 0, row_bytes);
        std::memset(handle->update.contents, 0, trace_bytes);
        std::memset(handle->residual_trace.contents, 0, trace_bytes);
        std::memset(handle->normalized_trace.contents, 0, trace_bytes);
        handle->params = LinearLayerParams{kHiddenSize, rms_norm_eps};
        handle->has_staged_seed = false;
        handle->has_snapshot = false;
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_stage_fixture(
    void *opaque_handle, const float *seed, uint32_t seed_count,
    const float *updates, uint32_t update_count, char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
        constexpr uint32_t trace_count = kHiddenSize * kSeamsPerRun;
        if (handle == nullptr || seed_count != kHiddenSize ||
            update_count != trace_count || !finite_f32(seed, seed_count) ||
            !finite_f32(updates, update_count)) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS count18 staged fixture");
            return 1;
        }
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        const size_t trace_bytes = static_cast<size_t>(trace_count) * sizeof(float);
        std::memcpy(handle->seed.contents, seed, row_bytes);
        std::memcpy(handle->update.contents, updates, trace_bytes);
        handle->has_staged_seed = true;
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_poison_traces(
    void *opaque_handle, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !handle->has_staged_seed) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS trace-poison contract");
            return 1;
        }
        constexpr size_t trace_count =
            static_cast<size_t>(kHiddenSize) * kSeamsPerRun;
        auto *residual = static_cast<float *>(handle->residual_trace.contents);
        auto *normalized = static_cast<float *>(handle->normalized_trace.contents);
        const float poison = std::numeric_limits<float>::quiet_NaN();
        for (size_t index = 0; index < trace_count; ++index) {
            residual[index] = poison;
            normalized[index] = poison;
        }
        handle->has_snapshot = false;
        return 0;
    }
}

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_run(
    void *opaque_handle, uint32_t profile,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || !valid_profile(profile) ||
            !handle->has_staged_seed) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS count18 run contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal residual-RMS function or pipeline identity changed");
            return 1;
        }

        handle->has_snapshot = false;
        ResidualRmsObservedTopologyV1 actual{};
        const size_t row_bytes = static_cast<size_t>(kHiddenSize) * sizeof(float);
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal residual-RMS command buffer failed");
            return 1;
        }
        ++actual.command_buffers;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) {
            write_error(error_output, error_capacity,
                        "create Metal residual-RMS compute encoder failed");
            return 1;
        }
        ++actual.compute_encoders;

        for (uint32_t seam = 0; seam < kSeamsPerRun; ++seam) {
            id<MTLBuffer> input = seam == 0 ? handle->seed
                                             : handle->normalized_trace;
            const size_t input_offset =
                seam == 0 ? 0 : static_cast<size_t>(seam - 1) * row_bytes;
            const size_t output_offset = static_cast<size_t>(seam) * row_bytes;
            if (profile == kLegacyProfile) {
                [encoder setComputePipelineState:handle->residual_pipeline];
                [encoder setBuffer:input offset:input_offset atIndex:0];
                [encoder setBuffer:handle->update
                              offset:output_offset
                            atIndex:1];
                [encoder setBuffer:handle->residual_trace
                              offset:output_offset
                            atIndex:2];
                [encoder setBytes:&handle->params
                           length:sizeof(handle->params)
                          atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(kHiddenSize, 1, 1)
                      threadsPerThreadgroup:
                          MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
                ++actual.kernel_dispatches;
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                ++actual.explicit_buffer_barriers;
                ++actual.pair_local_raw_barriers;

                [encoder setComputePipelineState:handle->rms_pipeline];
                [encoder setBuffer:handle->residual_trace
                              offset:output_offset
                            atIndex:0];
                [encoder setBuffer:handle->weight
                              offset:output_offset
                            atIndex:1];
                [encoder setBuffer:handle->normalized_trace
                              offset:output_offset
                            atIndex:2];
                [encoder setBytes:&handle->params
                           length:sizeof(handle->params)
                          atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                         threadsPerThreadgroup:
                             MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
                ++actual.kernel_dispatches;
            } else {
                [encoder setComputePipelineState:handle->fused_pipeline];
                [encoder setBuffer:input offset:input_offset atIndex:0];
                [encoder setBuffer:handle->update
                              offset:output_offset
                            atIndex:1];
                [encoder setBuffer:handle->weight
                              offset:output_offset
                            atIndex:2];
                [encoder setBuffer:handle->residual_trace
                              offset:output_offset
                            atIndex:3];
                [encoder setBuffer:handle->normalized_trace
                              offset:output_offset
                            atIndex:4];
                [encoder setBytes:&handle->params
                           length:sizeof(handle->params)
                          atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                         threadsPerThreadgroup:
                             MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
                ++actual.kernel_dispatches;
            }
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ++actual.explicit_buffer_barriers;
            ++actual.common_consumer_barriers;
        }
        [encoder endEncoding];
        if (actual.command_buffers != 1 || actual.compute_encoders != 1 ||
            actual.kernel_dispatches != expected_dispatches(profile) ||
            actual.explicit_buffer_barriers !=
                expected_explicit_barriers(profile) ||
            actual.pair_local_raw_barriers !=
                expected_pair_local_barriers(profile) ||
            actual.common_consumer_barriers != expected_common_barriers()) {
            write_error(error_output, error_capacity,
                        "Metal residual-RMS pre-commit topology mismatch");
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
                        "Metal residual-RMS completion topology mismatch");
            return 1;
        }
        ++handle->successful_runs[profile];
        handle->last_observed[profile] = actual;
        handle->has_snapshot = true;
        return 0;
    }
}

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_snapshot(
    void *opaque_handle, float *residual_output, uint32_t residual_count,
    float *normalized_output, uint32_t normalized_count,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
        constexpr uint32_t expected_count = kHiddenSize * kSeamsPerRun;
        if (handle == nullptr || residual_output == nullptr ||
            normalized_output == nullptr || residual_count != expected_count ||
            normalized_count != expected_count || !handle->has_snapshot) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS snapshot contract");
            return 1;
        }
        const size_t trace_bytes =
            static_cast<size_t>(expected_count) * sizeof(float);
        std::memcpy(residual_output, handle->residual_trace.contents,
                    trace_bytes);
        std::memcpy(normalized_output, handle->normalized_trace.contents,
                    trace_bytes);
        return 0;
    }
}

extern "C" int apxinf_metal_residual_rms_count18_profile_v1_receipt(
    void *opaque_handle, uint32_t profile,
    ResidualRmsCount18RuntimeReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<
            ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr || !valid_profile(profile)) {
            write_error(error_output, error_capacity,
                        "invalid Metal residual-RMS receipt contract");
            return 1;
        }
        if (!live_profile_matches(handle, profile)) {
            write_error(error_output, error_capacity,
                        "live Metal residual-RMS receipt identity changed");
            return 1;
        }
        std::memset(receipt, 0, sizeof(*receipt));
        receipt->requested_profile = profile;
        receipt->observed_profile = profile;
        receipt->hidden_size = kHiddenSize;
        receipt->seams_per_run = kSeamsPerRun;
        receipt->threads_per_threadgroup = kThreadsPerThreadgroup;
        receipt->simdgroups_per_threadgroup =
            kThreadsPerThreadgroup / kRequiredThreadExecutionWidth;
        id<MTLComputePipelineState> primary =
            profile == kLegacyProfile ? handle->residual_pipeline
                                      : handle->fused_pipeline;
        id<MTLComputePipelineState> secondary =
            profile == kLegacyProfile ? handle->rms_pipeline : nil;
        receipt->primary_pipeline_max_total_threads_per_threadgroup =
            static_cast<uint32_t>(primary.maxTotalThreadsPerThreadgroup);
        receipt->primary_pipeline_thread_execution_width =
            static_cast<uint32_t>(primary.threadExecutionWidth);
        receipt->primary_static_threadgroup_memory_bytes =
            static_cast<uint32_t>(primary.staticThreadgroupMemoryLength);
        if (secondary != nil) {
            receipt->secondary_pipeline_max_total_threads_per_threadgroup =
                static_cast<uint32_t>(
                    secondary.maxTotalThreadsPerThreadgroup);
            receipt->secondary_pipeline_thread_execution_width =
                static_cast<uint32_t>(secondary.threadExecutionWidth);
            receipt->secondary_static_threadgroup_memory_bytes =
                static_cast<uint32_t>(secondary.staticThreadgroupMemoryLength);
        }
        receipt->dynamic_threadgroup_memory_bytes = 0;
        receipt->internal_threadgroup_barriers_per_seam = 9;
        receipt->internal_threadgroup_barriers_per_run = 9 * kSeamsPerRun;
        receipt->command_buffers_per_run = 1;
        receipt->compute_encoders_per_run = 1;
        receipt->kernel_dispatches_per_run = expected_dispatches(profile);
        receipt->explicit_buffer_barriers_per_run =
            expected_explicit_barriers(profile);
        receipt->pair_local_raw_barriers_per_run =
            expected_pair_local_barriers(profile);
        receipt->common_consumer_barriers_per_run = expected_common_barriers();
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
        receipt->last_observed_pair_local_raw_barriers =
            last.pair_local_raw_barriers;
        receipt->last_observed_common_consumer_barriers =
            last.common_consumer_barriers;
        receipt->last_observed_commits = last.commits;
        receipt->last_observed_waits = last.waits;
        if (profile == kLegacyProfile) {
            copy_function_name(receipt->requested_primary_function_name,
                               @"linear_layer_residual_add");
            copy_function_name(receipt->observed_primary_function_name,
                               handle->residual_function.name);
            copy_function_name(receipt->requested_secondary_function_name,
                               @"linear_layer_rms_norm");
            copy_function_name(receipt->observed_secondary_function_name,
                               handle->rms_function.name);
        } else {
            copy_function_name(
                receipt->requested_primary_function_name,
                @"linear_layer_residual_rms_norm_fused_exact_v1");
            copy_function_name(receipt->observed_primary_function_name,
                               handle->fused_function.name);
        }
        return 0;
    }
}

extern "C" void apxinf_metal_residual_rms_count18_profile_v1_destroy(
    void *opaque_handle) {
    auto handle = static_cast<
        ApxinfMetalResidualRmsCount18ProfileV1Handle *>(opaque_handle);
    delete handle;
}
