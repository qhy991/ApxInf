#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kMatVecThreads = kRowsPerThreadgroup * 32;
constexpr uint32_t kActivationThreads = 256;
constexpr uint32_t kSimdWidth = 32;
constexpr uint32_t kGateUpProfileLegacySeparate = 0;
constexpr uint32_t kGateUpProfileSemanticPairSilu = 1;
constexpr size_t kFunctionNameCapacity = 64;

struct MlpParams {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t gate_up_groups_per_row;
    uint32_t down_groups_per_row;
};

struct MlpCallTopologyCountersV1 {
    uint32_t command_buffers;
    uint32_t compute_encoders;
    uint32_t kernel_dispatches;
    uint32_t explicit_buffer_barriers;
};

struct MlpGateUpRuntimeReceiptV1 {
    uint32_t requested_profile;
    uint32_t observed_profile;
    char requested_function_name[kFunctionNameCapacity];
    char observed_function_name[kFunctionNameCapacity];
    uint32_t threads_per_threadgroup;
    uint32_t simdgroups_per_threadgroup;
    uint32_t semantic_pairs_per_threadgroup;
    uint32_t pipeline_max_total_threads_per_threadgroup;
    uint32_t pipeline_thread_execution_width;
    uint32_t static_threadgroup_memory_bytes;
    uint32_t dynamic_threadgroup_memory_bytes;
    uint32_t gate_up_threadgroups_per_call;
    uint32_t command_buffers_per_call;
    uint32_t compute_encoders_per_call;
    uint32_t kernel_dispatches_per_call;
    uint32_t explicit_buffer_barriers_per_call;
    uint32_t internal_threadgroup_barriers_per_call;
    uint64_t successful_calls;
    uint32_t last_observed_command_buffers;
    uint32_t last_observed_compute_encoders;
    uint32_t last_observed_kernel_dispatches;
    uint32_t last_observed_explicit_buffer_barriers;
};

static_assert(sizeof(MlpGateUpRuntimeReceiptV1) == 216,
              "Metal W8 MLP gate/up runtime receipt ABI changed");

struct ApxinfMetalW8MlpBlockHandle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFunction> gate_up_function;
    id<MTLComputePipelineState> gate_up_pipeline;
    id<MTLComputePipelineState> activation_pipeline;
    id<MTLComputePipelineState> down_pipeline;
    id<MTLBuffer> gate_up_weights;
    id<MTLBuffer> gate_up_scales;
    id<MTLBuffer> down_weights;
    id<MTLBuffer> down_scales;
    id<MTLBuffer> input;
    id<MTLBuffer> gate_up;
    id<MTLBuffer> activated;
    id<MTLBuffer> output;
    MlpParams params;
    uint32_t requested_gate_up_profile;
    uint32_t observed_gate_up_profile;
    bool production_topology;
    uint64_t successful_calls;
    MlpCallTopologyCountersV1 last_observed_topology;
};

#include "metal_w8_mlp_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(output, capacity, "%s", message == nullptr ? "unknown Metal error" : message);
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
    if (function == nil) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

NSString *gate_up_function_name(uint32_t profile) {
    switch (profile) {
        case kGateUpProfileLegacySeparate:
            return @"w8_mlp_gate_up";
        case kGateUpProfileSemanticPairSilu:
            return @"w8_mlp_gate_up_semantic_pair_silu";
        default:
            return nil;
    }
}

uint32_t observed_gate_up_profile(NSString *function_name) {
    if ([function_name isEqualToString:@"w8_mlp_gate_up"]) {
        return kGateUpProfileLegacySeparate;
    }
    if ([function_name isEqualToString:@"w8_mlp_gate_up_semantic_pair_silu"]) {
        return kGateUpProfileSemanticPairSilu;
    }
    return UINT32_MAX;
}

bool live_gate_up_profile_matches(
    const ApxinfMetalW8MlpBlockHandle *handle) {
    return handle != nullptr && handle->gate_up_function != nil &&
           observed_gate_up_profile(handle->gate_up_function.name) ==
               handle->requested_gate_up_profile;
}

MlpCallTopologyCountersV1 expected_call_topology(
    const ApxinfMetalW8MlpBlockHandle *handle) {
    if (handle->production_topology) {
        return MlpCallTopologyCountersV1{
            1,
            1,
            handle->observed_gate_up_profile ==
                    kGateUpProfileSemanticPairSilu
                ? 2u
                : 3u,
            handle->observed_gate_up_profile ==
                    kGateUpProfileSemanticPairSilu
                ? 1u
                : 2u,
        };
    }
    return MlpCallTopologyCountersV1{1, 3, 3, 0};
}

bool call_topology_matches(const MlpCallTopologyCountersV1 &left,
                           const MlpCallTopologyCountersV1 &right) {
    return left.command_buffers == right.command_buffers &&
           left.compute_encoders == right.compute_encoders &&
           left.kernel_dispatches == right.kernel_dispatches &&
           left.explicit_buffer_barriers ==
               right.explicit_buffer_barriers;
}

void encode_gate_up(ApxinfMetalW8MlpBlockHandle *handle,
                    id<MTLComputeCommandEncoder> encoder,
                    MlpCallTopologyCountersV1 *topology) {
    [encoder setComputePipelineState:handle->gate_up_pipeline];
    [encoder setBuffer:handle->gate_up_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->gate_up_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->input offset:0 atIndex:2];
    [encoder setBuffer:(handle->observed_gate_up_profile ==
                                kGateUpProfileSemanticPairSilu
                            ? handle->activated
                            : handle->gate_up)
                offset:0
              atIndex:3];
    [encoder setBytes:&handle->params
               length:sizeof(handle->params)
              atIndex:4];
    const uint32_t gate_up_rows =
        handle->observed_gate_up_profile == kGateUpProfileSemanticPairSilu
            ? handle->params.intermediate_size
            : handle->params.intermediate_size * 2;
    [encoder dispatchThreadgroups:MTLSizeMake(
                 (gate_up_rows + kRowsPerThreadgroup - 1) /
                     kRowsPerThreadgroup,
                 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    topology->kernel_dispatches += 1;
}

void encode_activation(ApxinfMetalW8MlpBlockHandle *handle,
                       id<MTLComputeCommandEncoder> encoder,
                       MlpCallTopologyCountersV1 *topology) {
    [encoder setComputePipelineState:handle->activation_pipeline];
    [encoder setBuffer:handle->gate_up offset:0 atIndex:0];
    [encoder setBuffer:handle->activated offset:0 atIndex:1];
    [encoder setBytes:&handle->params
               length:sizeof(handle->params)
              atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(
                 (handle->params.intermediate_size + kActivationThreads - 1) /
                     kActivationThreads,
                 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kActivationThreads, 1, 1)];
    topology->kernel_dispatches += 1;
}

void encode_down(ApxinfMetalW8MlpBlockHandle *handle,
                 id<MTLComputeCommandEncoder> encoder,
                 MlpCallTopologyCountersV1 *topology) {
    [encoder setComputePipelineState:handle->down_pipeline];
    [encoder setBuffer:handle->down_weights offset:0 atIndex:0];
    [encoder setBuffer:handle->down_scales offset:0 atIndex:1];
    [encoder setBuffer:handle->activated offset:0 atIndex:2];
    [encoder setBuffer:handle->output offset:0 atIndex:3];
    [encoder setBytes:&handle->params
               length:sizeof(handle->params)
              atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                 (handle->params.hidden_size + kRowsPerThreadgroup - 1) /
                     kRowsPerThreadgroup,
                 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
    topology->kernel_dispatches += 1;
}

void buffer_barrier(id<MTLComputeCommandEncoder> encoder,
                    MlpCallTopologyCountersV1 *topology) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    topology->explicit_buffer_barriers += 1;
}

}  // namespace

int create_mlp_block_handle(
    const int8_t *gate_up_weights, const float *gate_up_scales,
    const int8_t *down_weights, const float *down_scales,
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t group_size,
    uint32_t gate_up_profile, bool production_topology, void **output,
    char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity, "Metal W8 MLP output handle is null");
            return 1;
        }
        *output = nullptr;
        if (gate_up_weights == nullptr || gate_up_scales == nullptr ||
            down_weights == nullptr || down_scales == nullptr || hidden_size == 0 ||
            intermediate_size == 0 || group_size != 64 || hidden_size % group_size != 0 ||
            intermediate_size % group_size != 0 || hidden_size % 4 != 0 ||
            intermediate_size % 4 != 0 || intermediate_size > UINT32_MAX / 2 ||
            gate_up_profile > kGateUpProfileSemanticPairSilu) {
            write_error(error_output, error_capacity, "invalid Metal W8 MLP packed-weight contract");
            return 1;
        }

        const size_t gate_up_rows = static_cast<size_t>(intermediate_size) * 2;
        size_t gate_up_weight_bytes = 0;
        size_t gate_up_scale_count = 0;
        size_t down_weight_bytes = 0;
        size_t down_scale_count = 0;
        if (!checked_product(gate_up_rows, hidden_size, &gate_up_weight_bytes) ||
            !checked_product(gate_up_rows, hidden_size / group_size, &gate_up_scale_count) ||
            !checked_product(hidden_size, intermediate_size, &down_weight_bytes) ||
            !checked_product(hidden_size, intermediate_size / group_size, &down_scale_count)) {
            write_error(error_output, error_capacity, "Metal W8 MLP buffer dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalW8MlpBlockHandle{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity, "allocate Metal W8 MLP handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity, "no system Metal device is available");
            return 1;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:kMetalMlpSource];
        id<MTLLibrary> library = [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        NSString *requested_gate_up_name = gate_up_function_name(gate_up_profile);
        handle->gate_up_function =
            [library newFunctionWithName:requested_gate_up_name];
        if (handle->gate_up_function != nil) {
            handle->observed_gate_up_profile =
                observed_gate_up_profile(handle->gate_up_function.name);
            handle->gate_up_pipeline =
                [handle->device newComputePipelineStateWithFunction:
                                    handle->gate_up_function
                                                              error:&error];
        } else {
            handle->observed_gate_up_profile = UINT32_MAX;
        }
        if (gate_up_profile == kGateUpProfileLegacySeparate) {
            handle->activation_pipeline =
                make_pipeline(handle->device, library, @"w8_mlp_silu_mul", &error);
        }
        handle->down_pipeline = make_pipeline(handle->device, library, @"w8_mlp_down", &error);
        if (handle->gate_up_pipeline == nil || handle->down_pipeline == nil ||
            (gate_up_profile == kGateUpProfileLegacySeparate &&
             handle->activation_pipeline == nil)) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        if (handle->observed_gate_up_profile != gate_up_profile) {
            delete handle;
            write_error(error_output, error_capacity,
                        "selected Metal W8 MLP gate/up function identity drifted from the requested profile");
            return 1;
        }
        if (handle->gate_up_pipeline.maxTotalThreadsPerThreadgroup <
                kMatVecThreads ||
            handle->gate_up_pipeline.threadExecutionWidth != kSimdWidth) {
            delete handle;
            write_error(error_output, error_capacity,
                        "selected Metal W8 MLP gate/up pipeline does not satisfy the required 256-thread SIMD geometry");
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        const MTLResourceOptions private_storage = MTLResourceStorageModePrivate;
        handle->gate_up_weights = [handle->device newBufferWithBytes:gate_up_weights
                                                             length:gate_up_weight_bytes
                                                            options:shared];
        handle->gate_up_scales = [handle->device newBufferWithBytes:gate_up_scales
                                                            length:gate_up_scale_count * sizeof(float)
                                                           options:shared];
        handle->down_weights = [handle->device newBufferWithBytes:down_weights
                                                          length:down_weight_bytes
                                                         options:shared];
        handle->down_scales = [handle->device newBufferWithBytes:down_scales
                                                         length:down_scale_count * sizeof(float)
                                                        options:shared];
        handle->input = [handle->device newBufferWithLength:static_cast<size_t>(hidden_size) * sizeof(float)
                                                    options:shared];
        handle->gate_up = [handle->device newBufferWithLength:gate_up_rows * sizeof(float)
                                                      options:private_storage];
        handle->activated = [handle->device newBufferWithLength:static_cast<size_t>(intermediate_size) * sizeof(float)
                                                        options:private_storage];
        handle->output = [handle->device newBufferWithLength:static_cast<size_t>(hidden_size) * sizeof(float)
                                                     options:shared];
        if (handle->queue == nil || handle->gate_up_weights == nil ||
            handle->gate_up_scales == nil || handle->down_weights == nil ||
            handle->down_scales == nil || handle->input == nil || handle->gate_up == nil ||
            handle->activated == nil || handle->output == nil) {
            delete handle;
            write_error(error_output, error_capacity, "allocate persistent Metal W8 MLP buffers failed");
            return 1;
        }
        handle->params = MlpParams{
            hidden_size,
            intermediate_size,
            hidden_size / group_size,
            intermediate_size / group_size,
        };
        handle->requested_gate_up_profile = gate_up_profile;
        handle->production_topology = production_topology;
        handle->successful_calls = 0;
        handle->last_observed_topology = MlpCallTopologyCountersV1{};
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_block_create(
    const int8_t *gate_up_weights, const float *gate_up_scales,
    const int8_t *down_weights, const float *down_scales,
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t group_size,
    void **output, char *error_output, size_t error_capacity) {
    return create_mlp_block_handle(
        gate_up_weights, gate_up_scales, down_weights, down_scales, hidden_size,
        intermediate_size, group_size, kGateUpProfileLegacySeparate, false,
        output, error_output, error_capacity);
}

extern "C" int apxinf_metal_w8_mlp_block_create_with_gate_up_profile_v1(
    const int8_t *gate_up_weights, const float *gate_up_scales,
    const int8_t *down_weights, const float *down_scales,
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t group_size,
    uint32_t gate_up_profile, void **output, char *error_output,
    size_t error_capacity) {
    return create_mlp_block_handle(
        gate_up_weights, gate_up_scales, down_weights, down_scales, hidden_size,
        intermediate_size, group_size, gate_up_profile, true, output,
        error_output, error_capacity);
}

extern "C" int apxinf_metal_w8_mlp_block_gate_up_runtime_receipt_v1(
    void *opaque_handle, MlpGateUpRuntimeReceiptV1 *receipt,
    char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8MlpBlockHandle *>(opaque_handle);
        if (handle == nullptr || receipt == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP gate/up runtime receipt handle or output is null");
            return 1;
        }
        *receipt = MlpGateUpRuntimeReceiptV1{};
        NSString *requested_name =
            gate_up_function_name(handle->requested_gate_up_profile);
        NSString *observed_name = handle->gate_up_function.name;
        const uint32_t live_observed_profile =
            observed_gate_up_profile(observed_name);
        if (requested_name == nil || observed_name == nil ||
            live_observed_profile != handle->observed_gate_up_profile ||
            !live_gate_up_profile_matches(handle) ||
            handle->gate_up_pipeline == nil) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP gate/up runtime identity is invalid");
            return 1;
        }
        receipt->requested_profile = handle->requested_gate_up_profile;
        receipt->observed_profile = live_observed_profile;
        std::snprintf(receipt->requested_function_name,
                      kFunctionNameCapacity, "%s",
                      requested_name.UTF8String);
        std::snprintf(receipt->observed_function_name,
                      kFunctionNameCapacity, "%s", observed_name.UTF8String);
        receipt->threads_per_threadgroup = kMatVecThreads;
        receipt->simdgroups_per_threadgroup = kRowsPerThreadgroup;
        receipt->semantic_pairs_per_threadgroup =
            handle->observed_gate_up_profile ==
                    kGateUpProfileSemanticPairSilu
                ? kRowsPerThreadgroup
                : 0;
        receipt->pipeline_max_total_threads_per_threadgroup =
            static_cast<uint32_t>(
                handle->gate_up_pipeline.maxTotalThreadsPerThreadgroup);
        receipt->pipeline_thread_execution_width =
            static_cast<uint32_t>(handle->gate_up_pipeline.threadExecutionWidth);
        receipt->static_threadgroup_memory_bytes =
            static_cast<uint32_t>(
                handle->gate_up_pipeline.staticThreadgroupMemoryLength);
        receipt->dynamic_threadgroup_memory_bytes = 0;
        const uint32_t gate_up_rows =
            live_observed_profile == kGateUpProfileSemanticPairSilu
                ? handle->params.intermediate_size
                : handle->params.intermediate_size * 2;
        receipt->gate_up_threadgroups_per_call =
            (gate_up_rows + kRowsPerThreadgroup - 1) /
            kRowsPerThreadgroup;
        const MlpCallTopologyCountersV1 declared_topology =
            expected_call_topology(handle);
        receipt->command_buffers_per_call =
            declared_topology.command_buffers;
        receipt->compute_encoders_per_call =
            declared_topology.compute_encoders;
        receipt->kernel_dispatches_per_call =
            declared_topology.kernel_dispatches;
        receipt->explicit_buffer_barriers_per_call =
            declared_topology.explicit_buffer_barriers;
        receipt->internal_threadgroup_barriers_per_call = 0;
        receipt->successful_calls = handle->successful_calls;
        receipt->last_observed_command_buffers =
            handle->last_observed_topology.command_buffers;
        receipt->last_observed_compute_encoders =
            handle->last_observed_topology.compute_encoders;
        receipt->last_observed_kernel_dispatches =
            handle->last_observed_topology.kernel_dispatches;
        receipt->last_observed_explicit_buffer_barriers =
            handle->last_observed_topology.explicit_buffer_barriers;
        return 0;
    }
}

extern "C" int apxinf_metal_w8_mlp_block_forward(
    void *opaque_handle, const float *input, uint32_t input_count,
    float *output, uint32_t output_count, char *error_output, size_t error_capacity) {
    @autoreleasepool {
        auto handle = static_cast<ApxinfMetalW8MlpBlockHandle *>(opaque_handle);
        if (handle == nullptr || input == nullptr || output == nullptr ||
            input_count != handle->params.hidden_size || output_count != handle->params.hidden_size) {
            write_error(error_output, error_capacity, "invalid Metal W8 MLP input or output");
            return 1;
        }
        if (!live_gate_up_profile_matches(handle) ||
            observed_gate_up_profile(handle->gate_up_function.name) !=
                handle->observed_gate_up_profile) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP gate/up live function identity drifted before forward");
            return 1;
        }
        std::memcpy(handle->input.contents, input,
                    static_cast<size_t>(input_count) * sizeof(float));

        MlpCallTopologyCountersV1 observed_topology{};
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity, "create Metal W8 MLP command buffer failed");
            return 1;
        }
        observed_topology.command_buffers += 1;

        if (handle->production_topology) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) {
                write_error(error_output, error_capacity,
                            "create Metal W8 MLP production encoder failed");
                return 1;
            }
            observed_topology.compute_encoders += 1;
            encode_gate_up(handle, encoder, &observed_topology);
            buffer_barrier(encoder, &observed_topology);
            if (handle->observed_gate_up_profile ==
                kGateUpProfileLegacySeparate) {
                encode_activation(handle, encoder, &observed_topology);
                buffer_barrier(encoder, &observed_topology);
            }
            encode_down(handle, encoder, &observed_topology);
            [encoder endEncoding];
        } else {
            id<MTLComputeCommandEncoder> gate_up =
                [command computeCommandEncoder];
            if (gate_up == nil) {
                write_error(error_output, error_capacity,
                            "create Metal W8 MLP gate+up encoder failed");
                return 1;
            }
            observed_topology.compute_encoders += 1;
            encode_gate_up(handle, gate_up, &observed_topology);
            [gate_up endEncoding];

            id<MTLComputeCommandEncoder> activation =
                [command computeCommandEncoder];
            if (activation == nil) {
                write_error(error_output, error_capacity,
                            "create Metal W8 MLP activation encoder failed");
                return 1;
            }
            observed_topology.compute_encoders += 1;
            encode_activation(handle, activation, &observed_topology);
            [activation endEncoding];

            id<MTLComputeCommandEncoder> down =
                [command computeCommandEncoder];
            if (down == nil) {
                write_error(error_output, error_capacity,
                            "create Metal W8 MLP down encoder failed");
                return 1;
            }
            observed_topology.compute_encoders += 1;
            encode_down(handle, down, &observed_topology);
            [down endEncoding];
        }

        const MlpCallTopologyCountersV1 expected_topology =
            expected_call_topology(handle);
        if (!call_topology_matches(observed_topology,
                                   expected_topology)) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP observed command topology drifted before commit");
            return 1;
        }
        if (handle->successful_calls == UINT64_MAX) {
            write_error(error_output, error_capacity,
                        "Metal W8 MLP successful-call counter overflowed");
            return 1;
        }

        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        handle->successful_calls += 1;
        handle->last_observed_topology = observed_topology;
        std::memcpy(output, handle->output.contents,
                    static_cast<size_t>(output_count) * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_mlp_block_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8MlpBlockHandle *>(opaque_handle);
    delete handle;
}
