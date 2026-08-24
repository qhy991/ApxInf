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

struct MlpParams {
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t gate_up_groups_per_row;
    uint32_t down_groups_per_row;
};

struct ApxinfMetalW8MlpBlockHandle {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
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

}  // namespace

extern "C" int apxinf_metal_w8_mlp_block_create(
    const int8_t *gate_up_weights, const float *gate_up_scales,
    const int8_t *down_weights, const float *down_scales,
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t group_size,
    void **output, char *error_output, size_t error_capacity) {
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
            intermediate_size % 4 != 0 || intermediate_size > UINT32_MAX / 2) {
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
        handle->gate_up_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_gate_up", &error);
        handle->activation_pipeline =
            make_pipeline(handle->device, library, @"w8_mlp_silu_mul", &error);
        handle->down_pipeline = make_pipeline(handle->device, library, @"w8_mlp_down", &error);
        if (handle->gate_up_pipeline == nil || handle->activation_pipeline == nil ||
            handle->down_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
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
        *output = handle;
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
        std::memcpy(handle->input.contents, input,
                    static_cast<size_t>(input_count) * sizeof(float));

        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity, "create Metal W8 MLP command buffer failed");
            return 1;
        }

        id<MTLComputeCommandEncoder> gate_up = [command computeCommandEncoder];
        if (gate_up == nil) {
            write_error(error_output, error_capacity, "create Metal W8 MLP gate+up encoder failed");
            return 1;
        }
        [gate_up setComputePipelineState:handle->gate_up_pipeline];
        [gate_up setBuffer:handle->gate_up_weights offset:0 atIndex:0];
        [gate_up setBuffer:handle->gate_up_scales offset:0 atIndex:1];
        [gate_up setBuffer:handle->input offset:0 atIndex:2];
        [gate_up setBuffer:handle->gate_up offset:0 atIndex:3];
        [gate_up setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        const uint32_t gate_up_rows = handle->params.intermediate_size * 2;
        [gate_up dispatchThreadgroups:MTLSizeMake(
                    (gate_up_rows + kRowsPerThreadgroup - 1) / kRowsPerThreadgroup, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        [gate_up endEncoding];

        id<MTLComputeCommandEncoder> activation = [command computeCommandEncoder];
        if (activation == nil) {
            write_error(error_output, error_capacity, "create Metal W8 MLP activation encoder failed");
            return 1;
        }
        [activation setComputePipelineState:handle->activation_pipeline];
        [activation setBuffer:handle->gate_up offset:0 atIndex:0];
        [activation setBuffer:handle->activated offset:0 atIndex:1];
        [activation setBytes:&handle->params length:sizeof(handle->params) atIndex:2];
        [activation dispatchThreadgroups:MTLSizeMake(
                       (handle->params.intermediate_size + kActivationThreads - 1) /
                           kActivationThreads,
                       1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kActivationThreads, 1, 1)];
        [activation endEncoding];

        id<MTLComputeCommandEncoder> down = [command computeCommandEncoder];
        if (down == nil) {
            write_error(error_output, error_capacity, "create Metal W8 MLP down encoder failed");
            return 1;
        }
        [down setComputePipelineState:handle->down_pipeline];
        [down setBuffer:handle->down_weights offset:0 atIndex:0];
        [down setBuffer:handle->down_scales offset:0 atIndex:1];
        [down setBuffer:handle->activated offset:0 atIndex:2];
        [down setBuffer:handle->output offset:0 atIndex:3];
        [down setBytes:&handle->params length:sizeof(handle->params) atIndex:4];
        [down dispatchThreadgroups:MTLSizeMake(
                  (handle->params.hidden_size + kRowsPerThreadgroup - 1) /
                      kRowsPerThreadgroup,
                  1, 1)
               threadsPerThreadgroup:MTLSizeMake(kMatVecThreads, 1, 1)];
        [down endEncoding];

        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            write_nserror(error_output, error_capacity, command.error);
            return 1;
        }
        std::memcpy(output, handle->output.contents,
                    static_cast<size_t>(output_count) * sizeof(float));
        return 0;
    }
}

extern "C" void apxinf_metal_w8_mlp_block_destroy(void *opaque_handle) {
    auto handle = static_cast<ApxinfMetalW8MlpBlockHandle *>(opaque_handle);
    delete handle;
}
