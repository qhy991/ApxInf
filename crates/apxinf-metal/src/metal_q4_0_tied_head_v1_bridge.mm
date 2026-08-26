#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint32_t kAbiVersionV1 = 1;
constexpr uint32_t kBlockSizeV1 = 32;
constexpr size_t kBlockBytesV1 = 18;
constexpr uint32_t kRowsPerThreadgroupV1 = 8;
constexpr uint32_t kThreadsPerThreadgroupV1 = 256;
constexpr uint32_t kTopKV1 = 4;
constexpr uint32_t kMaxExcludedTokensV1 = 5;

struct Q4_0TiedHeadParamsV1 {
    uint32_t columns;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t partial_count;
    uint32_t excluded_tokens[kMaxExcludedTokensV1];
    uint32_t excluded_count;
};

struct Q4_0CandidateV1 {
    float score;
    uint32_t token;
};

static_assert(sizeof(Q4_0CandidateV1) == 8,
              "Q4_0 tied-head v1 candidate ABI must remain 8 bytes");
static_assert(sizeof(Q4_0TiedHeadParamsV1) == 40,
              "Q4_0 tied-head v1 parameter ABI must remain 40 bytes");

struct ApxinfMetalQ4_0TiedHeadHandleV1 {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> rows_pipeline;
    id<MTLComputePipelineState> final_pipeline;
    id<MTLBuffer> packed;
    id<MTLBuffer> hidden;
    id<MTLBuffer> scores;
    id<MTLBuffer> partial;
    id<MTLBuffer> tokens;
    id<MTLBuffer> status;
    Q4_0TiedHeadParamsV1 params;
};

#include "metal_q4_0_tied_head_v1_source.inc"

void write_error(char *output, size_t capacity, const char *message) {
    if (output == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(output, capacity, "%s",
                  message == nullptr ? "unknown Metal Q4_0 tied-head v1 error"
                                     : message);
}

void write_nserror(char *output, size_t capacity, NSError *error) {
    write_error(output, capacity,
                error == nil ? "unknown Metal Q4_0 tied-head v1 error"
                             : error.localizedDescription.UTF8String);
}

bool validate_hidden_before_state(
    const float *hidden,
    uint32_t hidden_count,
    uint32_t expected_count,
    char *error_output,
    size_t error_capacity) {
    if (hidden == nullptr || hidden_count != expected_count) {
        write_error(error_output, error_capacity,
                    "invalid Metal Q4_0 tied-head v1 hidden-row contract");
        return false;
    }
    for (uint32_t index = 0; index < hidden_count; ++index) {
        if (!std::isfinite(hidden[index])) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 hidden row contains a non-finite value");
            return false;
        }
    }
    return true;
}

bool configure_exclusions_before_state(
    const Q4_0TiedHeadParamsV1& base,
    const uint32_t *excluded_tokens,
    uint32_t excluded_count,
    Q4_0TiedHeadParamsV1 *output,
    char *error_output,
    size_t error_capacity) {
    if (output == nullptr || excluded_count > kMaxExcludedTokensV1 ||
        (excluded_count != 0 && excluded_tokens == nullptr) ||
        base.rows < kTopKV1) {
        write_error(error_output, error_capacity,
                    "invalid Metal Q4_0 tied-head v1 exclusion contract");
        return false;
    }
    *output = base;
    for (uint32_t index = 0; index < kMaxExcludedTokensV1; ++index) {
        output->excluded_tokens[index] = UINT32_MAX;
    }
    for (uint32_t index = 0; index < excluded_count; ++index) {
        const uint32_t token = excluded_tokens[index];
        if (token >= base.rows) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 exclusion token is outside vocabulary");
            return false;
        }
        for (uint32_t earlier = 0; earlier < index; ++earlier) {
            if (token == excluded_tokens[earlier]) {
                write_error(error_output, error_capacity,
                            "Metal Q4_0 tied-head v1 exclusion tokens contain a duplicate");
                return false;
            }
        }
        output->excluded_tokens[index] = token;
    }
    if (excluded_count > base.rows - kTopKV1) {
        write_error(error_output, error_capacity,
                    "Metal Q4_0 tied-head v1 exclusions leave fewer than four rows");
        return false;
    }
    output->excluded_count = excluded_count;
    return true;
}

bool valid_candidates(
    const uint32_t *tokens,
    uint32_t rows,
    const uint32_t *excluded_tokens,
    uint32_t excluded_count) {
    if (tokens == nullptr) {
        return false;
    }
    for (uint32_t index = 0; index < kTopKV1; ++index) {
        if (tokens[index] >= rows) {
            return false;
        }
        for (uint32_t earlier = 0; earlier < index; ++earlier) {
            if (tokens[index] == tokens[earlier]) {
                return false;
            }
        }
        for (uint32_t excluded = 0; excluded < excluded_count; ++excluded) {
            if (tokens[index] == excluded_tokens[excluded]) {
                return false;
            }
        }
    }
    return true;
}

bool encode_rows(
    ApxinfMetalQ4_0TiedHeadHandleV1 *handle,
    const Q4_0TiedHeadParamsV1& params,
    id<MTLCommandBuffer> command,
    char *error_output,
    size_t error_capacity) {
    id<MTLComputeCommandEncoder> rows = [command computeCommandEncoder];
    if (rows == nil) {
        write_error(error_output, error_capacity,
                    "create Metal Q4_0 tied-head v1 rows encoder failed");
        return false;
    }
    [rows setComputePipelineState:handle->rows_pipeline];
    [rows setBuffer:handle->packed offset:0 atIndex:0];
    [rows setBuffer:handle->hidden offset:0 atIndex:1];
    [rows setBuffer:handle->scores offset:0 atIndex:2];
    [rows setBuffer:handle->partial offset:0 atIndex:3];
    [rows setBytes:&params length:sizeof(params) atIndex:4];
    [rows setBuffer:handle->status offset:0 atIndex:5];
    [rows dispatchThreadgroups:MTLSizeMake(params.partial_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kThreadsPerThreadgroupV1, 1, 1)];
    [rows endEncoding];
    return true;
}

bool command_completed_without_score_poison(
    ApxinfMetalQ4_0TiedHeadHandleV1 *handle,
    id<MTLCommandBuffer> command,
    char *error_output,
    size_t error_capacity) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        write_nserror(error_output, error_capacity, command.error);
        return false;
    }
    const uint32_t status =
        *static_cast<const uint32_t *>(handle->status.contents);
    if (status != 0) {
        write_error(error_output, error_capacity,
                    "Metal Q4_0 tied-head v1 produced a non-finite score");
        return false;
    }
    return true;
}

void reset_status(ApxinfMetalQ4_0TiedHeadHandleV1 *handle) {
    const uint32_t clear = 0;
    std::memcpy(handle->status.contents, &clear, sizeof(clear));
}

void poison_tokens(ApxinfMetalQ4_0TiedHeadHandleV1 *handle) {
    const uint32_t poison[kTopKV1] = {
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    std::memcpy(handle->tokens.contents, poison, sizeof(poison));
}

}  // namespace

extern "C" int apxinf_metal_q4_0_tied_head_v1_create(
    const uint8_t *packed_bytes,
    size_t packed_byte_count,
    uint32_t rows,
    uint32_t columns,
    uint32_t abi_version,
    void **output,
    char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 output handle is null");
            return 1;
        }
        *output = nullptr;
        if (packed_bytes == nullptr || rows < kTopKV1 || columns == 0 ||
            columns % kBlockSizeV1 != 0 || abi_version != kAbiVersionV1) {
            write_error(error_output, error_capacity,
                        "invalid Metal Q4_0 tied-head v1 packed-weight ABI");
            return 1;
        }

        const size_t blocks_per_row = columns / kBlockSizeV1;
        if (blocks_per_row > std::numeric_limits<size_t>::max() / kBlockBytesV1) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 row-byte dimensions overflow");
            return 1;
        }
        const size_t row_bytes = blocks_per_row * kBlockBytesV1;
        if (static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max() / row_bytes) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 packed-byte dimensions overflow");
            return 1;
        }
        const size_t expected_packed_bytes = static_cast<size_t>(rows) * row_bytes;
        if (packed_byte_count != expected_packed_bytes) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 packed-byte count does not match dimensions");
            return 1;
        }
        for (size_t offset = 0; offset < packed_byte_count; offset += kBlockBytesV1) {
            const uint16_t scale_bits =
                static_cast<uint16_t>(packed_bytes[offset]) |
                (static_cast<uint16_t>(packed_bytes[offset + 1]) << 8);
            if ((scale_bits & 0x7c00u) == 0x7c00u) {
                write_error(error_output, error_capacity,
                            "Metal Q4_0 tied-head v1 packed stream contains a non-finite FP16 scale");
                return 1;
            }
        }

        const uint64_t partial_count_u64 =
            (static_cast<uint64_t>(rows) + kRowsPerThreadgroupV1 - 1) /
            kRowsPerThreadgroupV1;
        if (partial_count_u64 > UINT32_MAX) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 partial count exceeds the u32 ABI");
            return 1;
        }
        const uint32_t partial_count = static_cast<uint32_t>(partial_count_u64);
        if (static_cast<size_t>(rows) >
                std::numeric_limits<size_t>::max() / sizeof(float) ||
            static_cast<size_t>(partial_count) >
                std::numeric_limits<size_t>::max() /
                    (kTopKV1 * sizeof(Q4_0CandidateV1))) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 scratch dimensions overflow");
            return 1;
        }

        auto handle = new (std::nothrow) ApxinfMetalQ4_0TiedHeadHandleV1{};
        if (handle == nullptr) {
            write_error(error_output, error_capacity,
                        "allocate Metal Q4_0 tied-head v1 handle failed");
            return 1;
        }
        handle->device = MTLCreateSystemDefaultDevice();
        if (handle->device == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "no system Metal device is available for Q4_0 tied-head v1");
            return 1;
        }

        NSError *error = nil;
        NSString *source =
            [NSString stringWithUTF8String:kMetalQ4_0TiedHeadSourceV1];
        id<MTLLibrary> library =
            [handle->device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        id<MTLFunction> rows_function =
            [library newFunctionWithName:@"q4_0_tied_head_rows_v1"];
        id<MTLFunction> final_function =
            [library newFunctionWithName:@"q4_0_tied_head_final_topk4_v1"];
        if (rows_function == nil || final_function == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "load Metal Q4_0 tied-head v1 kernel functions failed");
            return 1;
        }
        handle->rows_pipeline =
            [handle->device newComputePipelineStateWithFunction:rows_function
                                                          error:&error];
        if (handle->rows_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        handle->final_pipeline =
            [handle->device newComputePipelineStateWithFunction:final_function
                                                          error:&error];
        if (handle->final_pipeline == nil) {
            delete handle;
            write_nserror(error_output, error_capacity, error);
            return 1;
        }
        if (handle->rows_pipeline.threadExecutionWidth != 32 ||
            handle->rows_pipeline.maxTotalThreadsPerThreadgroup <
                kThreadsPerThreadgroupV1 ||
            handle->final_pipeline.maxTotalThreadsPerThreadgroup <
                kThreadsPerThreadgroupV1) {
            delete handle;
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 requires SIMD32 and 256-thread pipelines");
            return 1;
        }
        handle->queue = [handle->device newCommandQueue];
        if (handle->queue == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "create Metal Q4_0 tied-head v1 command queue failed");
            return 1;
        }

        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        handle->packed = [handle->device newBufferWithBytes:packed_bytes
                                                    length:packed_byte_count
                                                   options:shared];
        handle->hidden = [handle->device
            newBufferWithLength:static_cast<size_t>(columns) * sizeof(float)
                       options:shared];
        handle->scores = [handle->device
            newBufferWithLength:static_cast<size_t>(rows) * sizeof(float)
                       options:MTLResourceStorageModePrivate];
        handle->partial = [handle->device
            newBufferWithLength:static_cast<size_t>(partial_count) * kTopKV1 *
                                sizeof(Q4_0CandidateV1)
                       options:MTLResourceStorageModePrivate];
        handle->tokens = [handle->device
            newBufferWithLength:kTopKV1 * sizeof(uint32_t)
                       options:shared];
        handle->status = [handle->device
            newBufferWithLength:sizeof(uint32_t)
                       options:shared];
        if (handle->packed == nil || handle->hidden == nil ||
            handle->scores == nil || handle->partial == nil ||
            handle->tokens == nil || handle->status == nil) {
            delete handle;
            write_error(error_output, error_capacity,
                        "allocate persistent Metal Q4_0 tied-head v1 buffers failed");
            return 1;
        }
        handle->params = Q4_0TiedHeadParamsV1{
            columns,
            rows,
            static_cast<uint32_t>(blocks_per_row),
            partial_count,
            {},
            0};
        reset_status(handle);
        poison_tokens(handle);
        *output = handle;
        return 0;
    }
}

extern "C" int apxinf_metal_q4_0_tied_head_v1_scores(
    void *opaque_handle,
    const float *hidden,
    uint32_t hidden_count,
    float *output_scores,
    uint32_t output_count,
    char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalQ4_0TiedHeadHandleV1 *>(opaque_handle);
        if (handle == nullptr || output_scores == nullptr ||
            output_count != handle->params.rows) {
            write_error(error_output, error_capacity,
                        "invalid Metal Q4_0 tied-head v1 score output contract");
            return 1;
        }
        Q4_0TiedHeadParamsV1 params{};
        if (!configure_exclusions_before_state(
                handle->params, nullptr, 0, &params,
                error_output, error_capacity) ||
            !validate_hidden_before_state(
                hidden, hidden_count, handle->params.columns,
                error_output, error_capacity)) {
            return 1;
        }

        id<MTLBuffer> readback = [handle->device
            newBufferWithLength:static_cast<size_t>(output_count) * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (readback == nil || command == nil) {
            write_error(error_output, error_capacity,
                        "allocate Metal Q4_0 tied-head v1 score readback command failed");
            return 1;
        }

        // All validation and temporary allocation above precede the first
        // persistent-buffer mutation for this call.
        std::memcpy(handle->hidden.contents, hidden,
                    static_cast<size_t>(hidden_count) * sizeof(float));
        reset_status(handle);
        if (!encode_rows(handle, params, command, error_output, error_capacity)) {
            return 1;
        }
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (blit == nil) {
            write_error(error_output, error_capacity,
                        "create Metal Q4_0 tied-head v1 score blit encoder failed");
            return 1;
        }
        const size_t score_bytes =
            static_cast<size_t>(output_count) * sizeof(float);
        [blit copyFromBuffer:handle->scores
               sourceOffset:0
                   toBuffer:readback
          destinationOffset:0
                       size:score_bytes];
        [blit endEncoding];
        if (!command_completed_without_score_poison(
                handle, command, error_output, error_capacity)) {
            return 1;
        }
        std::memcpy(output_scores, readback.contents, score_bytes);
        return 0;
    }
}

extern "C" int apxinf_metal_q4_0_tied_head_v1_topk4_excluding(
    void *opaque_handle,
    const float *hidden,
    uint32_t hidden_count,
    const uint32_t *excluded_tokens,
    uint32_t excluded_count,
    uint32_t *output_tokens,
    char *error_output,
    size_t error_capacity) {
    @autoreleasepool {
        auto handle =
            static_cast<ApxinfMetalQ4_0TiedHeadHandleV1 *>(opaque_handle);
        if (handle == nullptr || output_tokens == nullptr) {
            write_error(error_output, error_capacity,
                        "invalid Metal Q4_0 tied-head v1 candidate output contract");
            return 1;
        }
        Q4_0TiedHeadParamsV1 params{};
        if (!configure_exclusions_before_state(
                handle->params, excluded_tokens, excluded_count, &params,
                error_output, error_capacity) ||
            !validate_hidden_before_state(
                hidden, hidden_count, handle->params.columns,
                error_output, error_capacity)) {
            return 1;
        }
        id<MTLCommandBuffer> command = [handle->queue commandBuffer];
        if (command == nil) {
            write_error(error_output, error_capacity,
                        "create Metal Q4_0 tied-head v1 candidate command failed");
            return 1;
        }

        // All validation and command allocation above precede the first
        // persistent-buffer mutation or dispatch for this call.
        std::memcpy(handle->hidden.contents, hidden,
                    static_cast<size_t>(hidden_count) * sizeof(float));
        reset_status(handle);
        poison_tokens(handle);
        if (!encode_rows(handle, params, command, error_output, error_capacity)) {
            return 1;
        }
        id<MTLComputeCommandEncoder> final = [command computeCommandEncoder];
        if (final == nil) {
            write_error(error_output, error_capacity,
                        "create Metal Q4_0 tied-head v1 final encoder failed");
            return 1;
        }
        [final setComputePipelineState:handle->final_pipeline];
        [final setBuffer:handle->partial offset:0 atIndex:0];
        [final setBuffer:handle->tokens offset:0 atIndex:1];
        [final setBytes:&params length:sizeof(params) atIndex:2];
        [final dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kThreadsPerThreadgroupV1, 1, 1)];
        [final endEncoding];
        if (!command_completed_without_score_poison(
                handle, command, error_output, error_capacity)) {
            return 1;
        }
        const uint32_t *published_tokens =
            static_cast<const uint32_t *>(handle->tokens.contents);
        if (!valid_candidates(
                published_tokens, handle->params.rows,
                excluded_tokens, excluded_count)) {
            write_error(error_output, error_capacity,
                        "Metal Q4_0 tied-head v1 GPU candidates failed validation");
            return 1;
        }
        std::memcpy(output_tokens, published_tokens,
                    kTopKV1 * sizeof(uint32_t));
        return 0;
    }
}

extern "C" void apxinf_metal_q4_0_tied_head_v1_destroy(
    void *opaque_handle) {
    auto handle =
        static_cast<ApxinfMetalQ4_0TiedHeadHandleV1 *>(opaque_handle);
    delete handle;
}
