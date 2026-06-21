#ifndef COMMON_CUH
#define COMMON_CUH

#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

enum class ScanDType {
    kFP32,
    kBF16,
    kFP16,
};

template <typename StorageT>
struct StorageTypeName;

template <>
struct StorageTypeName<float> {
    static constexpr const char* value = "fp32";
};

template <>
struct StorageTypeName<__nv_bfloat16> {
    static constexpr const char* value = "bf16";
};

template <>
struct StorageTypeName<__half> {
    static constexpr const char* value = "fp16";
};

static inline const char* scan_dtype_name(ScanDType dtype) {
    switch (dtype) {
        case ScanDType::kFP32: return "fp32";
        case ScanDType::kBF16: return "bf16";
        case ScanDType::kFP16: return "fp16";
    }
    return "unknown";
}

static inline bool parse_scan_dtype(const char* text, ScanDType* out) {
    if (!text || !out) return false;
    if (strcmp(text, "fp32") == 0) {
        *out = ScanDType::kFP32;
        return true;
    }
    if (strcmp(text, "bf16") == 0) {
        *out = ScanDType::kBF16;
        return true;
    }
    if (strcmp(text, "fp16") == 0) {
        *out = ScanDType::kFP16;
        return true;
    }
    return false;
}

template <typename StorageT>
__host__ __device__ inline float storage_to_float(StorageT value);

template <>
__host__ __device__ inline float storage_to_float<float>(float value) {
    return value;
}

template <>
__host__ __device__ inline float storage_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__host__ __device__ inline float storage_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename StorageT>
__host__ __device__ inline StorageT float_to_storage(float value);

template <>
__host__ __device__ inline float float_to_storage<float>(float value) {
    return value;
}

template <>
__host__ __device__ inline __half float_to_storage<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__host__ __device__ inline __nv_bfloat16 float_to_storage<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

// ---------------------------------------------------------------------------
// D is now a RUNTIME parameter — no -DD= compile flag needed.
// A single binary handles any hidden-state dimension.
//
// Why this is possible with the scalar design:
//   D is just a grid dimension (blockIdx.y).  The kernels never loop over D,
//   never store D floats per thread, and shared memory is fixed at 16 KB
//   regardless of D.  So D can be passed as a plain int at launch time.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CHUNK_SIZE: timesteps processed per block, per dimension.
// Fixed at 1024 regardless of D — shared memory cost is always 16 KB/block.
// ---------------------------------------------------------------------------
#define CHUNK_SIZE 1024

// Shared-memory bank geometry used by Blelloch's conflict-free padding.
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define BLELLOCH_PADDED_CHUNK  (CHUNK_SIZE + CHUNK_SIZE / NUM_BANKS)

// ---------------------------------------------------------------------------
// Data layout: dimension-major, two separate float arrays.
//   a_ptr[d * L + t]   b_ptr[d * L + t]    for d in [0,D), t in [0,L)
//
// Threads in a warp all share the same d (blockIdx.y) and hold consecutive
// t values (threadIdx.x) → perfectly coalesced HBM access.
// ---------------------------------------------------------------------------

// Shared-memory layout:
//   shmem[0            .. CHUNK_SIZE)   -> s_a       (primary a-values)
//   shmem[CHUNK_SIZE   .. 2*CHUNK_SIZE) -> s_b       (primary b-values)
//   shmem[2*CHUNK_SIZE .. 3*CHUNK_SIZE) -> aux_a     (HillisSteele buffer)
//   shmem[3*CHUNK_SIZE .. 4*CHUNK_SIZE) -> aux_b     (HillisSteele buffer)
//   shmem[4*CHUNK_SIZE .. SHMEM_FLOATS) -> extra tail used only by Blelloch's
//                                          conflict-free padded workspace
//
// We over-allocate by 64 floats (256 B) so Blelloch can scan in a padded,
// bank-conflict-resistant workspace while keeping the external compact layout
// unchanged for all kernels.
#define SHMEM_FLOATS  (2 * CHUNK_SIZE + 2 * BLELLOCH_PADDED_CHUNK)
#define SHMEM_BYTES   (SHMEM_FLOATS * sizeof(float))

#endif // COMMON_CUH
