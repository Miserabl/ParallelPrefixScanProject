#ifndef CHUNKED_HIERARCHICAL_RECURSIVE_CUH
#define CHUNKED_HIERARCHICAL_RECURSIVE_CUH

#include "common.cuh"
#include "hillis_steele.cu"
#include "blelloch.cu"
#include "warp_shuffle.cu"

// ---------------------------------------------------------------------------
// D is a RUNTIME int passed to chunked_scan / chunked_scan_impl.
// It is used only as a grid dimension and in index arithmetic — never inside
// any kernel body as a loop bound or array size.  This is why a single
// compiled binary can benchmark any hidden-state dimension.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 1: local scan per chunk × per dimension.
//   Grid(num_chunks, D_rt),  Block(CHUNK_SIZE)
//   blockIdx.y = d  (runtime dimension index)
// ---------------------------------------------------------------------------
template <typename ScanImpl, typename StorageT>
__global__ void phase1_scan(
    const StorageT* __restrict__ a_in,
    const StorageT* __restrict__ b_in,
          StorageT* __restrict__ a_out,
          StorageT* __restrict__ b_out,
          float* __restrict__ tot_a,   // [D_rt * num_chunks]
          float* __restrict__ tot_b,
    int L, int num_chunks)
{
    extern __shared__ float shmem[];

    const int d   = blockIdx.y;
    const int cid = blockIdx.x;
    const int tid = threadIdx.x;
    const int t   = cid * CHUNK_SIZE + tid;

    shmem[tid]              = (t < L) ? storage_to_float(a_in[d * L + t]) : 1.0f;
    shmem[CHUNK_SIZE + tid] = (t < L) ? storage_to_float(b_in[d * L + t]) : 0.0f;
    __syncthreads();

    __shared__ float s_tot_a, s_tot_b;
    ScanImpl::block_scan(shmem, CHUNK_SIZE, &s_tot_a, &s_tot_b);

    if (t < L) {
        a_out[d * L + t] = float_to_storage<StorageT>(shmem[tid]);
        b_out[d * L + t] = float_to_storage<StorageT>(shmem[CHUNK_SIZE + tid]);
    }
    if (tid == 0) {
        tot_a[d * num_chunks + cid] = s_tot_a;
        tot_b[d * num_chunks + cid] = s_tot_b;
    }
}

// ---------------------------------------------------------------------------
// Phase 2 (single-block): scan block totals when num_chunks <= CHUNK_SIZE.
//   Grid(D_rt, 1),  Block(CHUNK_SIZE)
// ---------------------------------------------------------------------------
template <typename ScanImpl>
__global__ void phase2_scan_totals(
    float* __restrict__ tot_a,
    float* __restrict__ tot_b,
    int num_chunks)
{
    extern __shared__ float shmem[];

    const int d   = blockIdx.x;
    const int tid = threadIdx.x;

    shmem[tid]              = (tid < num_chunks) ? tot_a[d * num_chunks + tid] : 1.0f;
    shmem[CHUNK_SIZE + tid] = (tid < num_chunks) ? tot_b[d * num_chunks + tid] : 0.0f;
    __syncthreads();

    __shared__ float s_tot_a, s_tot_b;
    ScanImpl::block_scan(shmem, num_chunks, &s_tot_a, &s_tot_b);

    if (tid < num_chunks) {
        tot_a[d * num_chunks + tid] = shmem[tid];
        tot_b[d * num_chunks + tid] = shmem[CHUNK_SIZE + tid];
    }
}

// ---------------------------------------------------------------------------
// Phase 3: propagate scanned totals into each chunk's elements.
//   Grid(num_chunks, D_rt),  Block(CHUNK_SIZE)
// ---------------------------------------------------------------------------
template <typename StorageT>
__global__ void phase3_propagate(
          StorageT* __restrict__ a_out,
          StorageT* __restrict__ b_out,
    const float* __restrict__ tot_a,
    const float* __restrict__ tot_b,
    int L, int num_chunks)
{
    const int d   = blockIdx.y;
    const int cid = blockIdx.x;
    const int tid = threadIdx.x;
    const int t   = cid * CHUNK_SIZE + tid;

    if (cid == 0 || t >= L) return;

    float pa = tot_a[d * num_chunks + cid - 1];
    float pb = tot_b[d * num_chunks + cid - 1];
    float xa = storage_to_float(a_out[d * L + t]);
    float xb = storage_to_float(b_out[d * L + t]);

    // combine(left=(pa,pb), right=(xa,xb))
    a_out[d * L + t] = float_to_storage<StorageT>(xa * pa);
    b_out[d * L + t] = float_to_storage<StorageT>(xa * pb + xb);
}

// ---------------------------------------------------------------------------
// chunked_scan_impl — internal recursive driver.
//   D_rt : runtime hidden-state dimension
//   d_scratch : pre-allocated, >= 2 * D_rt * L floats
// ---------------------------------------------------------------------------
template <typename ScanImpl, typename StorageT>
void chunked_scan_impl(
    StorageT* d_a_in,  StorageT* d_b_in,
    StorageT* d_a_out, StorageT* d_b_out,
    int L, int D_rt,
    float* d_scratch)
{
    if (L <= 0) return;

    const int num_chunks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;

    float* tot_a        = d_scratch;
    float* tot_b        = d_scratch + (size_t)D_rt * num_chunks;
    float* next_scratch = d_scratch + (size_t)D_rt * num_chunks * 2;

    dim3 grid1(num_chunks, D_rt);
    phase1_scan<ScanImpl, StorageT><<<grid1, CHUNK_SIZE, SHMEM_BYTES>>>(
        d_a_in, d_b_in, d_a_out, d_b_out, tot_a, tot_b, L, num_chunks);

    if (num_chunks <= CHUNK_SIZE) {
        phase2_scan_totals<ScanImpl><<<D_rt, CHUNK_SIZE, SHMEM_BYTES>>>(
            tot_a, tot_b, num_chunks);
    } else {
        // Recursive case (only for L > CHUNK_SIZE^2 = 1,048,576)
        float* tmp_a        = next_scratch;
        float* tmp_b        = next_scratch + (size_t)D_rt * num_chunks;
        float* deep_scratch = next_scratch + (size_t)D_rt * num_chunks * 2;
        chunked_scan_impl<ScanImpl, float>(
            tot_a, tot_b, tmp_a, tmp_b, num_chunks, D_rt, deep_scratch);
        cudaMemcpy(tot_a, tmp_a, (size_t)D_rt * num_chunks * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(tot_b, tmp_b, (size_t)D_rt * num_chunks * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    dim3 grid3(num_chunks, D_rt);
    phase3_propagate<StorageT><<<grid3, CHUNK_SIZE>>>(
        d_a_out, d_b_out, tot_a, tot_b, L, num_chunks);
}

// ---------------------------------------------------------------------------
// Public interface — D is now a runtime int argument.
//
//   chunked_scan<Impl>(a_in, b_in, a_out, b_out, L, D)
//     Allocates scratch internally (convenient, not for benchmarking).
//
//   chunked_scan<Impl>(a_in, b_in, a_out, b_out, L, D, d_scratch)
//     Caller provides scratch (>= 2 * D * L floats).  No allocation inside
//     the timed region.
// ---------------------------------------------------------------------------
template <typename ScanImpl, typename StorageT>
void chunked_scan(StorageT* d_a_in, StorageT* d_b_in,
                  StorageT* d_a_out, StorageT* d_b_out,
                  int L, int D_rt)
{
    float* d_scratch;
    cudaMalloc(&d_scratch, 2ULL * D_rt * L * sizeof(float));
    chunked_scan_impl<ScanImpl, StorageT>(d_a_in, d_b_in, d_a_out, d_b_out, L, D_rt, d_scratch);
    cudaFree(d_scratch);
}

template <typename ScanImpl, typename StorageT>
void chunked_scan(StorageT* d_a_in, StorageT* d_b_in,
                  StorageT* d_a_out, StorageT* d_b_out,
                  int L, int D_rt,
                  float* d_scratch)
{
    chunked_scan_impl<ScanImpl, StorageT>(d_a_in, d_b_in, d_a_out, d_b_out, L, D_rt, d_scratch);
}

#endif // CHUNKED_HIERARCHICAL_RECURSIVE_CUH
