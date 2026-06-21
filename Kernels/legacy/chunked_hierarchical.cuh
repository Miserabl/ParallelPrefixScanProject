#ifndef CHUNKED_HIERARCHICAL_CUH
#define CHUNKED_HIERARCHICAL_CUH

#include <cassert>
#include "../common.cuh"
#include "hillis_steele.cuh"
#include "blelloch.cuh"
#include "warp_shuffle.cuh"

// Phase 1: each block runs a local scan over its chunk of the input.
// Results are correct relative to each block only; block totals saved separately.
template <typename ScanImpl>
__global__ void phase1_local_scan(const Element* __restrict__ input,
                                   Element* __restrict__ output,
                                   Element* __restrict__ block_totals,
                                   int L) {
    __shared__ Element shared_data[BLOCK_SIZE];

    int block_start = blockIdx.x * BLOCK_SIZE;
    int tid = threadIdx.x;
    int global_idx = block_start + tid;

    shared_data[tid] = (global_idx < L) ? input[global_idx] : identity();
    __syncthreads();

    Element total;
    ScanImpl::block_scan(shared_data, BLOCK_SIZE, &total);
    __syncthreads();

    if (global_idx < L)
        output[global_idx] = shared_data[tid];

    if (tid == 0)
        block_totals[blockIdx.x] = total;
}

// Phase 2: scan the block totals so that block_totals[i] holds the prefix
// for all elements in blocks 0..i. Requires num_blocks <= BLOCK_SIZE.
template <typename ScanImpl>
__global__ void phase2_scan_totals(Element* __restrict__ block_totals,
                                    int num_blocks) {
    __shared__ Element shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;

    shared_data[tid] = (tid < num_blocks) ? block_totals[tid] : identity();
    __syncthreads();

    Element total;
    ScanImpl::block_scan(shared_data, num_blocks, &total);
    __syncthreads();

    if (tid < num_blocks)
        block_totals[tid] = shared_data[tid];
}

// Phase 3: propagate block prefixes into each block's locally-scanned output.
// Block 0 is already globally correct and returns immediately.
__global__ void phase3_propagate(Element* __restrict__ output,
                                  const Element* __restrict__ block_totals,
                                  int L) {
    if (blockIdx.x == 0) return;

    int global_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (global_idx >= L) return;

    Element prefix = block_totals[blockIdx.x - 1];
    output[global_idx] = combine(prefix, output[global_idx]);
}

// Host launch function. Allocates temp storage and runs all three phases.
// Limitation: num_blocks must be <= BLOCK_SIZE (256), i.e. L <= 65536.
// For L > 65536, Phase 2 would need to be applied recursively.
template <typename ScanImpl>
void chunked_scan(Element* d_input, Element* d_output, int L) {
    int num_blocks = (L + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert(num_blocks <= BLOCK_SIZE &&
           "chunked_scan: L too large — Phase 2 fits at most BLOCK_SIZE^2 elements");

    Element* d_block_totals;
    cudaMalloc(&d_block_totals, num_blocks * sizeof(Element));

    phase1_local_scan<ScanImpl><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, d_block_totals, L);

    phase2_scan_totals<ScanImpl><<<1, BLOCK_SIZE>>>(
        d_block_totals, num_blocks);

    phase3_propagate<<<num_blocks, BLOCK_SIZE>>>(
        d_output, d_block_totals, L);

    cudaFree(d_block_totals);
}

#endif // CHUNKED_HIERARCHICAL_CUH
