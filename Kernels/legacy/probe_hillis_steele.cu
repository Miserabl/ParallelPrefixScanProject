/**
 * probe_hillis_steele.cu — Legacy probe for the compile-time-D HillisSteele path.
 *
 * Hillis-Steele requires double-buffering: 2 * L * sizeof(Element) shared memory.
 * Uses dynamic shared memory split into two halves to probe beyond BLOCK_SIZE.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o probe_hs_D16  probe_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o probe_hs_D64  probe_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o probe_hs_D256 probe_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o probe_hs_D512 probe_hillis_steele.cu
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common.cuh"
#include "hillis_steele.cuh"

static constexpr size_t DEFAULT_SHMEM_LIMIT  = 48 * 1024;
static constexpr size_t EXTENDED_SHMEM_LIMIT = 96 * 1024;

// Probe kernel uses dynamic shared memory split into two halves for double-buffering.
// This lets us probe L values beyond the static BLOCK_SIZE limit.
// Layout: [buf_A: L elements][buf_B: L elements]
__global__ void probe_kernel(Element* d_in, Element* d_out, int L) {
    extern __shared__ Element shmem[];
    Element* buf_A = shmem;
    Element* buf_B = shmem + L;
    __shared__ Element block_total;

    int tid = threadIdx.x;
    if (tid < L) buf_A[tid] = d_in[tid];
    __syncthreads();

    // Run Hillis-Steele inline using buf_A/buf_B directly
    Element* in_buf  = buf_A;
    Element* out_buf = buf_B;

    for (int offset = 1; offset < L; offset <<= 1) {
        __syncthreads();
        if (tid < L) {
            if (tid >= offset)
                out_buf[tid] = combine(in_buf[tid - offset], in_buf[tid]);
            else
                out_buf[tid] = in_buf[tid];
        }
        Element* tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }
    __syncthreads();

    if (in_buf != buf_A && tid < L)
        buf_A[tid] = in_buf[tid];
    __syncthreads();

    if (tid < L) d_out[tid] = buf_A[tid];
    if (tid == L - 1) block_total = buf_A[tid];
    __syncthreads();
}

int main() {
    printf("HillisSteele shared memory probe  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);
    printf("  HillisSteele needs 2 buffers: shared memory = 2 * L * sizeof(Element)\n");
    printf("  Default shmem limit:  48KB (no opt-in needed)\n");
    printf("  Extended shmem limit: 96KB (requires cudaFuncSetAttribute)\n\n");

    printf("%-8s %-12s %-10s %-24s %-s\n", "L", "shmem_KB", "threads", "shmem_regime", "result");
    printf("%.72s\n", "------------------------------------------------------------------------");

    int last_pass = -1;

    for (int L = 1; L <= 2048; L *= 2) {
        // double buffer: 2 * L elements
        size_t shared_bytes = (size_t)2 * L * sizeof(Element);
        double shmem_kb = shared_bytes / 1024.0;
        int num_threads = L;

        const char* regime;
        if (shared_bytes <= DEFAULT_SHMEM_LIMIT)
            regime = "default (<=48KB)";
        else if (shared_bytes <= EXTENDED_SHMEM_LIMIT)
            regime = "*** extended (>48KB) ***";
        else
            regime = "EXCEEDS 96KB";

        std::vector<Element> h_in(L), h_out(L);
        for (int t = 0; t < L; t++)
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = 0.9f;
                h_in[t].b[d] = 1.0f;
            }

        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));
        cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(probe_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_bytes);

        probe_kernel<<<1, num_threads, shared_bytes>>>(d_in, d_out, L);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        printf("%-8d %-12.1f %-10d %-24s %s\n",
               L, shmem_kb, num_threads, regime,
               err == cudaSuccess ? "PASS" : cudaGetErrorString(err));

        if (err == cudaSuccess) last_pass = L;

        cudaFree(d_in);
        cudaFree(d_out);

        if (err != cudaSuccess) break;
    }

    printf("\nMax L per block for D=%d: %d\n", D, last_pass);
    return 0;
}
