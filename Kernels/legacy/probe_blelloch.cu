/**
 * probe_blelloch.cu — Legacy probe for the compile-time-D Blelloch path.
 *
 * Increments L (power-of-2) until a CUDA error occurs, reporting the last
 * successful L and the shared memory usage at that point. Also notes the
 * 48KB -> 96KB transition point where cudaFuncSetAttribute is required.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o probe_bl_D16  probe_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o probe_bl_D64  probe_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o probe_bl_D256 probe_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o probe_bl_D512 probe_blelloch.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../blelloch.cu"  // active runtime file; this legacy probe is kept for reference only

static constexpr size_t DEFAULT_SHMEM_LIMIT = 48 * 1024;   // 48KB default
static constexpr size_t EXTENDED_SHMEM_LIMIT = 96 * 1024;  // 96KB with opt-in

__global__ void blelloch_kernel(Element* d_in, Element* d_out, int L) {
    extern __shared__ Element shared_data[];
    __shared__ Element block_total;

    int tid = threadIdx.x;
    int half = L / 2;

    if (tid < half) {
        shared_data[phys(tid)]        = d_in[tid];
        shared_data[phys(tid + half)] = d_in[tid + half];
    }
    __syncthreads();

    Blelloch::block_scan(shared_data, L, &block_total);

    if (tid < half) {
        d_out[tid]        = shared_data[phys(tid)];
        d_out[tid + half] = shared_data[phys(tid + half)];
    }
}

int main() {
    printf("Blelloch shared memory probe  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);
    printf("  Default shmem limit:  48KB (no opt-in needed)\n");
    printf("  Extended shmem limit: 96KB (requires cudaFuncSetAttribute)\n\n");

    printf("%-8s %-12s %-10s %-20s %-s\n", "L", "shmem_KB", "threads", "shmem_regime", "result");
    printf("%.70s\n", "----------------------------------------------------------------------");

    int last_pass = -1;

    for (int L = 32; L <= 2048; L *= 2) {
        int shared_elems = L + CONFLICT_FREE_OFFSET(L - 1);
        size_t shared_bytes = (size_t)shared_elems * sizeof(Element);
        double shmem_kb = shared_bytes / 1024.0;
        int num_threads = L / 2;

        // Determine regime
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

        // Always use cudaFuncSetAttribute so we probe the true hardware limit
        cudaFuncSetAttribute(blelloch_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_bytes);

        blelloch_kernel<<<1, num_threads, shared_bytes>>>(d_in, d_out, L);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        printf("%-8d %-12.1f %-10d %-20s %s\n",
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
