/**
 * test_hillis_steele.cu — Legacy correctness test for HillisSteele::block_scan
 *
 * Tests single-block scan (L <= BLOCK_SIZE) against CPU reference .bin files.
 * For L > BLOCK_SIZE you need the chunked wrapper.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o test_hs_D16  test_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o test_hs_D64  test_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o test_hs_D256 test_hillis_steele.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o test_hs_D512 test_hillis_steele.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../hillis_steele.cu"  // active runtime file; this legacy test is kept for reference only

// Kernel: one block, one scan.
// Allocates 2 * BLOCK_SIZE elements of dynamic shared memory:
//   [0..BLOCK_SIZE)      → shared_data (passed to block_scan)
//   [BLOCK_SIZE..2*BLOCK_SIZE) → buf_B (used internally by block_scan via pointer arithmetic)
__global__ void hillis_steele_kernel(Element* d_in, Element* d_out, int L) {
    extern __shared__ Element shmem[];   // 2 * BLOCK_SIZE elements
    Element* shared_data = shmem;        // first half
    __shared__ Element block_total;

    int tid = threadIdx.x;
    if (tid < L) shared_data[tid] = d_in[tid];
    __syncthreads();

    HillisSteele::block_scan(shared_data, L, &block_total);

    if (tid < L) d_out[tid] = shared_data[tid];
}


// File I/O
static bool load_inputs(const char* path, float* a, float* b, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(a, sizeof(float), n, f) == n
           && fread(b, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

static bool load_ref(const char* path, float* x, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(x, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

// Compare GPU output against reference x array.
// Element.b[d] after the scan holds exactly x[t,d].
static bool check(const Element* gpu_out, const float* ref, int L, float tol = 1e-4f) {
    int n_errors = 0;
    float max_err = 0.f;
    for (int t = 0; t < L; t++) {
        for (int d = 0; d < D; d++) {
            float got      = gpu_out[t].b[d];
            float expected = ref[t * D + d];
            float err      = fabsf(got - expected);
            if (err > max_err) max_err = err;
            if (err > tol) {
                if (n_errors < 5)
                    printf("  MISMATCH t=%d d=%d  got=%.6f  expected=%.6f  err=%.2e\n",
                           t, d, got, expected, err);
                n_errors++;
            }
        }
    }
    printf("  max_err=%.2e  n_errors=%d/%d  %s\n",
           max_err, n_errors, L * D, n_errors == 0 ? "PASS" : "FAIL");
    return n_errors == 0;
}


int main(int argc, char* argv[]) {
    const char* indir  = (argc > 1) ? argv[1] : "../../SyntheticData/inputs";
    const char* refdir = (argc > 2) ? argv[2] : "../../SequentialBaseline/SequentialData";

    // Test lengths scale with BLOCK_SIZE (D-dependent)
    const int TEST_LENGTHS[] = {
        BLOCK_SIZE / 8,
        BLOCK_SIZE / 4,
        BLOCK_SIZE / 2,
        BLOCK_SIZE
    };
    const int N_L = sizeof(TEST_LENGTHS) / sizeof(TEST_LENGTHS[0]);

    printf("HillisSteele correctness test  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);

    bool all_pass = true;

    for (int li = 0; li < N_L; li++) {
        int L = TEST_LENGTHS[li];
        int B = 1;

        char inpath[256];
        snprintf(inpath, sizeof(inpath), "%s/input_B%d_L1024_D%d.bin", indir, B, D);
        std::vector<float> a(1024 * D), b(1024 * D);
        if (!load_inputs(inpath, a.data(), b.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n", inpath);
            return 1;
        }

        char refpath[256];
        snprintf(refpath, sizeof(refpath), "%s/ref_B%d_L1024_D%d.bin", refdir, B, D);
        std::vector<float> ref(1024 * D);
        if (!load_ref(refpath, ref.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n  Run run_reference first.\n", refpath);
            return 1;
        }

        std::vector<Element> h_in(L), h_out(L);
        for (int t = 0; t < L; t++)
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = a[t * D + d];
                h_in[t].b[d] = b[t * D + d];
            }

        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));
        cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice);

        // 2 * BLOCK_SIZE for double-buffer: shared_data + buf_B
        size_t shmem_bytes = 2 * BLOCK_SIZE * sizeof(Element);
        cudaFuncSetAttribute(hillis_steele_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem_bytes);
        hillis_steele_kernel<<<1, BLOCK_SIZE, shmem_bytes>>>(d_in, d_out, L);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("L=%d: CUDA error: %s\n", L, cudaGetErrorString(err));
            all_pass = false;
            cudaFree(d_in); cudaFree(d_out);
            continue;
        }

        cudaMemcpy(h_out.data(), d_out, L * sizeof(Element), cudaMemcpyDeviceToHost);

        printf("L=%-4d D=%-4d  ", L, D);
        bool pass = check(h_out.data(), ref.data(), L);
        all_pass = all_pass && pass;

        cudaFree(d_in);
        cudaFree(d_out);
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
