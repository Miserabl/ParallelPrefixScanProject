/**
 * test_blelloch.cu — Legacy correctness test for Blelloch::block_scan
 *
 * Tests single-block scan (L <= BLOCK_SIZE) against CPU reference .bin files.
 * For L > BLOCK_SIZE you need the chunked wrapper — test those configs later.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o test_blelloch_D16  test_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o test_blelloch_D64  test_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o test_blelloch_D256 test_blelloch.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o test_blelloch_D512 test_blelloch.cu
 *
 * Usage:
 *   ./test_blelloch_D16 [input_dir] [ref_dir]
 *   defaults: ../../SyntheticData/inputs   ../../SequentialBaseline/SequentialData
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../blelloch.cu"   // active runtime file; this legacy test is kept for reference only

// Kernel: one block, one scan
// Blelloch uses L/2 threads (each thread handles 2 elements)
// and dynamically allocated shared memory
__global__ void blelloch_kernel(Element* d_in, Element* d_out, int L) {
    extern __shared__ Element shared_data[];
    __shared__ Element block_total;

    int tid = threadIdx.x;
    int half = L / 2;

    if (tid < half) {
        int ai = tid;
        int bi = tid + half;
        shared_data[ai] = d_in[ai];
        shared_data[bi] = d_in[bi];
    }
    __syncthreads();

    Blelloch::block_scan(shared_data, L, &block_total);

    if (tid < half) {
        int ai = tid;
        int bi = tid + half;
        d_out[ai] = shared_data[ai];
        d_out[bi] = shared_data[bi];
    }
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

// ---------------------------------------------------------------------------
// Compare GPU output (Element array) against reference x array.
// Element.b[d] after the scan holds exactly x[t,d].
// ---------------------------------------------------------------------------
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

    // Test lengths scale with BLOCK_SIZE (D-dependent), must be power-of-2
    const int TEST_LENGTHS[] = {
        BLOCK_SIZE / 8,
        BLOCK_SIZE / 4,
        BLOCK_SIZE / 2,
        BLOCK_SIZE
    };
    const int N_L = sizeof(TEST_LENGTHS) / sizeof(TEST_LENGTHS[0]);

    printf("Blelloch correctness test  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);

    bool all_pass = true;

    for (int li = 0; li < N_L; li++) {
        int L = TEST_LENGTHS[li];
        int B = 1;

        // Load from L=1024 file, use first L elements
        char inpath[256];
        snprintf(inpath, sizeof(inpath), "%s/input_B%d_L1024_D%d.bin", indir, B, D);

        std::vector<float> a(1024 * D), b(1024 * D);
        if (!load_inputs(inpath, a.data(), b.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n", inpath);
            return 1;
        }

        // Load reference (from L=1024 file, use first L elements)
        char refpath[256];
        snprintf(refpath, sizeof(refpath), "%s/ref_B%d_L1024_D%d.bin", refdir, B, D);
        std::vector<float> ref(1024 * D);
        if (!load_ref(refpath, ref.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n  Run run_reference first.\n", refpath);
            return 1;
        }

        // Pack into Element array
        std::vector<Element> h_in(L), h_out(L);
        for (int t = 0; t < L; t++) {
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = a[t * D + d];
                h_in[t].b[d] = b[t * D + d];
            }
        }

        // Copy to device
        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));
        cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice);

        // Opt into extended shared memory (up to 96KB on A100)
        // Default is 48KB which is insufficient at large L and D.
        // Reduced occupancy is an expected tradeoff worth measuring in the benchmark.
        int num_threads = L / 2;
        int shared_elems = L + CONFLICT_FREE_OFFSET(L - 1);
        size_t shared_bytes = shared_elems * sizeof(Element);
        cudaFuncSetAttribute(
            blelloch_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);

        blelloch_kernel<<<1, num_threads, shared_bytes>>>(d_in, d_out, L);
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
