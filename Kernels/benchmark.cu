#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/stat.h>

#include "benchmark_runtime.cuh"
#include "chunked_hierarchical_recursive.cuh"

static const int DEFAULT_DIMS[] = { 1, 16, 64, 256, 512 };
static const int N_DEFAULT_DIMS = static_cast<int>(sizeof(DEFAULT_DIMS) / sizeof(DEFAULT_DIMS[0]));
static const int SEQ_LENGTHS[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
static const int N_L = static_cast<int>(sizeof(SEQ_LENGTHS) / sizeof(SEQ_LENGTHS[0]));
static const int B_BATCH = 1;
static const int DEFAULT_WARMUP = 10;
static const int DEFAULT_REPEAT = 50;

static const char* CSV_HEADER =
    "kernel,D,L,time_ms,median_ms,p25_ms,p75_ms,iqr_ms,mean_ms,stddev_ms,"
    "correct,throughput_GB_s,throughput_metric,warmup,repeat,dtype,samples_ms\n";

struct Options {
    const char* input_dir = "../SyntheticData/inputs";
    const char* ref_dir = "../SequentialBaseline/SequentialData";
    const char* output_dir = "../Results";
    const char* kernel_filter = nullptr;
    int warmup = DEFAULT_WARMUP;
    int repeat = DEFAULT_REPEAT;
    ScanDType dtype = ScanDType::kFP32;
    std::vector<int> dims = std::vector<int>(DEFAULT_DIMS, DEFAULT_DIMS + N_DEFAULT_DIMS);
    std::vector<int> lengths = std::vector<int>(SEQ_LENGTHS, SEQ_LENGTHS + N_L);
};

static bool load_inputs(const char* path, float* a, float* b, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return false;
    }
    bool ok = fread(a, sizeof(float), n, f) == n && fread(b, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

static bool load_ref(const char* path, float* x, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return false;
    }
    bool ok = fread(x, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

static void to_dim_major(const float* src, float* dst, int L, int D) {
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            dst[d * L + t] = src[t * D + d];
        }
    }
}

template <typename StorageT>
static void run_warp_shuffle(StorageT* ai, StorageT* bi, StorageT* ao, StorageT* bo, int L, int D, float* sc) {
    chunked_scan<WarpShuffle, StorageT>(ai, bi, ao, bo, L, D, sc);
}

template <typename StorageT>
static void run_blelloch(StorageT* ai, StorageT* bi, StorageT* ao, StorageT* bo, int L, int D, float* sc) {
    chunked_scan<Blelloch, StorageT>(ai, bi, ao, bo, L, D, sc);
}

template <typename StorageT>
static void run_hillis_steele(StorageT* ai, StorageT* bi, StorageT* ao, StorageT* bo, int L, int D, float* sc) {
    chunked_scan<HillisSteele, StorageT>(ai, bi, ao, bo, L, D, sc);
}

static bool parse_int(const char* text, int* out) {
    if (!text || !out) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    if (value < 0 || value > 2147483647L) return false;
    *out = static_cast<int>(value);
    return true;
}

static bool is_known_kernel_name(const char* kernel_name) {
    if (!kernel_name) return false;
    return std::strcmp(kernel_name, "warp_shuffle") == 0 ||
           std::strcmp(kernel_name, "blelloch") == 0 ||
           std::strcmp(kernel_name, "hillis_steele") == 0 ||
           std::strcmp(kernel_name, "cub") == 0;
}

static void print_usage(const char* argv0) {
    printf("Usage:\n");
    printf("  %s [input_dir] [ref_dir] [output_dir] [options]\n", argv0);
    printf("\nOptions:\n");
    printf("  --D <dim>             Restrict sweep to one hidden dimension\n");
    printf("  --L <length>          Restrict sweep to one sequence length\n");
    printf("  --kernel <name>       One of: warp_shuffle, blelloch, hillis_steele, cub\n");
    printf("  --dtype <name>        One of: fp32, bf16, fp16\n");
    printf("  --warmup <int>        Default: %d\n", DEFAULT_WARMUP);
    printf("  --repeat <int>        Default: %d\n", DEFAULT_REPEAT);
    printf("  --help                Show this message\n");
}

static bool parse_args(int argc, char* argv[], Options* options) {
    if (!options) return false;

    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if (std::strcmp(arg, "--D") == 0 && i + 1 < argc) {
            int value = 0;
            if (!parse_int(argv[++i], &value) || value <= 0) return false;
            options->dims = { value };
            continue;
        }
        if (std::strcmp(arg, "--L") == 0 && i + 1 < argc) {
            int value = 0;
            if (!parse_int(argv[++i], &value) || value <= 0) return false;
            options->lengths = { value };
            continue;
        }
        if (std::strcmp(arg, "--kernel") == 0 && i + 1 < argc) {
            options->kernel_filter = argv[++i];
            continue;
        }
        if (std::strcmp(arg, "--dtype") == 0 && i + 1 < argc) {
            if (!parse_scan_dtype(argv[++i], &options->dtype)) return false;
            continue;
        }
        if (std::strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &options->warmup) || options->warmup < 0) return false;
            continue;
        }
        if (std::strcmp(arg, "--repeat") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &options->repeat) || options->repeat <= 0) return false;
            continue;
        }
        if (arg[0] != '-') {
            if (positional == 0) options->input_dir = arg;
            else if (positional == 1) options->ref_dir = arg;
            else if (positional == 2) options->output_dir = arg;
            else return false;
            ++positional;
            continue;
        }
        fprintf(stderr, "Unknown argument: %s\n", arg);
        return false;
    }

    if (options->kernel_filter && !is_known_kernel_name(options->kernel_filter)) {
        fprintf(stderr, "Unknown kernel '%s'\n", options->kernel_filter);
        return false;
    }
    return true;
}

template <typename StorageT>
static void run_one_LD_t(int L, int D, const Options& options, FILE* csv) {
    const size_t n = static_cast<size_t>(D) * L;
    const char* dtype_name = StorageTypeName<StorageT>::value;

    char input_path[256];
    char ref_path[256];
    snprintf(input_path, sizeof(input_path), "%s/input_B%d_L%d_D%d.bin", options.input_dir, B_BATCH, L, D);
    snprintf(ref_path, sizeof(ref_path), "%s/ref_B%d_L%d_D%d.bin", options.ref_dir, B_BATCH, L, D);

    std::vector<float> a_tm(n), b_tm(n), ref(n);
    if (!load_inputs(input_path, a_tm.data(), b_tm.data(), n)) {
        fprintf(stderr, "  Missing input: %s — skipping\n", input_path);
        return;
    }
    if (!load_ref(ref_path, ref.data(), n)) {
        fprintf(stderr, "  Missing ref: %s — skipping\n", ref_path);
        return;
    }

    std::vector<float> a_dm_fp32(n), b_dm_fp32(n);
    to_dim_major(a_tm.data(), a_dm_fp32.data(), L, D);
    to_dim_major(b_tm.data(), b_dm_fp32.data(), L, D);

    std::vector<StorageT> a_dm(n), b_dm(n);
    convert_float_buffer_to_storage(a_dm_fp32.data(), a_dm.data(), n);
    convert_float_buffer_to_storage(b_dm_fp32.data(), b_dm.data(), n);

    std::vector<PairABT<StorageT>> cub_in(n);
    pack_pairs(a_dm.data(), b_dm.data(), cub_in.data(), n);

    std::vector<int> cub_keys(n);
    for (int d = 0; d < D; ++d) {
        for (int t = 0; t < L; ++t) cub_keys[d * L + t] = d;
    }

    StorageT* d_ai = nullptr;
    StorageT* d_bi = nullptr;
    StorageT* d_ao = nullptr;
    StorageT* d_bo = nullptr;
    float* d_sc = nullptr;
    cudaMalloc(&d_ai, n * sizeof(StorageT));
    cudaMalloc(&d_bi, n * sizeof(StorageT));
    cudaMalloc(&d_ao, n * sizeof(StorageT));
    cudaMalloc(&d_bo, n * sizeof(StorageT));
    cudaMalloc(&d_sc, 2ULL * D * L * sizeof(float));

    PairABT<StorageT>* d_cub_in = nullptr;
    PairABT<StorageT>* d_cub_out = nullptr;
    int* d_cub_keys = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
    cudaMalloc(&d_cub_in, n * sizeof(PairABT<StorageT>));
    cudaMalloc(&d_cub_out, n * sizeof(PairABT<StorageT>));
    cudaMalloc(&d_cub_keys, n * sizeof(int));
    cudaMemcpy(d_cub_keys, cub_keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, cub_temp_bytes,
        d_cub_keys, d_cub_in, d_cub_out,
        PairABScanOpT<StorageT>{}, static_cast<int>(n));
    cudaMalloc(&d_cub_temp, cub_temp_bytes);

    struct KernelSpec {
        const char* name;
        KernelFnT<StorageT> fn;
    } kernels[] = {
        { "warp_shuffle", run_warp_shuffle<StorageT> },
        { "blelloch", run_blelloch<StorageT> },
        { "hillis_steele", run_hillis_steele<StorageT> },
    };

    for (const auto& kernel : kernels) {
        if (options.kernel_filter && std::strcmp(options.kernel_filter, kernel.name) != 0) continue;
        const BenchmarkResult result = benchmark_kernel<StorageT>(
            kernel.fn,
            d_ai, d_bi, d_ao, d_bo, d_sc,
            a_dm.data(), b_dm.data(), ref.data(),
            L, D, options.warmup, options.repeat);
        print_result(kernel.name, dtype_name, D, L, result);
        write_csv_row(csv, kernel.name, dtype_name, D, L, result, options.warmup, options.repeat);
    }

    if (!options.kernel_filter || std::strcmp(options.kernel_filter, "cub") == 0) {
        const BenchmarkResult result = benchmark_cub_kernel<StorageT>(
            d_cub_in, d_cub_out, d_cub_keys, d_cub_temp, cub_temp_bytes,
            cub_in.data(), ref.data(), L, D, options.warmup, options.repeat);
        print_result("cub", dtype_name, D, L, result);
        write_csv_row(csv, "cub", dtype_name, D, L, result, options.warmup, options.repeat);
    }

    cudaFree(d_ai);
    cudaFree(d_bi);
    cudaFree(d_ao);
    cudaFree(d_bo);
    cudaFree(d_sc);
    cudaFree(d_cub_in);
    cudaFree(d_cub_out);
    cudaFree(d_cub_keys);
    cudaFree(d_cub_temp);
}

static void run_one_LD(int L, int D, const Options& options, FILE* csv) {
    switch (options.dtype) {
        case ScanDType::kFP32:
            run_one_LD_t<float>(L, D, options, csv);
            return;
        case ScanDType::kBF16:
            run_one_LD_t<__nv_bfloat16>(L, D, options, csv);
            return;
        case ScanDType::kFP16:
            run_one_LD_t<__half>(L, D, options, csv);
            return;
    }
}

int main(int argc, char* argv[]) {
    Options options;
    if (!parse_args(argc, argv, &options)) {
        if (argc > 1 && std::strcmp(argv[1], "--help") == 0) return 0;
        print_usage(argv[0]);
        return 1;
    }

    mkdir(options.output_dir, 0755);
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/benchmark.csv", options.output_dir);
    FILE* csv = fopen(csv_path, "w");
    if (csv) fputs(CSV_HEADER, csv);

    printf("SSM Prefix Scan Benchmark  CHUNK_SIZE=%d\n", CHUNK_SIZE);
    printf("warmup=%d repeat=%d metric=logical_payload dtype=%s\n",
           options.warmup, options.repeat, scan_dtype_name(options.dtype));
    if (options.kernel_filter) printf("Kernel: %s\n", options.kernel_filter);
    printf("Dims: ");
    for (int d : options.dims) printf("%d ", d);
    printf("\nLens: ");
    for (int l : options.lengths) printf("%d ", l);
    printf("\n\n");

    for (int D : options.dims) {
        printf("=== D = %d ===\n", D);
        for (int L : options.lengths) run_one_LD(L, D, options, csv);
        printf("\n");
    }

    if (csv) fclose(csv);
    printf("Results saved to %s\n", csv_path);
    return 0;
}
