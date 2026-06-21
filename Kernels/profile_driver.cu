#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_runtime.cuh"
#include "chunked_hierarchical_recursive.cuh"

constexpr int kBatchSize = 1;
constexpr int kDefaultWarmup = 10;
constexpr int kDefaultRepeat = 50;
const char* kCsvHeader =
    "kernel,D,L,time_ms,median_ms,p25_ms,p75_ms,iqr_ms,mean_ms,stddev_ms,"
    "correct,throughput_GB_s,throughput_metric,warmup,repeat,dtype,samples_ms\n";

struct Options {
    std::string kernel;
    int D = -1;
    int L = -1;
    std::string input_dir = "../SyntheticData/inputs";
    std::string ref_dir = "../SequentialBaseline/SequentialData";
    std::string csv_append;
    int warmup = kDefaultWarmup;
    int repeat = kDefaultRepeat;
    ScanDType dtype = ScanDType::kFP32;
    bool no_print = false;
    bool skip_check = false;
};

template <typename StorageT>
static void launch_warp_shuffle(StorageT* d_ai, StorageT* d_bi, StorageT* d_ao, StorageT* d_bo,
                                int L, int D, float* d_sc) {
    chunked_scan<WarpShuffle, StorageT>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
}

template <typename StorageT>
static void launch_blelloch(StorageT* d_ai, StorageT* d_bi, StorageT* d_ao, StorageT* d_bo,
                            int L, int D, float* d_sc) {
    chunked_scan<Blelloch, StorageT>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
}

template <typename StorageT>
static void launch_hillis_steele(StorageT* d_ai, StorageT* d_bi, StorageT* d_ao, StorageT* d_bo,
                                 int L, int D, float* d_sc) {
    chunked_scan<HillisSteele, StorageT>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
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

static bool load_inputs(const std::string& path, float* a, float* b, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    bool ok = std::fread(a, sizeof(float), n, f) == n && std::fread(b, sizeof(float), n, f) == n;
    std::fclose(f);
    return ok;
}

static bool load_ref(const std::string& path, float* x, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    bool ok = std::fread(x, sizeof(float), n, f) == n;
    std::fclose(f);
    return ok;
}

static void to_dim_major(const float* src, float* dst, int L, int D) {
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            dst[d * L + t] = src[t * D + d];
        }
    }
}

static bool is_known_kernel(const std::string& kernel_name) {
    return kernel_name == "warp_shuffle" || kernel_name == "blelloch" ||
           kernel_name == "hillis_steele" || kernel_name == "cub";
}

static bool is_cub_kernel(const std::string& kernel_name) {
    return kernel_name == "cub";
}

static void print_usage(const char* argv0) {
    printf("Usage:\n");
    printf("  %s --kernel <warp_shuffle|blelloch|hillis_steele|cub> --D <dim> --L <length> [options]\n", argv0);
    printf("\nOptions:\n");
    printf("  --input_dir <path>    Default: ../SyntheticData/inputs\n");
    printf("  --ref_dir <path>      Default: ../SequentialBaseline/SequentialData\n");
    printf("  --warmup <int>        Default: %d\n", kDefaultWarmup);
    printf("  --repeat <int>        Default: %d\n", kDefaultRepeat);
    printf("  --dtype <name>        One of: fp32, bf16, fp16\n");
    printf("  --csv_append <path>   Append timing row to CSV file\n");
    printf("  --skip_check          Skip correctness check\n");
    printf("  --no_print            Suppress normal stdout output\n");
    printf("  --help                Show this message\n");
}

static bool parse_args(int argc, char* argv[], Options* options) {
    if (!options) return false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else if (std::strcmp(arg, "--kernel") == 0) {
            if (i + 1 >= argc) return false;
            options->kernel = argv[++i];
        } else if (std::strcmp(arg, "--D") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->D)) return false;
        } else if (std::strcmp(arg, "--L") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->L)) return false;
        } else if (std::strcmp(arg, "--input_dir") == 0) {
            if (i + 1 >= argc) return false;
            options->input_dir = argv[++i];
        } else if (std::strcmp(arg, "--ref_dir") == 0) {
            if (i + 1 >= argc) return false;
            options->ref_dir = argv[++i];
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->warmup)) return false;
        } else if (std::strcmp(arg, "--repeat") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->repeat)) return false;
        } else if (std::strcmp(arg, "--dtype") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_scan_dtype(argv[++i], &options->dtype)) return false;
        } else if (std::strcmp(arg, "--csv_append") == 0) {
            if (i + 1 >= argc) return false;
            options->csv_append = argv[++i];
        } else if (std::strcmp(arg, "--no_print") == 0) {
            options->no_print = true;
        } else if (std::strcmp(arg, "--skip_check") == 0) {
            options->skip_check = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return false;
        }
    }

    if (options->kernel.empty() || options->D <= 0 || options->L <= 0 || options->warmup < 0 || options->repeat <= 0) {
        return false;
    }
    if (!is_known_kernel(options->kernel)) {
        fprintf(stderr, "Unknown kernel '%s'\n", options->kernel.c_str());
        return false;
    }
    return true;
}

template <typename StorageT>
static BenchmarkResult run_custom_kernel(const Options& opt,
                                         KernelFnT<StorageT> kernel_fn,
                                         const std::vector<float>& a_dm_fp32,
                                         const std::vector<float>& b_dm_fp32,
                                         const std::vector<float>& ref) {
    const size_t n = static_cast<size_t>(opt.D) * opt.L;
    std::vector<StorageT> a_dm(n), b_dm(n);
    convert_float_buffer_to_storage(a_dm_fp32.data(), a_dm.data(), n);
    convert_float_buffer_to_storage(b_dm_fp32.data(), b_dm.data(), n);

    StorageT* d_ai = nullptr;
    StorageT* d_bi = nullptr;
    StorageT* d_ao = nullptr;
    StorageT* d_bo = nullptr;
    float* d_sc = nullptr;

    cudaMalloc(&d_ai, n * sizeof(StorageT));
    cudaMalloc(&d_bi, n * sizeof(StorageT));
    cudaMalloc(&d_ao, n * sizeof(StorageT));
    cudaMalloc(&d_bo, n * sizeof(StorageT));
    cudaMalloc(&d_sc, 2ULL * opt.D * opt.L * sizeof(float));

    const BenchmarkResult result = benchmark_kernel<StorageT>(
        kernel_fn, d_ai, d_bi, d_ao, d_bo, d_sc,
        a_dm.data(), b_dm.data(), ref.data(), opt.L, opt.D,
        opt.warmup, opt.repeat, opt.skip_check);

    cudaFree(d_ai);
    cudaFree(d_bi);
    cudaFree(d_ao);
    cudaFree(d_bo);
    cudaFree(d_sc);
    return result;
}

template <typename StorageT>
static BenchmarkResult run_cub_kernel(const Options& opt,
                                      const std::vector<float>& a_dm_fp32,
                                      const std::vector<float>& b_dm_fp32,
                                      const std::vector<float>& ref) {
    const size_t n = static_cast<size_t>(opt.D) * opt.L;
    std::vector<StorageT> a_dm(n), b_dm(n);
    convert_float_buffer_to_storage(a_dm_fp32.data(), a_dm.data(), n);
    convert_float_buffer_to_storage(b_dm_fp32.data(), b_dm.data(), n);

    std::vector<PairABT<StorageT>> h_in(n);
    std::vector<int> h_keys(n);
    pack_pairs(a_dm.data(), b_dm.data(), h_in.data(), n);
    for (int d = 0; d < opt.D; ++d) {
        for (int t = 0; t < opt.L; ++t) h_keys[static_cast<size_t>(d) * opt.L + t] = d;
    }

    PairABT<StorageT>* d_in = nullptr;
    PairABT<StorageT>* d_out = nullptr;
    int* d_keys = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaMalloc(&d_in, n * sizeof(PairABT<StorageT>));
    cudaMalloc(&d_out, n * sizeof(PairABT<StorageT>));
    cudaMalloc(&d_keys, n * sizeof(int));
    cudaMemcpy(d_keys, h_keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_storage_bytes,
        d_keys, d_in, d_out,
        PairABScanOpT<StorageT>{}, static_cast<int>(n));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    const BenchmarkResult result = benchmark_cub_kernel<StorageT>(
        d_in, d_out, d_keys, d_temp_storage, temp_storage_bytes,
        h_in.data(), ref.data(), opt.L, opt.D,
        opt.warmup, opt.repeat, opt.skip_check);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_keys);
    cudaFree(d_temp_storage);
    return result;
}

template <typename StorageT>
static bool run_typed(const Options& opt,
                      const std::vector<float>& a_dm_fp32,
                      const std::vector<float>& b_dm_fp32,
                      const std::vector<float>& ref,
                      BenchmarkResult* result) {
    if (opt.kernel == "warp_shuffle") {
        *result = run_custom_kernel<StorageT>(opt, launch_warp_shuffle<StorageT>, a_dm_fp32, b_dm_fp32, ref);
        return true;
    }
    if (opt.kernel == "blelloch") {
        *result = run_custom_kernel<StorageT>(opt, launch_blelloch<StorageT>, a_dm_fp32, b_dm_fp32, ref);
        return true;
    }
    if (opt.kernel == "hillis_steele") {
        *result = run_custom_kernel<StorageT>(opt, launch_hillis_steele<StorageT>, a_dm_fp32, b_dm_fp32, ref);
        return true;
    }
    if (is_cub_kernel(opt.kernel)) {
        *result = run_cub_kernel<StorageT>(opt, a_dm_fp32, b_dm_fp32, ref);
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        print_usage(argv[0]);
        return 1;
    }

    const size_t n = static_cast<size_t>(kBatchSize) * opt.L * opt.D;

    std::string input_path = opt.input_dir + "/input_B" + std::to_string(kBatchSize) +
                             "_L" + std::to_string(opt.L) +
                             "_D" + std::to_string(opt.D) + ".bin";
    std::string ref_path = opt.ref_dir + "/ref_B" + std::to_string(kBatchSize) +
                           "_L" + std::to_string(opt.L) +
                           "_D" + std::to_string(opt.D) + ".bin";

    std::vector<float> a_tm(n), b_tm(n), ref(n);
    if (!load_inputs(input_path, a_tm.data(), b_tm.data(), n)) {
        fprintf(stderr, "Failed to load inputs: %s\n", input_path.c_str());
        return 1;
    }
    if (!load_ref(ref_path, ref.data(), n)) {
        fprintf(stderr, "Failed to load reference: %s\n", ref_path.c_str());
        return 1;
    }

    std::vector<float> a_dm(n), b_dm(n);
    to_dim_major(a_tm.data(), a_dm.data(), opt.L, opt.D);
    to_dim_major(b_tm.data(), b_dm.data(), opt.L, opt.D);

    BenchmarkResult result;
    bool ok = false;
    switch (opt.dtype) {
        case ScanDType::kFP32:
            ok = run_typed<float>(opt, a_dm, b_dm, ref, &result);
            break;
        case ScanDType::kBF16:
            ok = run_typed<__nv_bfloat16>(opt, a_dm, b_dm, ref, &result);
            break;
        case ScanDType::kFP16:
            ok = run_typed<__half>(opt, a_dm, b_dm, ref, &result);
            break;
    }
    if (!ok) return 1;

    if (!opt.csv_append.empty()) {
        if (!append_csv_row(opt.csv_append, opt.kernel, scan_dtype_name(opt.dtype), opt.D, opt.L,
                            result, opt.warmup, opt.repeat, kCsvHeader)) {
            return 1;
        }
    }

    if (!opt.no_print) {
        print_result(opt.kernel.c_str(), scan_dtype_name(opt.dtype), opt.D, opt.L, result);
    }

    return result.correct ? 0 : 2;
}
