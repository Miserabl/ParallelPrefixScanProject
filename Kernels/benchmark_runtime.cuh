#ifndef BENCHMARK_RUNTIME_CUH
#define BENCHMARK_RUNTIME_CUH

#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <vector>

#include <cub/device/device_scan.cuh>

#include "benchmark_stats.cuh"
#include "common.cuh"

struct BenchmarkResult {
    TimingSummary summary;
    bool correct = false;
};

template <typename StorageT>
using KernelFnT = void (*)(StorageT*, StorageT*, StorageT*, StorageT*, int, int, float*);

template <typename StorageT>
struct PairABT {
    StorageT a;
    StorageT b;
};

template <typename StorageT>
struct PairABScanOpT {
    __host__ __device__ PairABT<StorageT> operator()(const PairABT<StorageT>& left,
                                                     const PairABT<StorageT>& right) const {
        const float left_a = storage_to_float(left.a);
        const float left_b = storage_to_float(left.b);
        const float right_a = storage_to_float(right.a);
        const float right_b = storage_to_float(right.b);
        PairABT<StorageT> out;
        out.a = float_to_storage<StorageT>(right_a * left_a);
        out.b = float_to_storage<StorageT>(right_a * left_b + right_b);
        return out;
    }
};

template <typename StorageT>
inline void convert_float_buffer_to_storage(const float* src, StorageT* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = float_to_storage<StorageT>(src[i]);
}

template <typename StorageT>
inline float default_tolerance();

template <>
inline float default_tolerance<float>() {
    return 1e-3f;
}

template <>
inline float default_tolerance<__nv_bfloat16>() {
    return 2e-2f;
}

template <>
inline float default_tolerance<__half>() {
    return 5e-3f;
}

template <typename StorageT>
inline void pack_pairs(const StorageT* a_dm, const StorageT* b_dm, PairABT<StorageT>* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i].a = a_dm[i];
        dst[i].b = b_dm[i];
    }
}

template <typename StorageT>
inline void extract_b(const PairABT<StorageT>* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = storage_to_float(src[i].b);
}

template <typename StorageT>
inline bool check_output_storage(const StorageT* b_out_dm, const float* ref, int L, int D,
                                 float tol = default_tolerance<StorageT>()) {
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            const float gpu = storage_to_float(b_out_dm[d * L + t]);
            const float cpu = ref[t * D + d];
            const float err = fabsf(gpu - cpu);
            const float rel = err / fmaxf(fabsf(cpu), 1e-6f);
            if (err > tol && rel > tol) {
                printf("  MISMATCH t=%d d=%d gpu=%.6f ref=%.6f abserr=%.2e relerr=%.2e\n",
                       t, d, gpu, cpu, err, rel);
                return false;
            }
        }
    }
    return true;
}

template <typename StorageT>
inline BenchmarkResult benchmark_kernel(
    KernelFnT<StorageT> fn,
    StorageT* d_ai, StorageT* d_bi, StorageT* d_ao, StorageT* d_bo, float* d_sc,
    const StorageT* h_a_dm, const StorageT* h_b_dm, const float* ref,
    int L, int D, int warmup, int repeat, bool skip_check = false)
{
    const size_t n = static_cast<size_t>(D) * L;
    BenchmarkResult result;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        cudaMemcpy(d_ai, h_a_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi, h_b_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
        fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
    }
    cudaDeviceSynchronize();

    result.correct = true;
    if (!skip_check) {
        if (warmup == 0) {
            cudaMemcpy(d_ai, h_a_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bi, h_b_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
            fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
            cudaDeviceSynchronize();
        }
        std::vector<StorageT> h_bo(n);
        cudaMemcpy(h_bo.data(), d_bo, n * sizeof(StorageT), cudaMemcpyDeviceToHost);
        result.correct = check_output_storage(h_bo.data(), ref, L, D);
    }

    std::vector<float> samples;
    samples.reserve(repeat);
    for (int r = 0; r < repeat; ++r) {
        cudaMemcpy(d_ai, h_a_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi, h_b_dm, n * sizeof(StorageT), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        samples.push_back(elapsed);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    result.summary = summarize_samples_ms(samples);
    return result;
}

template <typename StorageT>
inline BenchmarkResult benchmark_cub_kernel(
    PairABT<StorageT>* d_in, PairABT<StorageT>* d_out,
    int* d_keys,
    void* d_temp_storage, size_t temp_storage_bytes,
    const PairABT<StorageT>* h_in, const float* ref,
    int L, int D, int warmup, int repeat, bool skip_check = false)
{
    const size_t n = static_cast<size_t>(D) * L;
    BenchmarkResult result;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        cudaMemcpy(d_in, h_in, n * sizeof(PairABT<StorageT>), cudaMemcpyHostToDevice);
        size_t temp_bytes = temp_storage_bytes;
        cub::DeviceScan::InclusiveScanByKey(
            d_temp_storage, temp_bytes,
            d_keys, d_in, d_out,
            PairABScanOpT<StorageT>{}, static_cast<int>(n));
    }
    cudaDeviceSynchronize();

    result.correct = true;
    if (!skip_check) {
        if (warmup == 0) {
            cudaMemcpy(d_in, h_in, n * sizeof(PairABT<StorageT>), cudaMemcpyHostToDevice);
            size_t temp_bytes = temp_storage_bytes;
            cub::DeviceScan::InclusiveScanByKey(
                d_temp_storage, temp_bytes,
                d_keys, d_in, d_out,
                PairABScanOpT<StorageT>{}, static_cast<int>(n));
            cudaDeviceSynchronize();
        }
        std::vector<PairABT<StorageT>> h_out(n);
        std::vector<float> h_bo(n);
        cudaMemcpy(h_out.data(), d_out, n * sizeof(PairABT<StorageT>), cudaMemcpyDeviceToHost);
        extract_b(h_out.data(), h_bo.data(), n);
        result.correct = check_output_storage(h_bo.data(), ref, L, D);
    }

    std::vector<float> samples;
    samples.reserve(repeat);
    for (int r = 0; r < repeat; ++r) {
        cudaMemcpy(d_in, h_in, n * sizeof(PairABT<StorageT>), cudaMemcpyHostToDevice);
        size_t temp_bytes = temp_storage_bytes;
        cudaEventRecord(start);
        cub::DeviceScan::InclusiveScanByKey(
            d_temp_storage, temp_bytes,
            d_keys, d_in, d_out,
            PairABScanOpT<StorageT>{}, static_cast<int>(n));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        samples.push_back(elapsed);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    result.summary = summarize_samples_ms(samples);
    return result;
}

inline void write_csv_row(FILE* csv, const char* kernel_name, const char* dtype_name,
                          int D, int L, const BenchmarkResult& result,
                          int warmup, int repeat) {
    if (!csv) return;

    const double throughput_gbs = logical_payload_throughput_gbs(static_cast<size_t>(D) * L, result.summary.median_ms);
    fprintf(csv,
            "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%s,%d,%d,%s,%s\n",
            kernel_name,
            D,
            L,
            result.summary.median_ms,
            result.summary.median_ms,
            result.summary.p25_ms,
            result.summary.p75_ms,
            result.summary.iqr_ms,
            result.summary.mean_ms,
            result.summary.stddev_ms,
            result.correct ? 1 : 0,
            throughput_gbs,
            "logical_payload",
            warmup,
            repeat,
            dtype_name,
            result.summary.samples_ms.c_str());
    fflush(csv);
}

inline bool append_csv_row(const std::string& csv_path,
                           const std::string& kernel,
                           const char* dtype_name,
                           int D,
                           int L,
                           const BenchmarkResult& result,
                           int warmup,
                           int repeat,
                           const char* csv_header) {
    FILE* f = std::fopen(csv_path.c_str(), "a");
    if (!f) {
        std::perror(csv_path.c_str());
        return false;
    }
    struct stat st;
    const bool need_header = stat(csv_path.c_str(), &st) != 0 || st.st_size == 0;
    if (need_header) {
        std::fputs(csv_header, f);
    }
    write_csv_row(f, kernel.c_str(), dtype_name, D, L, result, warmup, repeat);
    std::fclose(f);
    return true;
}

inline void print_result(const char* kernel_name, const char* dtype_name, int D, int L,
                         const BenchmarkResult& result) {
    const double throughput_gbs = logical_payload_throughput_gbs(static_cast<size_t>(D) * L, result.summary.median_ms);
    printf("  %-20s dtype=%-4s D=%-5d L=%-8d median=%8.4f ms  iqr=%8.4f ms  logical=%8.2f GB/s  %s\n",
           kernel_name,
           dtype_name,
           D,
           L,
           result.summary.median_ms,
           result.summary.iqr_ms,
           throughput_gbs,
           result.correct ? "PASS" : "FAIL");
}

#endif // BENCHMARK_RUNTIME_CUH
