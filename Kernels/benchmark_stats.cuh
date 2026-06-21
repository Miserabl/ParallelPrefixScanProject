#ifndef BENCHMARK_STATS_CUH
#define BENCHMARK_STATS_CUH

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

struct TimingSummary {
    float median_ms = -1.0f;
    float p25_ms = -1.0f;
    float p75_ms = -1.0f;
    float iqr_ms = -1.0f;
    float mean_ms = -1.0f;
    float stddev_ms = -1.0f;
    std::string samples_ms;
};

static inline std::string serialize_samples_ms(const std::vector<float>& values) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(6);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ';';
        out << values[i];
    }
    return out.str();
}

static inline float percentile_from_sorted(const std::vector<float>& sorted_values, double q) {
    if (sorted_values.empty()) return -1.0f;
    if (sorted_values.size() == 1) return sorted_values[0];

    const double pos = q * static_cast<double>(sorted_values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, sorted_values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac);
}

static inline TimingSummary summarize_samples_ms(const std::vector<float>& values) {
    TimingSummary summary;
    if (values.empty()) return summary;

    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    double sum = 0.0;
    for (float value : values) sum += static_cast<double>(value);
    const double mean = sum / static_cast<double>(values.size());

    double sq_sum = 0.0;
    for (float value : values) {
        const double delta = static_cast<double>(value) - mean;
        sq_sum += delta * delta;
    }

    summary.median_ms = percentile_from_sorted(sorted_values, 0.5);
    summary.p25_ms = percentile_from_sorted(sorted_values, 0.25);
    summary.p75_ms = percentile_from_sorted(sorted_values, 0.75);
    summary.iqr_ms = summary.p75_ms - summary.p25_ms;
    summary.mean_ms = static_cast<float>(mean);
    summary.stddev_ms = static_cast<float>(std::sqrt(sq_sum / static_cast<double>(values.size())));
    summary.samples_ms = serialize_samples_ms(values);
    return summary;
}

static inline double logical_payload_throughput_gbs(size_t n, float median_ms) {
    if (median_ms <= 0.0f) return 0.0;
    const double logical_bytes = 3.0 * static_cast<double>(n) * sizeof(float);
    return logical_bytes / (static_cast<double>(median_ms) * 1e-3) * 1e-9;
}

#endif // BENCHMARK_STATS_CUH
