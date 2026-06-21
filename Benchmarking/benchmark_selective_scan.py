#!/usr/bin/env python3

import argparse
import csv
import math
import os
import statistics
import sys


KERNEL_NAME = "selective_scan_cuda"
DEFAULT_DIMS = [1, 16, 64, 256, 512]
DEFAULT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def torch_dtype_for_name(torch, dtype_name):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype {dtype_name}")


def tolerance_for_dtype(dtype_name):
    if dtype_name == "fp32":
        return 1e-3, 1e-3
    if dtype_name == "bf16":
        return 1e-2, 1e-2
    if dtype_name == "fp16":
        return 5e-3, 5e-3
    raise ValueError(f"unsupported dtype {dtype_name}")


def parse_list(text):
    if not text:
        return []
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def load_inputs(path, n):
    with open(path, "rb") as f:
        payload = f.read()
    expected_bytes = 2 * n * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes in {path}, found {len(payload)}")
    return payload


def load_ref(path, n):
    with open(path, "rb") as f:
        payload = f.read()
    expected_bytes = n * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes in {path}, found {len(payload)}")
    return payload


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(samples):
    sorted_samples = sorted(samples)
    p25 = percentile(sorted_samples, 0.25)
    p50 = percentile(sorted_samples, 0.50)
    p75 = percentile(sorted_samples, 0.75)
    return {
        "time_ms": p50,
        "median_ms": p50,
        "p25_ms": p25,
        "p75_ms": p75,
        "iqr_ms": p75 - p25,
        "mean_ms": statistics.fmean(samples),
        "stddev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": ";".join(f"{sample:.6f}" for sample in samples),
    }


def logical_payload_throughput_gbs(n, median_ms):
    if not median_ms or median_ms <= 0.0:
        return 0.0
    logical_bytes = 3.0 * n * 4
    return logical_bytes / (median_ms * 1e-3) * 1e-9


def append_row(path, row):
    need_header = not os.path.exists(path)
    fieldnames = [
        "kernel",
        "D",
        "L",
        "time_ms",
        "median_ms",
        "p25_ms",
        "p75_ms",
        "iqr_ms",
        "mean_ms",
        "stddev_ms",
        "correct",
        "throughput_GB_s",
        "throughput_metric",
        "warmup",
        "repeat",
        "dtype",
        "samples_ms",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def import_selective_scan():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return None, f"torch import failed: {exc}"

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn, None
    except Exception as exc:
        return None, f"mamba_ssm selective_scan import failed: {exc}"


def run_case(selective_scan_fn, torch, input_dir, ref_dir, D, L, warmup, repeat, dtype_name):
    n = D * L
    input_path = os.path.join(input_dir, f"input_B1_L{L}_D{D}.bin")
    ref_path = os.path.join(ref_dir, f"ref_B1_L{L}_D{D}.bin")

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(ref_path)

    raw_inputs = load_inputs(input_path, n)
    raw_ref = load_ref(ref_path, n)

    ab = torch.frombuffer(memoryview(raw_inputs), dtype=torch.float32).clone()
    ref = torch.frombuffer(memoryview(raw_ref), dtype=torch.float32).clone()
    a_tm = ab[:n].reshape(L, D)
    b_tm = ab[n:].reshape(L, D)
    ref_tm = ref.reshape(L, D)

    a_dm = a_tm.transpose(0, 1).contiguous()
    b_dm = b_tm.transpose(0, 1).contiguous()
    ref_dm = ref_tm.transpose(0, 1).contiguous().cuda()

    torch_dtype = torch_dtype_for_name(torch, dtype_name)
    atol, rtol = tolerance_for_dtype(dtype_name)

    u = torch.ones((1, D, L), device="cuda", dtype=torch_dtype)
    delta = torch.log(a_dm).unsqueeze(0).to(device="cuda", dtype=torch_dtype)
    A = torch.ones((D, 1), device="cuda", dtype=torch.float32)
    delta_fp32 = torch.log(a_dm).unsqueeze(0)
    B = (b_dm / delta_fp32.squeeze(0)).unsqueeze(0).unsqueeze(2).contiguous().to(device="cuda", dtype=torch_dtype)
    C = torch.ones((1, D, 1, L), device="cuda", dtype=torch_dtype)

    out = None
    for _ in range(warmup):
        out = selective_scan_fn(u, delta, A, B, C, delta_softplus=False)
    torch.cuda.synchronize()

    if out is None:
        out = selective_scan_fn(u, delta, A, B, C, delta_softplus=False)
        torch.cuda.synchronize()

    got = out.squeeze(0)
    correct = torch.allclose(got.float(), ref_dm.float(), atol=atol, rtol=rtol)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeat):
        start.record()
        selective_scan_fn(u, delta, A, B, C, delta_softplus=False)
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))

    summary = summarize(samples)
    summary["correct"] = 1 if correct else 0
    summary["throughput_GB_s"] = logical_payload_throughput_gbs(n, summary["median_ms"])
    summary["throughput_metric"] = "logical_payload"
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark mamba selective_scan_cuda against repo inputs")
    parser.add_argument("--input_dir", default="../SyntheticData/inputs")
    parser.add_argument("--ref_dir", default="../SequentialBaseline/SequentialData")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dims", default="")
    parser.add_argument("--lengths", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--no_print", action="store_true")
    parser.add_argument("--strict_import", action="store_true")
    args = parser.parse_args()

    selective_scan_fn, import_error = import_selective_scan()
    if selective_scan_fn is None:
        message = f"Skipping {KERNEL_NAME}: {import_error}"
        print(message, file=sys.stderr)
        return 1 if args.strict_import else 0

    import torch

    if not torch.cuda.is_available():
        print("Skipping selective_scan_cuda: CUDA is not available", file=sys.stderr)
        return 1 if args.strict_import else 0

    dims = parse_list(args.dims) or DEFAULT_DIMS
    lengths = parse_list(args.lengths) or DEFAULT_LENGTHS

    for D in dims:
        for L in lengths:
            result = run_case(selective_scan_fn, torch, args.input_dir, args.ref_dir, D, L, args.warmup, args.repeat, args.dtype)
            row = {
                "kernel": KERNEL_NAME,
                "D": D,
                "L": L,
                "time_ms": f"{result['time_ms']:.6f}",
                "median_ms": f"{result['median_ms']:.6f}",
                "p25_ms": f"{result['p25_ms']:.6f}",
                "p75_ms": f"{result['p75_ms']:.6f}",
                "iqr_ms": f"{result['iqr_ms']:.6f}",
                "mean_ms": f"{result['mean_ms']:.6f}",
                "stddev_ms": f"{result['stddev_ms']:.6f}",
                "correct": result["correct"],
                "throughput_GB_s": f"{result['throughput_GB_s']:.6f}",
                "throughput_metric": result["throughput_metric"],
                "warmup": args.warmup,
                "repeat": args.repeat,
                "dtype": args.dtype,
                "samples_ms": result["samples_ms"],
            }
            append_row(args.output_csv, row)
            if not args.no_print:
                print(
                    f"kernel={KERNEL_NAME:<20} D={D:<4d} L={L:<7d} "
                    f"dtype={args.dtype:<4s} median_ms={result['median_ms']:<10.4f} iqr_ms={result['iqr_ms']:<10.4f} "
                    f"GB/s={result['throughput_GB_s']:<10.3f} correct={'PASS' if result['correct'] else 'FAIL'}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
