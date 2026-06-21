#!/usr/bin/env python3

import argparse
import csv
import math
import os
import statistics


def parse_int_list(text):
    values = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


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
        "median_ms": p50,
        "p25_ms": p25,
        "p75_ms": p75,
        "iqr_ms": p75 - p25,
        "mean_ms": statistics.fmean(samples),
        "stddev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": ";".join(f"{sample:.6f}" for sample in samples),
    }


def torch_dtype_for_name(torch, dtype_name):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype {dtype_name}")


def append_row(path, row):
    need_header = not os.path.exists(path)
    fieldnames = [
        "component",
        "model",
        "layer_index",
        "dtype",
        "batch",
        "L",
        "median_ms",
        "p25_ms",
        "p75_ms",
        "iqr_ms",
        "mean_ms",
        "stddev_ms",
        "warmup",
        "repeat",
        "samples_ms",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark end-to-end Mamba mixer layer latency")
    parser.add_argument("--pretrained", default="state-spaces/mamba-2.8b")
    parser.add_argument("--layer_index", type=int, default=0)
    parser.add_argument("--lengths", default="1024,2048,4096,8192")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_csv", default="Results/mamba_layer_benchmark.csv")
    args = parser.parse_args()

    import torch
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch_dtype_for_name(torch, args.dtype)
    lengths = parse_int_list(args.lengths)

    model = MambaLMHeadModel.from_pretrained(args.pretrained, device=device, dtype=dtype)
    model.eval()
    mixer = model.backbone.layers[args.layer_index].mixer

    for length in lengths:
        input_ids = torch.randint(0, model.config.vocab_size, (args.batch, length), device=device)
        with torch.inference_mode():
            hidden_states = model.backbone.embedding(input_ids)

        for _ in range(args.warmup):
            with torch.inference_mode():
                mixer(hidden_states)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(args.repeat):
            start.record()
            with torch.inference_mode():
                mixer(hidden_states)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))

        stats = summarize(samples)
        append_row(args.output_csv, {
            "component": "mixer_layer",
            "model": args.pretrained,
            "layer_index": args.layer_index,
            "dtype": args.dtype,
            "batch": args.batch,
            "L": length,
            "median_ms": f"{stats['median_ms']:.6f}",
            "p25_ms": f"{stats['p25_ms']:.6f}",
            "p75_ms": f"{stats['p75_ms']:.6f}",
            "iqr_ms": f"{stats['iqr_ms']:.6f}",
            "mean_ms": f"{stats['mean_ms']:.6f}",
            "stddev_ms": f"{stats['stddev_ms']:.6f}",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "samples_ms": stats["samples_ms"],
        })
        print(
            f"component=mixer_layer model={args.pretrained} layer={args.layer_index} dtype={args.dtype} "
            f"L={length} median_ms={stats['median_ms']:.4f} iqr_ms={stats['iqr_ms']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
