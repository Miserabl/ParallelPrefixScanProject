#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict


KERNEL_ORDER = {
    "warp_shuffle": 0,
    "blelloch": 1,
    "hillis_steele": 2,
    "cub": 3,
    "selective_scan_cuda": 4,
}


def parse_float(text):
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    lower = value.lower()
    if lower in {"nan", "n/a", "na", "none", "-", "--"}:
        return None
    value = value.replace(",", "").replace("%", "")
    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def fmt(value, decimals=4):
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


def safe_div(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def chunk_size_for_d(d):
    _ = d
    return 1024


def parse_timing_csv(path):
    timing = {}
    if not os.path.exists(path):
        return timing
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kernel = row["kernel"].strip()
                d = int(row["D"])
                l = int(row["L"])
            except (KeyError, ValueError):
                continue
            dtype = (row.get("dtype") or "fp32").strip() or "fp32"
            key = (kernel, dtype, d, l)
            timing[key] = {
                "time_ms": parse_float(row.get("median_ms")) or parse_float(row.get("time_ms")),
                "median_ms": parse_float(row.get("median_ms")) or parse_float(row.get("time_ms")),
                "p25_ms": parse_float(row.get("p25_ms")),
                "p75_ms": parse_float(row.get("p75_ms")),
                "iqr_ms": parse_float(row.get("iqr_ms")),
                "mean_ms": parse_float(row.get("mean_ms")),
                "stddev_ms": parse_float(row.get("stddev_ms")),
                "correct": int(parse_float(row.get("correct")) or 0),
                "throughput_GB_s": parse_float(row.get("throughput_GB_s")),
                "throughput_metric": (row.get("throughput_metric") or "").strip() or None,
                "warmup": int(parse_float(row.get("warmup")) or 0),
                "repeat": int(parse_float(row.get("repeat")) or 0),
                "dtype": dtype,
                "samples_ms": (row.get("samples_ms") or "").strip() or None,
            }
    return timing


def parse_ncu_filename(path):
    name = os.path.basename(path)
    match = re.match(r"^(?P<kernel>.+)_(?P<dtype>fp32|bf16|fp16)_D(?P<D>\d+)_L(?P<L>\d+)\.csv$", name)
    if not match:
        match = re.match(r"^(?P<kernel>.+)_D(?P<D>\d+)_L(?P<L>\d+)\.csv$", name)
        if not match:
            return None
        return match.group("kernel"), "fp32", int(match.group("D")), int(match.group("L"))
    return match.group("kernel"), match.group("dtype"), int(match.group("D")), int(match.group("L"))


def parse_ncu_raw_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header_idx = None
    header = None
    header_kind = None

    for idx, row in enumerate(rows):
        if not row:
            continue
        cells = [c.strip() for c in row]
        if not any(cells):
            continue

        lower = {c.lower() for c in cells if c}

        if "metric name" in lower and ("metric value" in lower or "value" in lower):
            header_idx = idx
            header = cells
            header_kind = "legacy"
            break

        has_launch_id = "id" in lower or "kernel id" in lower
        has_kernel_name = "kernel name" in lower or "kernel name / demangled" in lower
        if has_launch_id and has_kernel_name:
            header_idx = idx
            header = cells
            header_kind = "wide"
            break

    if header_idx is None or header is None or header_kind is None:
        return []

    # Legacy long format: one metric per row.
    if header_kind == "legacy":
        launches = {}
        for row in rows[header_idx + 1:]:
            if not row:
                continue

            cells = [c.strip() for c in row]
            if not any(cells):
                continue
            if cells == header:
                continue

            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[:len(header)]

            record = {header[i]: cells[i] for i in range(len(header))}
            lower = {k.lower(): v for k, v in record.items()}

            metric_name = lower.get("metric name") or lower.get("metric name / label")
            metric_value_text = lower.get("metric value") or lower.get("value")
            if not metric_name or metric_value_text is None:
                continue

            metric_value = parse_float(metric_value_text)
            if metric_value is None:
                continue

            launch_id = lower.get("id") or lower.get("kernel id") or "unknown"
            kernel_name = lower.get("kernel name") or lower.get("kernel name / demangled") or "unknown"
            launch_key = f"{launch_id}|{kernel_name}"

            if launch_key not in launches:
                launches[launch_key] = {
                    "__launch_id__": launch_id,
                    "__kernel_name__": kernel_name,
                }
            launches[launch_key][metric_name] = metric_value

        return list(launches.values())

    # Nsight Compute wide format: one launch per row, metrics as columns.
    launches = []
    metadata_cols = {
        "id",
        "process id",
        "process name",
        "host name",
        "kernel id",
        "kernel name",
        "kernel name / demangled",
        "context",
        "stream",
        "device",
        "green context id",
        "section name",
        "section id",
    }

    key_map = {}
    for i, name in enumerate(header):
        key = name.lower().strip()
        if key:
            key_map[key] = i

    id_idx = key_map.get("id")
    if id_idx is None:
        id_idx = key_map.get("kernel id")

    name_idx = key_map.get("kernel name")
    if name_idx is None:
        name_idx = key_map.get("kernel name / demangled")

    if id_idx is None:
        return []

    for row in rows[header_idx + 1:]:
        if not row:
            continue

        cells = [c.strip() for c in row]
        if not any(cells):
            continue

        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]

        launch_id_val = parse_float(cells[id_idx])

        # Skip units row and any malformed/non-launch rows.
        if launch_id_val is None:
            continue

        if float(launch_id_val).is_integer():
            launch_id = str(int(launch_id_val))
        else:
            launch_id = str(launch_id_val)

        kernel_name = "unknown"
        if name_idx is not None:
            kernel_name = cells[name_idx].strip() or "unknown"

        launch = {
            "__launch_id__": launch_id,
            "__kernel_name__": kernel_name,
        }

        numeric_metrics = 0
        for i, col in enumerate(header):
            col_name = col.strip()
            if not col_name:
                continue
            if col_name.lower() in metadata_cols:
                continue

            value = parse_float(cells[i])
            if value is None:
                continue

            launch[col_name] = value
            numeric_metrics += 1

        if numeric_metrics > 0:
            launches.append(launch)

    return launches


def find_first_exact(metric_names, candidates):
    for name in candidates:
        if name in metric_names:
            return name
    return None


def find_first_contains(metric_names, required_tokens):
    for name in sorted(metric_names):
        if all(token in name for token in required_tokens):
            return name
    return None


def find_first_prefix(metric_names, prefixes, exclude_tokens=None):
    exclude_tokens = exclude_tokens or []
    for name in sorted(metric_names):
        if any(name.startswith(prefix) for prefix in prefixes):
            if any(token in name for token in exclude_tokens):
                continue
            return name
    return None


def classify_phase(kernel_name):
    name = (kernel_name or "").lower()
    if "phase1_scan" in name:
        return "phase1"
    if "phase2_scan_totals" in name:
        return "phase2"
    if "phase3_propagate" in name:
        return "phase3"
    return "other"


def block_scan_combine_count(kernel, n):
    if n <= 0:
        return 0

    if kernel == "blelloch":
        return 3 * n - 2

    if kernel == "hillis_steele":
        total = 0
        offset = 1
        while offset < n:
            total += n - offset
            offset <<= 1
        return total

    if kernel == "warp_shuffle":
        total = 0
        num_warps = (n + 31) // 32

        for wid in range(num_warps):
            active = min(32, n - 32 * wid)
            offset = 1
            while offset < 32:
                if active > offset:
                    total += active - offset
                offset <<= 1

        offset = 1
        while offset < 32:
            if num_warps > offset:
                total += num_warps - offset
            offset <<= 1

        if num_warps > 1:
            total += n - min(32, n)

        return total

    return 0


def model_phase_metrics(kernel, d, l):
    if kernel not in {"warp_shuffle", "blelloch", "hillis_steele"}:
        return {}

    chunk = chunk_size_for_d(d)
    if l <= 0:
        return {}

    num_chunks = (l + chunk - 1) // chunk
    chunk_sizes = [max(0, min(chunk, l - cid * chunk)) for cid in range(num_chunks)]

    phase1_combines = d * sum(block_scan_combine_count(kernel, n) for n in chunk_sizes)
    phase2_combines = d * block_scan_combine_count(kernel, num_chunks)
    propagated_elements = d * sum(chunk_sizes[1:])
    phase3_combines = propagated_elements

    return {
        "phase1": {
            "dram_bytes": float(d * (16 * sum(chunk_sizes) + 8 * num_chunks)),
            "flops": float(3 * phase1_combines),
        },
        "phase2": {
            "dram_bytes": float(16 * d * num_chunks),
            "flops": float(3 * phase2_combines),
        },
        "phase3": {
            "dram_bytes": float(24 * propagated_elements),
            "flops": float(3 * phase3_combines),
        },
    }


def model_run_metrics(kernel, d, l):
    phases = model_phase_metrics(kernel, d, l)
    if not phases:
        return {}

    dram_bytes = 0.0
    flops = 0.0
    for phase_data in phases.values():
        dram_bytes += phase_data.get("dram_bytes", 0.0)
        flops += phase_data.get("flops", 0.0)
    return {
        "dram_bytes": dram_bytes,
        "flops": flops,
    }


def select_metric_names(metric_names):
    duration = find_first_exact(
        metric_names,
        [
            "gpu__time_duration.sum",
            "gpu__time_duration.avg",
            "gpu__time_duration",
        ],
    )
    if duration is None:
        duration = find_first_contains(metric_names, ["time_duration", "gpu__"])

    return {
        "duration": duration,
        "sm_util": find_first_exact(
            metric_names,
            [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__throughput.avg.pct_of_peak_sustained_active",
                "sm__throughput",
            ],
        ) or find_first_contains(metric_names, ["sm__throughput", "pct_of_peak"]),
        "dram_util": find_first_exact(
            metric_names,
            [
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_active",
                "dram__throughput",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_active",
                "gpu__dram_throughput",
            ],
        ) or find_first_contains(metric_names, ["dram_throughput", "pct_of_peak"]),
        "occ": find_first_exact(
            metric_names,
            [
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active",
            ],
        ) or find_first_contains(metric_names, ["sm__warps_active", "pct_of_peak"]),
        "warp_ratio": find_first_exact(
            metric_names,
            [
                "smsp__thread_inst_executed_per_inst_executed.ratio",
                "smsp__thread_inst_executed_per_inst_executed",
            ],
        ) or find_first_contains(metric_names, ["thread_inst_executed_per_inst_executed"]),
        "dram_bytes": find_first_exact(metric_names, ["dram__bytes.sum", "dram__bytes"])
            or find_first_prefix(metric_names, ["dram__bytes"], exclude_tokens=["read", "write"]),
        "dram_bytes_read": find_first_exact(metric_names, ["dram__bytes_read.sum", "dram__bytes_read"])
            or find_first_prefix(metric_names, ["dram__bytes_read"]),
        "dram_bytes_write": find_first_exact(metric_names, ["dram__bytes_write.sum", "dram__bytes_write"])
            or find_first_prefix(metric_names, ["dram__bytes_write"]),
        "fadd": find_first_exact(metric_names, [
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on",
            "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fadd_pred_on",
        ]) or find_first_contains(metric_names, ["op_fadd", "pred_on"]),
        "fmul": find_first_exact(metric_names, [
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on",
            "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fmul_pred_on",
        ]) or find_first_contains(metric_names, ["op_fmul", "pred_on"]),
        "ffma": find_first_exact(metric_names, [
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_ffma_pred_on",
        ]) or find_first_contains(metric_names, ["op_ffma", "pred_on"]),
        "regs": find_first_exact(metric_names, ["launch__registers_per_thread"])
            or find_first_contains(metric_names, ["launch__registers_per_thread"]),
        "shmem": find_first_exact(metric_names, ["launch__shared_mem_per_block_allocated"])
            or find_first_contains(metric_names, ["launch__shared_mem_per_block"]),
        "block_size": find_first_exact(metric_names, ["launch__block_size"])
            or find_first_contains(metric_names, ["launch__block_size"]),
        "l1_hit": find_first_exact(metric_names, ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"])
            or find_first_contains(metric_names, ["l1tex__", "op_ld", "lookup_hit"]),
        "l1_miss": find_first_exact(metric_names, ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"])
            or find_first_contains(metric_names, ["l1tex__", "op_ld", "lookup_miss"]),
        "l2_hit": find_first_exact(metric_names, ["lts__t_sectors_srcunit_lsu_op_read_lookup_hit.sum"])
            or find_first_contains(metric_names, ["lts__", "op_read", "lookup_hit"]),
        "l2_miss": find_first_exact(metric_names, ["lts__t_sectors_srcunit_lsu_op_read_lookup_miss.sum"])
            or find_first_contains(metric_names, ["lts__", "op_read", "lookup_miss"]),
    }


def aggregate_launches(launches):
    if not launches:
        return {}

    metric_names = set()
    for launch in launches:
        metric_names.update(k for k in launch.keys() if not k.startswith("__"))

    selected = select_metric_names(metric_names)
    duration_metric = selected["duration"]

    def values(name):
        if name is None:
            return [], []
        vals = []
        weights = []
        for launch in launches:
            if name not in launch:
                continue
            vals.append(launch[name])
            if duration_metric and duration_metric in launch and launch[duration_metric] > 0:
                weights.append(launch[duration_metric])
            else:
                weights.append(1.0)
        return vals, weights

    def weighted_avg(name):
        vals, weights = values(name)
        if not vals:
            return None
        wsum = sum(weights)
        if wsum > 0:
            return sum(v * w for v, w in zip(vals, weights)) / wsum
        return sum(vals) / len(vals)

    def metric_sum(name):
        vals, _ = values(name)
        if not vals:
            return None
        return sum(vals)

    def metric_max(name):
        vals, _ = values(name)
        if not vals:
            return None
        return max(vals)

    sm_util_name = selected["sm_util"]
    dram_util_name = selected["dram_util"]
    occ_name = selected["occ"]
    warp_ratio_name = selected["warp_ratio"]

    dram_bytes_name = selected["dram_bytes"]
    dram_read_bytes_name = selected["dram_bytes_read"]
    dram_write_bytes_name = selected["dram_bytes_write"]

    fadd_name = selected["fadd"]
    fmul_name = selected["fmul"]
    ffma_name = selected["ffma"]

    regs_name = selected["regs"]
    shmem_name = selected["shmem"]
    block_size_name = selected["block_size"]

    l1_hit_name = selected["l1_hit"]
    l1_miss_name = selected["l1_miss"]

    l2_hit_name = selected["l2_hit"]
    l2_miss_name = selected["l2_miss"]

    dram_bytes = metric_sum(dram_bytes_name)
    if dram_bytes is None:
        read_bytes = metric_sum(dram_read_bytes_name)
        write_bytes = metric_sum(dram_write_bytes_name)
        if read_bytes is not None or write_bytes is not None:
            dram_bytes = (read_bytes or 0.0) + (write_bytes or 0.0)

    fadd = metric_sum(fadd_name) or 0.0
    fmul = metric_sum(fmul_name) or 0.0
    ffma = metric_sum(ffma_name) or 0.0
    flops = fadd + fmul + 2.0 * ffma
    if fadd == 0.0 and fmul == 0.0 and ffma == 0.0:
        flops = None

    l1_hits = metric_sum(l1_hit_name)
    l1_misses = metric_sum(l1_miss_name)
    l2_hits = metric_sum(l2_hit_name)
    l2_misses = metric_sum(l2_miss_name)

    l1_hit_rate = None
    if l1_hits is not None and l1_misses is not None and (l1_hits + l1_misses) > 0:
        l1_hit_rate = 100.0 * l1_hits / (l1_hits + l1_misses)

    l2_hit_rate = None
    if l2_hits is not None and l2_misses is not None and (l2_hits + l2_misses) > 0:
        l2_hit_rate = 100.0 * l2_hits / (l2_hits + l2_misses)

    warp_threads_per_inst = weighted_avg(warp_ratio_name)
    warp_efficiency = None
    if warp_threads_per_inst is not None:
        warp_efficiency = max(0.0, min(100.0, 100.0 * warp_threads_per_inst / 32.0))

    return {
        "sm_util_pct": weighted_avg(sm_util_name),
        "dram_util_pct": weighted_avg(dram_util_name),
        "achieved_occupancy_pct": weighted_avg(occ_name),
        "warp_threads_per_inst": warp_threads_per_inst,
        "warp_efficiency_pct": warp_efficiency,
        "dram_bytes": dram_bytes,
        "flops": flops,
        "registers_per_thread_peak": metric_max(regs_name),
        "shared_mem_per_block_peak_bytes": metric_max(shmem_name),
        "block_size_peak": metric_max(block_size_name),
        "l1_hit_rate_pct": l1_hit_rate,
        "l2_hit_rate_pct": l2_hit_rate,
    }


def launch_to_metrics_rows(launches, kernel, dtype, d, l):
    if not launches:
        return []

    metric_names = set()
    for launch in launches:
        metric_names.update(k for k in launch.keys() if not k.startswith("__"))

    selected = select_metric_names(metric_names)
    phase_models = model_phase_metrics(kernel, d, l)

    rows = []
    for launch in launches:
        def val(name):
            if name is None:
                return None
            return launch.get(name)

        gpu_time = val(selected["duration"])
        sm_util = val(selected["sm_util"])
        dram_util = val(selected["dram_util"])
        occ = val(selected["occ"])
        warp_ratio = val(selected["warp_ratio"])

        dram_bytes = val(selected["dram_bytes"])
        if dram_bytes is None:
            read_bytes = val(selected["dram_bytes_read"])
            write_bytes = val(selected["dram_bytes_write"])
            if read_bytes is not None or write_bytes is not None:
                dram_bytes = (read_bytes or 0.0) + (write_bytes or 0.0)

        fadd = val(selected["fadd"])
        fmul = val(selected["fmul"])
        ffma = val(selected["ffma"])
        flops = None
        if fadd is not None or fmul is not None or ffma is not None:
            flops = (fadd or 0.0) + (fmul or 0.0) + 2.0 * (ffma or 0.0)

        l1_hit = val(selected["l1_hit"])
        l1_miss = val(selected["l1_miss"])
        l1_hit_rate = None
        if l1_hit is not None and l1_miss is not None and (l1_hit + l1_miss) > 0:
            l1_hit_rate = 100.0 * l1_hit / (l1_hit + l1_miss)

        l2_hit = val(selected["l2_hit"])
        l2_miss = val(selected["l2_miss"])
        l2_hit_rate = None
        if l2_hit is not None and l2_miss is not None and (l2_hit + l2_miss) > 0:
            l2_hit_rate = 100.0 * l2_hit / (l2_hit + l2_miss)

        kernel_name = launch.get("__kernel_name__", "unknown")
        phase = classify_phase(kernel_name)
        phase_model = phase_models.get(phase, {})

        dram_bytes_source = "ncu" if dram_bytes is not None else None
        if dram_bytes is None and phase_model:
            dram_bytes = phase_model.get("dram_bytes")
            if dram_bytes is not None:
                dram_bytes_source = "model"

        flops_source = "ncu" if flops is not None else None
        if flops is None and phase_model:
            flops = phase_model.get("flops")
            if flops is not None:
                flops_source = "model"

        rows.append({
            "kernel": kernel,
            "dtype": dtype,
            "D": d,
            "L": l,
            "launch_id": launch.get("__launch_id__", "unknown"),
            "launch_kernel_name": kernel_name,
            "phase": phase,
            "gpu_time_duration": gpu_time,
            "sm_util_pct": sm_util,
            "dram_util_pct": dram_util,
            "achieved_occupancy_pct": occ,
            "warp_threads_per_inst": warp_ratio,
            "dram_bytes": dram_bytes,
            "dram_bytes_source": dram_bytes_source,
            "flops": flops,
            "flops_source": flops_source,
            "registers_per_thread": val(selected["regs"]),
            "shared_mem_per_block_bytes": val(selected["shmem"]),
            "block_size": val(selected["block_size"]),
            "l1_hit_rate_pct": l1_hit_rate,
            "l2_hit_rate_pct": l2_hit_rate,
        })

    return rows


def merge_launch_lists(existing, incoming):
    merged = {}
    order = []

    for launch_group in (existing, incoming):
        for launch in launch_group:
            key = (
                launch.get("__launch_id__", "unknown"),
                launch.get("__kernel_name__", "unknown"),
            )
            if key not in merged:
                merged[key] = dict(launch)
                order.append(key)
                continue

            for metric_name, metric_value in launch.items():
                if metric_name.startswith("__") and metric_name in merged[key]:
                    continue
                merged[key][metric_name] = metric_value

    return [merged[key] for key in order]


def group_mean(rows, key_fields, value_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row)

    out = []
    for key, members in sorted(groups.items(), key=lambda x: (x[0][1], KERNEL_ORDER.get(x[0][0], 999))):
        item = {k: v for k, v in zip(key_fields, key)}
        for field in value_fields:
            vals = [m[field] for m in members if m.get(field) is not None]
            item[field] = (sum(vals) / len(vals)) if vals else None
        out.append(item)
    return out


def sort_key(row):
    return (
        row.get("dtype") or "fp32",
        row["D"],
        row["L"],
        KERNEL_ORDER.get(row["kernel"], 999),
        row["kernel"],
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Analyze timing + Nsight Compute CSV outputs")
    parser.add_argument("--timing_csv", required=True)
    parser.add_argument("--raw_dir", action="append", required=True,
                        help="Raw Nsight Compute CSV directory. Pass multiple times for multi-pass profiling.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--peak_bw_gbs", type=float, default=1555.0,
                        help="Peak DRAM bandwidth in GB/s (default: 1555)")
    parser.add_argument("--peak_fp32_gflops", type=float, default=19500.0,
                        help="Peak FP32 throughput in GFLOP/s (default: 19500)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    timing = parse_timing_csv(args.timing_csv)

    raw_dirs = [path for path in args.raw_dir if os.path.isdir(path)]

    ncu_data = {}
    launch_metric_rows = []
    launches_by_key = {}
    for raw_dir in raw_dirs:
        for name in os.listdir(raw_dir):
            if not name.endswith(".csv"):
                continue
            full_path = os.path.join(raw_dir, name)
            parsed = parse_ncu_filename(full_path)
            if not parsed:
                continue
            kernel, dtype, d, l = parsed
            launches = parse_ncu_raw_csv(full_path)
            key = (kernel, dtype, d, l)
            launches_by_key[key] = merge_launch_lists(launches_by_key.get(key, []), launches)

    for (kernel, dtype, d, l), launches in launches_by_key.items():
        ncu_data[(kernel, dtype, d, l)] = aggregate_launches(launches)
        launch_metric_rows.extend(launch_to_metrics_rows(launches, kernel, dtype, d, l))

    all_keys = set(timing.keys()) | set(ncu_data.keys())
    rows = []
    ridge_ai = safe_div(args.peak_fp32_gflops, args.peak_bw_gbs)

    for key in sorted(all_keys, key=lambda k: (k[1], k[2], k[3], KERNEL_ORDER.get(k[0], 999), k[0])):
        kernel, dtype, d, l = key
        time_rec = timing.get(key, {})
        ncu_rec = ncu_data.get(key, {})
        model_rec = model_run_metrics(kernel, d, l)

        row = {
            "kernel": kernel,
            "dtype": dtype,
            "D": d,
            "L": l,
            "chunk_size": chunk_size_for_d(d),
            "time_ms": time_rec.get("time_ms"),
            "median_ms": time_rec.get("median_ms"),
            "p25_ms": time_rec.get("p25_ms"),
            "p75_ms": time_rec.get("p75_ms"),
            "iqr_ms": time_rec.get("iqr_ms"),
            "mean_ms": time_rec.get("mean_ms"),
            "stddev_ms": time_rec.get("stddev_ms"),
            "correct": time_rec.get("correct"),
            "throughput_GB_s_timing": time_rec.get("throughput_GB_s"),
            "throughput_metric": time_rec.get("throughput_metric"),
            "warmup": time_rec.get("warmup"),
            "repeat": time_rec.get("repeat"),
            "dtype": time_rec.get("dtype"),
            "samples_ms": time_rec.get("samples_ms"),
            "sm_util_pct": ncu_rec.get("sm_util_pct"),
            "dram_util_pct": ncu_rec.get("dram_util_pct"),
            "achieved_occupancy_pct": ncu_rec.get("achieved_occupancy_pct"),
            "warp_efficiency_pct": ncu_rec.get("warp_efficiency_pct"),
            "registers_per_thread_peak": ncu_rec.get("registers_per_thread_peak"),
            "shared_mem_per_block_peak_bytes": ncu_rec.get("shared_mem_per_block_peak_bytes"),
            "block_size_peak": ncu_rec.get("block_size_peak"),
            "dram_bytes": ncu_rec.get("dram_bytes"),
            "dram_bytes_source": "ncu" if ncu_rec.get("dram_bytes") is not None else None,
            "flops": ncu_rec.get("flops"),
            "flops_source": "ncu" if ncu_rec.get("flops") is not None else None,
            "l1_hit_rate_pct": ncu_rec.get("l1_hit_rate_pct"),
            "l2_hit_rate_pct": ncu_rec.get("l2_hit_rate_pct"),
        }

        if row["dram_bytes"] is None:
            row["dram_bytes"] = model_rec.get("dram_bytes")
            if row["dram_bytes"] is not None:
                row["dram_bytes_source"] = "model"

        if row["flops"] is None:
            row["flops"] = model_rec.get("flops")
            if row["flops"] is not None:
                row["flops_source"] = "model"

        row["resident_warps_est"] = None
        if row["achieved_occupancy_pct"] is not None:
            row["resident_warps_est"] = 64.0 * row["achieved_occupancy_pct"] / 100.0

        row["ai_flop_per_byte"] = safe_div(row["flops"], row["dram_bytes"])

        time_s = None
        if row["time_ms"] is not None:
            time_s = row["time_ms"] * 1e-3

        row["gflops"] = None
        if row["flops"] is not None and time_s and time_s > 0:
            row["gflops"] = row["flops"] / time_s / 1e9

        row["dram_bw_gb_s"] = None
        if row["dram_bytes"] is not None and time_s and time_s > 0:
            row["dram_bw_gb_s"] = row["dram_bytes"] / time_s / 1e9

        row["bandwidth_util_pct"] = row["dram_util_pct"]
        if row["bandwidth_util_pct"] is None and row["dram_bw_gb_s"] is not None:
            row["bandwidth_util_pct"] = 100.0 * row["dram_bw_gb_s"] / args.peak_bw_gbs

        row["roofline_bound"] = None
        if row["ai_flop_per_byte"] is not None and ridge_ai is not None:
            row["roofline_bound"] = "memory-bound" if row["ai_flop_per_byte"] < ridge_ai else "compute-bound"

        row["roofline_limit_gflops"] = None
        if row["ai_flop_per_byte"] is not None:
            row["roofline_limit_gflops"] = min(args.peak_fp32_gflops,
                                               row["ai_flop_per_byte"] * args.peak_bw_gbs)

        row["roofline_efficiency_pct"] = safe_div(100.0 * (row["gflops"] or 0.0),
                                                    row["roofline_limit_gflops"]) \
            if row["gflops"] is not None and row["roofline_limit_gflops"] not in (None, 0.0) else None

        row["utilization_bound"] = None
        if row["sm_util_pct"] is not None and row["bandwidth_util_pct"] is not None:
            if row["bandwidth_util_pct"] > row["sm_util_pct"] + 5.0:
                row["utilization_bound"] = "memory-bound"
            elif row["sm_util_pct"] > row["bandwidth_util_pct"] + 5.0:
                row["utilization_bound"] = "compute-bound"
            else:
                row["utilization_bound"] = "mixed"

        rows.append(row)

    rows.sort(key=sort_key)

    launch_metric_rows.sort(key=lambda r: (
        r["D"],
        r["L"],
        r["dtype"],
        KERNEL_ORDER.get(r["kernel"], 999),
        r["phase"],
        str(r["launch_id"]),
    ))

    launch_metrics_csv = os.path.join(args.out_dir, "kernel_launch_metrics.csv")
    write_csv(
        launch_metrics_csv,
        launch_metric_rows,
        [
            "kernel", "D", "L", "launch_id", "launch_kernel_name", "phase",
            "dtype",
            "gpu_time_duration", "sm_util_pct", "dram_util_pct",
            "achieved_occupancy_pct", "warp_threads_per_inst", "dram_bytes", "flops",
            "dram_bytes_source", "flops_source",
            "registers_per_thread", "shared_mem_per_block_bytes", "block_size",
            "l1_hit_rate_pct", "l2_hit_rate_pct",
        ],
    )

    phase_acc = {}
    for row in launch_metric_rows:
        key = (row["kernel"], row["dtype"], row["D"], row["L"])
        if key not in phase_acc:
            phase_acc[key] = {
                "phase1_time": 0.0,
                "phase2_time": 0.0,
                "phase3_time": 0.0,
                "other_time": 0.0,
                "phase1_launches": 0,
                "phase2_launches": 0,
                "phase3_launches": 0,
                "other_launches": 0,
            }
        phase = row["phase"]
        dur = row.get("gpu_time_duration")
        if phase == "phase1":
            phase_acc[key]["phase1_launches"] += 1
            if dur is not None:
                phase_acc[key]["phase1_time"] += dur
        elif phase == "phase2":
            phase_acc[key]["phase2_launches"] += 1
            if dur is not None:
                phase_acc[key]["phase2_time"] += dur
        elif phase == "phase3":
            phase_acc[key]["phase3_launches"] += 1
            if dur is not None:
                phase_acc[key]["phase3_time"] += dur
        else:
            phase_acc[key]["other_launches"] += 1
            if dur is not None:
                phase_acc[key]["other_time"] += dur

    phase_rows = []
    for (kernel, dtype, d, l), stats in sorted(phase_acc.items(), key=lambda x: (x[0][1], x[0][2], x[0][3], KERNEL_ORDER.get(x[0][0], 999))):
        total = stats["phase1_time"] + stats["phase2_time"] + stats["phase3_time"] + stats["other_time"]
        phase_rows.append({
            "kernel": kernel,
            "dtype": dtype,
            "D": d,
            "L": l,
            "phase1_time": stats["phase1_time"],
            "phase2_time": stats["phase2_time"],
            "phase3_time": stats["phase3_time"],
            "other_time": stats["other_time"],
            "total_time": total,
            "phase1_pct": (100.0 * stats["phase1_time"] / total) if total > 0 else None,
            "phase2_pct": (100.0 * stats["phase2_time"] / total) if total > 0 else None,
            "phase3_pct": (100.0 * stats["phase3_time"] / total) if total > 0 else None,
            "other_pct": (100.0 * stats["other_time"] / total) if total > 0 else None,
            "phase1_launches": stats["phase1_launches"],
            "phase2_launches": stats["phase2_launches"],
            "phase3_launches": stats["phase3_launches"],
            "other_launches": stats["other_launches"],
        })

    phase_breakdown_csv = os.path.join(args.out_dir, "phase_breakdown.csv")
    write_csv(
        phase_breakdown_csv,
        phase_rows,
        [
            "kernel", "dtype", "D", "L",
            "phase1_time", "phase2_time", "phase3_time", "other_time", "total_time",
            "phase1_pct", "phase2_pct", "phase3_pct", "other_pct",
            "phase1_launches", "phase2_launches", "phase3_launches", "other_launches",
        ],
    )

    merged_csv = os.path.join(args.out_dir, "merged_metrics.csv")
    merged_fields = [
        "kernel", "D", "L", "dtype", "chunk_size", "time_ms", "median_ms", "p25_ms", "p75_ms",
        "iqr_ms", "mean_ms", "stddev_ms", "correct", "throughput_GB_s_timing", "throughput_metric",
        "warmup", "repeat", "samples_ms",
        "sm_util_pct", "dram_util_pct", "achieved_occupancy_pct", "resident_warps_est",
        "warp_efficiency_pct", "registers_per_thread_peak", "shared_mem_per_block_peak_bytes",
        "block_size_peak", "dram_bytes", "dram_bytes_source", "flops", "flops_source",
        "ai_flop_per_byte", "gflops", "dram_bw_gb_s",
        "bandwidth_util_pct", "l1_hit_rate_pct", "l2_hit_rate_pct", "roofline_bound",
        "roofline_limit_gflops", "roofline_efficiency_pct", "utilization_bound",
    ]
    write_csv(merged_csv, rows, merged_fields)

    occ_summary = group_mean(
        rows,
        key_fields=["kernel", "dtype", "D"],
        value_fields=[
            "achieved_occupancy_pct",
            "resident_warps_est",
            "warp_efficiency_pct",
            "registers_per_thread_peak",
            "shared_mem_per_block_peak_bytes",
            "block_size_peak",
            "sm_util_pct",
            "dram_util_pct",
            "bandwidth_util_pct",
        ],
    )
    occ_summary_csv = os.path.join(args.out_dir, "occupancy_summary.csv")
    write_csv(
        occ_summary_csv,
        occ_summary,
        [
            "kernel", "dtype", "D", "achieved_occupancy_pct", "resident_warps_est", "warp_efficiency_pct",
            "registers_per_thread_peak", "shared_mem_per_block_peak_bytes", "block_size_peak",
            "sm_util_pct", "dram_util_pct", "bandwidth_util_pct",
        ],
    )

    crossover_rows = []
    dtypes = sorted({row["dtype"] for row in rows})
    ds = sorted({row["D"] for row in rows})
    ls = sorted({row["L"] for row in rows})
    time_lookup = {(row["kernel"], row["dtype"], row["D"], row["L"]): row["time_ms"] for row in rows}

    for dtype in dtypes:
        for d in ds:
            ratio_series = []
            for l in ls:
                hillis = time_lookup.get(("hillis_steele", dtype, d, l))
                blelloch = time_lookup.get(("blelloch", dtype, d, l))
                if hillis is None or blelloch in (None, 0.0):
                    continue
                ratio_series.append((l, hillis / blelloch))

            crossing = None
            for i in range(1, len(ratio_series)):
                prev_l, prev_r = ratio_series[i - 1]
                cur_l, cur_r = ratio_series[i]
                if (prev_r - 1.0) == 0:
                    crossing = (prev_l, prev_l)
                    break
                if (prev_r - 1.0) * (cur_r - 1.0) < 0:
                    crossing = (prev_l, cur_l)
                    break

            if crossing:
                note = f"crosses between L={crossing[0]} and L={crossing[1]}"
            elif ratio_series and all(r > 1.0 for _, r in ratio_series):
                note = "hillis_steele slower than blelloch for all tested L"
            elif ratio_series and all(r < 1.0 for _, r in ratio_series):
                note = "hillis_steele faster than blelloch for all tested L"
            else:
                note = "no crossover detected in tested range"

            crossover_rows.append({
                "dtype": dtype,
                "D": d,
                "note": note,
                "ratio_series": "; ".join(f"L={l}: {r:.4f}" for l, r in ratio_series),
            })

    crossover_csv = os.path.join(args.out_dir, "crossover_summary.csv")
    write_csv(crossover_csv, crossover_rows, ["dtype", "D", "note", "ratio_series"])

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=== SSM PREFIX SCAN METRICS REPORT ===")
    emit(f"timing_csv: {args.timing_csv}")
    emit(f"raw_dirs:   {', '.join(raw_dirs) if raw_dirs else 'NA'}")
    emit(f"out_dir:    {args.out_dir}")
    emit(f"peak_bw_gbs={args.peak_bw_gbs:.3f} peak_fp32_gflops={args.peak_fp32_gflops:.3f}")
    emit("")

    emit("=== PERFORMANCE TABLES (time_ms) ===")
    kernels_present = [
        kernel for kernel, _ in sorted(KERNEL_ORDER.items(), key=lambda item: item[1])
        if any(row["kernel"] == kernel for row in rows)
    ]
    for dtype in dtypes:
        emit(f"dtype={dtype}")
        for d in ds:
            emit(f"D={d}")
            emit(",".join(["L"] + kernels_present))
            for l in ls:
                values = [fmt(time_lookup.get((kernel, dtype, d, l)), 4) for kernel in kernels_present]
                emit(",".join([str(l)] + values))
            emit("")

    emit("=== ROOFLINE POINTS (all runs) ===")
    emit("kernel,dtype,D,L,AI_flop_per_byte,GFLOP_s,DRAM_BW_GB_s,roofline_bound,roofline_eff_pct")
    for row in rows:
        emit(
            f"{row['kernel']},{row['dtype']},{row['D']},{row['L']},"
            f"{fmt(row['ai_flop_per_byte'], 6)},"
            f"{fmt(row['gflops'], 3)},"
            f"{fmt(row['dram_bw_gb_s'], 3)},"
            f"{row['roofline_bound'] or 'NA'},"
            f"{fmt(row['roofline_efficiency_pct'], 2)}"
        )
    emit("")

    emit("=== FULL PER-RUN UTILIZATION TABLE ===")
    emit("kernel,dtype,D,L,occ_pct,warp_eff_pct,sm_util_pct,dram_util_pct,bw_util_pct,resident_warps,regs_per_thread,block_size,shared_mem_B,l1_hit_pct,l2_hit_pct,utilization_bound")
    for row in rows:
        emit(
            f"{row['kernel']},{row['dtype']},{row['D']},{row['L']},"
            f"{fmt(row['achieved_occupancy_pct'], 2)},"
            f"{fmt(row['warp_efficiency_pct'], 2)},"
            f"{fmt(row['sm_util_pct'], 2)},"
            f"{fmt(row['dram_util_pct'], 2)},"
            f"{fmt(row['bandwidth_util_pct'], 2)},"
            f"{fmt(row['resident_warps_est'], 2)},"
            f"{fmt(row['registers_per_thread_peak'], 2)},"
            f"{fmt(row['block_size_peak'], 2)},"
            f"{fmt(row['shared_mem_per_block_peak_bytes'], 2)},"
            f"{fmt(row['l1_hit_rate_pct'], 2)},"
            f"{fmt(row['l2_hit_rate_pct'], 2)},"
            f"{row['utilization_bound'] or 'NA'}"
        )
    emit("")

    emit("=== PHASE BREAKDOWN (gpu_time_duration shares) ===")
    emit("kernel,dtype,D,L,phase1_time,phase2_time,phase3_time,other_time,total_time,phase1_pct,phase2_pct,phase3_pct,other_pct")
    for row in phase_rows:
        emit(
            f"{row['kernel']},{row['dtype']},{row['D']},{row['L']},"
            f"{fmt(row['phase1_time'], 4)},"
            f"{fmt(row['phase2_time'], 4)},"
            f"{fmt(row['phase3_time'], 4)},"
            f"{fmt(row['other_time'], 4)},"
            f"{fmt(row['total_time'], 4)},"
            f"{fmt(row['phase1_pct'], 2)},"
            f"{fmt(row['phase2_pct'], 2)},"
            f"{fmt(row['phase3_pct'], 2)},"
            f"{fmt(row['other_pct'], 2)}"
        )
    emit("")

    emit("=== OCCUPANCY/REGISTER SUMMARY (avg over L) ===")
    emit("kernel,dtype,D,occ_pct_avg,resident_warps_avg,warp_eff_avg,regs_peak_avg,block_size_avg,shared_mem_B_avg,sm_util_avg,dram_util_avg,bw_util_avg")
    for item in occ_summary:
        emit(
            f"{item['kernel']},{item['dtype']},{item['D']},"
            f"{fmt(item['achieved_occupancy_pct'], 2)},"
            f"{fmt(item['resident_warps_est'], 2)},"
            f"{fmt(item['warp_efficiency_pct'], 2)},"
            f"{fmt(item['registers_per_thread_peak'], 2)},"
            f"{fmt(item['block_size_peak'], 2)},"
            f"{fmt(item['shared_mem_per_block_peak_bytes'], 2)},"
            f"{fmt(item['sm_util_pct'], 2)},"
            f"{fmt(item['dram_util_pct'], 2)},"
            f"{fmt(item['bandwidth_util_pct'], 2)}"
        )
    emit("")

    emit("=== CROSSOVER ANALYSIS (hillis_steele / blelloch) ===")
    for row in crossover_rows:
        emit(f"dtype={row['dtype']} D={row['D']}: {row['note']}")
        emit(f"  {row['ratio_series']}")
    emit("")

    emit("=== OUTPUT FILES ===")
    emit(f"merged_metrics_csv: {merged_csv}")
    emit(f"kernel_launch_metrics_csv: {launch_metrics_csv}")
    emit(f"phase_breakdown_csv: {phase_breakdown_csv}")
    emit(f"occupancy_summary_csv: {occ_summary_csv}")
    emit(f"crossover_summary_csv: {crossover_csv}")

    report_path = os.path.join(args.out_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
