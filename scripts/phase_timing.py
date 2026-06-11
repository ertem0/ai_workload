import json
import sys
import argparse
import re
import bisect
from collections import defaultdict

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="CPU vs GPU time breakdown for prefill and decode phases"
)
parser.add_argument("--trace", required=True, help="Path to inference_trace.json")
parser.add_argument(
    "--show-decode-steps",
    action="store_true",
    help="Show each decode step individually instead of an average",
)
args = parser.parse_args()

# ── Load trace ────────────────────────────────────────────────────────────────
print(f"\nLoading {args.trace}...")
with open(args.trace) as f:
    data = json.load(f)
events = data.get("traceEvents", data)

# ── Pre-build ext_id → GPU annotation index ───────────────────────────────────
gpu_by_ext_id = {}
for e in events:
    if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X":
        ext_id = e.get("args", {}).get("External id")
        if ext_id is not None and ext_id not in gpu_by_ext_id:
            gpu_by_ext_id[ext_id] = e

def get_gpu_event(cpu_event):
    ext_id = cpu_event.get("args", {}).get("External id")
    return gpu_by_ext_id.get(ext_id) if ext_id is not None else None

# ── Pre-sort GPU kernel events for fast window queries ────────────────────────
GEMM_KW = ("cutlass", "gemm", "sgemm", "gemv")

sorted_kernels = sorted(
    [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy") and e.get("ph") == "X"],
    key=lambda e: e["ts"],
)
kernel_ts = [e["ts"] for e in sorted_kernels]

# ── Pre-sort CPU annotations for subtree child lookup ─────────────────────────
sorted_cpu_anns = sorted(
    [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"],
    key=lambda e: e["ts"],
)
cpu_ann_ts = [e["ts"] for e in sorted_cpu_anns]


def _merge_intervals(intervals):
    """Merge a list of (start, end) intervals, returning sorted non-overlapping list."""
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(ivs[0])]
    for start, end in ivs[1:]:
        if start <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def gpu_info_for_subtree(cpu_event):
    """
    Collect GPU windows for cpu_event AND all CPU annotations contained within
    its CPU window (i.e. its children in the module tree).  This handles models
    like Gemma4 where the parent .mlp annotation has a tiny GPU window but the
    real computation is captured under child .gate_proj / .up_proj / .down_proj
    annotations.

    Returns a dict:
      gpu        — duration of the union of all GPU windows (no double-count)
      gemm       — sum of GEMM kernel durations inside those windows
      kernel     — sum of ALL kernel durations inside those windows
      gap        — sum of idle gaps >2μs between consecutive kernels
      sync_gap   — gap time adjacent to a gpu_memcpy (DtoH sync proxy)
    Returns None for gpu if no GPU annotation was found.
    """
    cpu_start = cpu_event["ts"]
    cpu_end   = cpu_start + cpu_event.get("dur", 0)

    lo = bisect.bisect_left(cpu_ann_ts, cpu_start - 5)
    seen_ext_ids = set()
    raw_intervals = []

    for i in range(lo, len(sorted_cpu_anns)):
        e = sorted_cpu_anns[i]
        if e["ts"] > cpu_end + 5:
            break
        e_end = e["ts"] + e.get("dur", 0)
        if e["ts"] >= cpu_start - 5 and e_end <= cpu_end + 5:
            ext_id = e.get("args", {}).get("External id")
            if ext_id is not None and ext_id not in seen_ext_ids:
                seen_ext_ids.add(ext_id)
                gpu_ev = gpu_by_ext_id.get(ext_id)
                if gpu_ev:
                    raw_intervals.append((gpu_ev["ts"], gpu_ev["ts"] + gpu_ev["dur"]))

    if not raw_intervals:
        return None

    merged = _merge_intervals(raw_intervals)
    total_gpu = sum(end - start for start, end in merged)

    # Collect all kernel/memcpy events inside the merged GPU windows
    search_lo = bisect.bisect_left(kernel_ts, merged[0][0] - 5)
    search_hi = bisect.bisect_right(kernel_ts, merged[-1][1] + 5)

    active = []   # events inside the window, sorted by ts
    for e in sorted_kernels[search_lo:search_hi]:
        ts    = e["ts"]
        e_end = ts + e.get("dur", 0)
        if any(ts >= s - 5 and e_end <= en + 5 for s, en in merged):
            active.append(e)
    active.sort(key=lambda x: x["ts"])

    total_kernel = sum(e.get("dur", 0) for e in active)
    total_gemm   = sum(
        e.get("dur", 0) for e in active
        if any(x in e["name"].lower() for x in GEMM_KW)
    )

    # Gap analysis: idle periods >2μs between consecutive kernels.
    # Computed per merged interval so gaps between different layers'
    # GPU windows (e.g. layer0.attention vs layer1.attention) are not
    # counted — only gaps inside a single module's GPU window matter.
    total_gap = 0.0
    sync_gap  = 0.0
    for s, en in merged:
        w_kernels = sorted(
            (e for e in active if e["ts"] >= s - 5 and
             e["ts"] + e.get("dur", 0) <= en + 5),
            key=lambda x: x["ts"],
        )
        for i in range(1, len(w_kernels)):
            prev_end   = w_kernels[i-1]["ts"] + w_kernels[i-1].get("dur", 0)
            curr_start = w_kernels[i]["ts"]
            gap_dur    = curr_start - prev_end
            if gap_dur <= 2:
                continue
            total_gap += gap_dur
            if (w_kernels[i-1].get("cat") == "gpu_memcpy" or
                    w_kernels[i].get("cat") == "gpu_memcpy"):
                sync_gap += gap_dur

    return {
        "gpu":      total_gpu,
        "gemm":     total_gemm,
        "kernel":   total_kernel,
        "gap":      total_gap,
        "sync_gap": sync_gap,
    }


def gemm_time_in_window(gpu_start, gpu_end):
    """Sum GEMM kernel durations within a single GPU window (used for total_pass)."""
    lo = bisect.bisect_left(kernel_ts, gpu_start - 5)
    hi = bisect.bisect_right(kernel_ts, gpu_end + 5)
    total = 0.0
    for e in sorted_kernels[lo:hi]:
        if e["ts"] + e.get("dur", 0) <= gpu_end + 5:
            if any(x in e["name"].lower() for x in GEMM_KW):
                total += e.get("dur", 0)
    return total


def _window_stats(gpu_start, gpu_end):
    """
    Return (gemm, kernel, gap, sync_gap) for a single GPU window.
    Used for total_pass accumulation where the full-layer annotation
    already covers the whole layer — no subtree walk needed.
    """
    lo = bisect.bisect_left(kernel_ts, gpu_start - 5)
    hi = bisect.bisect_right(kernel_ts, gpu_end + 5)

    active = []
    for e in sorted_kernels[lo:hi]:
        if e["ts"] + e.get("dur", 0) <= gpu_end + 5:
            active.append(e)
    active.sort(key=lambda x: x["ts"])

    total_gemm   = sum(e.get("dur", 0) for e in active
                       if any(x in e["name"].lower() for x in GEMM_KW))
    total_kernel = sum(e.get("dur", 0) for e in active)

    total_gap = 0.0
    sync_gap  = 0.0
    for i in range(1, len(active)):
        prev_end   = active[i-1]["ts"] + active[i-1].get("dur", 0)
        curr_start = active[i]["ts"]
        gap_dur    = curr_start - prev_end
        if gap_dur <= 2:
            continue
        total_gap += gap_dur
        if (active[i-1].get("cat") == "gpu_memcpy" or
                active[i].get("cat") == "gpu_memcpy"):
            sync_gap += gap_dur

    return total_gemm, total_kernel, total_gap, sync_gap

# ── Identify forward passes via layer0 (full layer) annotations ───────────────
layer0_anns = sorted(
    [
        e for e in events
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and re.match(r"^layer0\s+\(", e.get("name", ""))
    ],
    key=lambda e: e["ts"],
)

if not layer0_anns:
    print("ERROR: Could not find 'layer0 (...)' annotations to identify passes.")
    print("Make sure the trace was recorded with profiling enabled.")
    sys.exit(1)

n_passes = len(layer0_anns)
n_decode  = n_passes - 1
pass_starts = [e["ts"] for e in layer0_anns]
print(f"Found {n_passes} forward passes: 1 prefill + {n_decode} decode step(s)")

def get_pass_idx(ts):
    """0 = prefill, 1+ = decode steps."""
    idx = bisect.bisect_right(pass_starts, ts) - 1
    return max(0, idx)

# ── Component filters ─────────────────────────────────────────────────────────
def _is_attention(name):
    return ".attention (" in name

def _is_expert(name):
    if re.search(r"\.experts\s+\(", name):
        return True
    m = re.search(r"\.expert\d+\s+\((\w+)\)", name)
    return bool(m) and ("MLP" in m.group(1) or "Expert" in m.group(1))

def _is_router(name):
    return bool(re.search(r"\.router\s+\(", name))

def _is_mlp(name):
    m = re.search(r"\.mlp\s+\((\w+)\)", name)
    return bool(m) and "MLP" in m.group(1)

def _is_lm_head(name):
    return name.startswith("lm_head (")

COMPONENTS = [
    ("attention", _is_attention),
    ("expert",    _is_expert),
    ("router",    _is_router),
    ("mlp",       _is_mlp),
    ("lm_head",   _is_lm_head),
]

# ── Aggregate per component per pass ─────────────────────────────────────────
# agg[comp][pass] = {cpu, gpu, gemm, kernel, gap, sync_gap, count, gpu_miss}
def _zero_bucket():
    return {"cpu": 0.0, "gpu": 0.0, "gemm": 0.0,
            "kernel": 0.0, "gap": 0.0, "sync_gap": 0.0,
            "count": 0, "gpu_miss": 0}

agg = defaultdict(lambda: defaultdict(_zero_bucket))

# total_pass[pass] = {cpu, gpu, gemm, kernel, gap, sync_gap}
total_pass = defaultdict(lambda: {"cpu": 0.0, "gpu": 0.0, "gemm": 0.0,
                                   "kernel": 0.0, "gap": 0.0, "sync_gap": 0.0})

for e in events:
    if e.get("cat") != "user_annotation" or e.get("ph") != "X":
        continue
    name = e.get("name", "")
    ts   = e.get("ts", 0)
    dur  = e.get("dur", 0)
    pass_idx = get_pass_idx(ts)

    # Attribute to first matching component
    for comp_name, comp_fn in COMPONENTS:
        if comp_fn(name):
            # Use the full subtree of GPU windows so child-annotated models
            # (e.g. Gemma4 .mlp → .gate_proj / .up_proj / .down_proj) are
            # correctly accounted for rather than only the parent's tiny window.
            info = gpu_info_for_subtree(e)
            agg[comp_name][pass_idx]["cpu"] += dur
            if info is not None:
                agg[comp_name][pass_idx]["gpu"]      += info["gpu"]
                agg[comp_name][pass_idx]["gemm"]     += info["gemm"]
                agg[comp_name][pass_idx]["kernel"]   += info["kernel"]
                agg[comp_name][pass_idx]["gap"]      += info["gap"]
                agg[comp_name][pass_idx]["sync_gap"] += info["sync_gap"]
            else:
                agg[comp_name][pass_idx]["gpu_miss"] += 1
            agg[comp_name][pass_idx]["count"] += 1
            break

    # Total-pass time: full-layer + embedding + lm_head annotations only.
    # Use the direct GPU window (not subtree) — the full-layer annotation already
    # spans the entire layer, so its single GPU window is the correct reference.
    is_full_layer = bool(re.match(r"^layer\d+\s+\(", name))
    is_embedding  = name.startswith("embedding (")
    is_lmh        = name.startswith("lm_head (")
    if is_full_layer or is_embedding or is_lmh:
        gpu_ev = get_gpu_event(e)
        if gpu_ev:
            gstart = gpu_ev["ts"]
            gend   = gstart + gpu_ev["dur"]
            tp_gemm, tp_kernel, tp_gap, tp_sync = _window_stats(gstart, gend)
            total_pass[pass_idx]["gpu"]      += gpu_ev["dur"]
            total_pass[pass_idx]["gemm"]     += tp_gemm
            total_pass[pass_idx]["kernel"]   += tp_kernel
            total_pass[pass_idx]["gap"]      += tp_gap
            total_pass[pass_idx]["sync_gap"] += tp_sync
        total_pass[pass_idx]["cpu"] += dur

# ── Only keep components present in this trace ───────────────────────────────
present_components = [
    (comp_name, comp_fn)
    for comp_name, comp_fn in COMPONENTS
    if any(agg[comp_name][p]["count"] > 0 for p in range(n_passes))
]

# ── Aggregation helpers ───────────────────────────────────────────────────────
def decode_avg(comp_name, field):
    if n_decode == 0:
        return 0.0
    return sum(agg[comp_name][p][field] for p in range(1, n_passes)) / n_decode

def pass_total(pass_idx, field):
    return total_pass[pass_idx][field]

def decode_total_avg(field):
    if n_decode == 0:
        return 0.0
    return sum(pass_total(p, field) for p in range(1, n_passes)) / n_decode

# ── Formatting helpers ────────────────────────────────────────────────────────
W = 100

def fmt(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s  "
    if us >= 1_000:
        return f"{us/1_000:.2f}ms"
    return f"{us:.1f}μs "

def pct(num, den):
    if den == 0:
        return "   n/a"
    return f"{num/den:5.1%}"

# ══════════════════════════════════════════════════════════════════════════════
# PRINT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  Trace:  {args.trace}")
print(f"  Passes: 1 prefill + {n_decode} decode step(s)")
print(f"{'='*W}")

pre_cpu  = pass_total(0, "cpu")
pre_gpu  = pass_total(0, "gpu")
pre_gemm = pass_total(0, "gemm")
dec_cpu  = decode_total_avg("cpu")
dec_gpu  = decode_total_avg("gpu")
dec_gemm = decode_total_avg("gemm")

# ── Phase summary ─────────────────────────────────────────────────────────────
print(f"\n{'PHASE SUMMARY  (layers + embedding + lm_head)':─<{W}}")
print(f"  {'Metric':<35} {'PREFILL':>16}  {'DECODE avg/step':>16}")
print(f"  {'─'*35} {'─'*16}  {'─'*16}")
print(f"  {'CPU total time':<35} {fmt(pre_cpu):>16}  {fmt(dec_cpu):>16}")
print(f"  {'GPU total time':<35} {fmt(pre_gpu):>16}  {fmt(dec_gpu):>16}")
print(f"  {'Static matmul (GEMM) time':<35} {fmt(pre_gemm):>16}  {fmt(dec_gemm):>16}")
print(f"  {'GEMM as % of GPU':<35} {pct(pre_gemm, pre_gpu):>16}  {pct(dec_gemm, dec_gpu):>16}")
print(f"  {'Non-GEMM GPU time':<35} {fmt(pre_gpu - pre_gemm):>16}  {fmt(dec_gpu - dec_gemm):>16}")
if pre_cpu > 0 and dec_cpu > 0:
    print(f"  {'Prefill/Decode ratio (CPU)':<35} {pre_cpu/dec_cpu:>15.1f}x  {'':>16}")
if pre_gpu > 0 and dec_gpu > 0:
    print(f"  {'Prefill/Decode ratio (GPU)':<35} {pre_gpu/dec_gpu:>15.1f}x  {'':>16}")

# ── Per-component breakdown ───────────────────────────────────────────────────
print(f"\n{'COMPONENT BREAKDOWN':─<{W}}")
print(f"  {'Component':<14}"
      f"  {'PREFILL CPU':>14} {'%tot':>5}"
      f"  {'PREFILL GPU':>14} {'%tot':>5}"
      f"  {'DECODE CPU avg':>14} {'%tot':>5}"
      f"  {'DECODE GPU avg':>14} {'%tot':>5}")
print(f"  {'─'*14}"
      f"  {'─'*14} {'─'*5}"
      f"  {'─'*14} {'─'*5}"
      f"  {'─'*14} {'─'*5}"
      f"  {'─'*14} {'─'*5}")

for comp_name, _ in present_components:
    p_cpu = agg[comp_name][0]["cpu"]
    p_gpu = agg[comp_name][0]["gpu"]
    d_cpu = decode_avg(comp_name, "cpu")
    d_gpu = decode_avg(comp_name, "gpu")
    print(
        f"  {comp_name:<14}"
        f"  {fmt(p_cpu):>14} {pct(p_cpu, pre_cpu):>5}"
        f"  {fmt(p_gpu):>14} {pct(p_gpu, pre_gpu):>5}"
        f"  {fmt(d_cpu):>14} {pct(d_cpu, dec_cpu):>5}"
        f"  {fmt(d_gpu):>14} {pct(d_gpu, dec_gpu):>5}"
    )

# ── Static matmul breakdown ───────────────────────────────────────────────────
print(f"\n{'STATIC MATMUL (GEMM) BREAKDOWN  — cutlass / gemm / sgemm / gemv kernels':─<{W}}")
print(f"  {'Component':<14}"
      f"  {'PREFILL GEMM':>14} {'% GPU':>6}"
      f"  {'DECODE GEMM avg':>15} {'% GPU':>6}")
print(f"  {'─'*14}"
      f"  {'─'*14} {'─'*6}"
      f"  {'─'*15} {'─'*6}")

for comp_name, _ in present_components:
    p_gemm = agg[comp_name][0]["gemm"]
    p_gpu  = agg[comp_name][0]["gpu"]
    d_gemm = decode_avg(comp_name, "gemm")
    d_gpu  = decode_avg(comp_name, "gpu")
    print(
        f"  {comp_name:<14}"
        f"  {fmt(p_gemm):>14} {pct(p_gemm, p_gpu):>6}"
        f"  {fmt(d_gemm):>15} {pct(d_gemm, d_gpu):>6}"
    )

# totals row
print(
    f"  {'TOTAL':<14}"
    f"  {fmt(pre_gemm):>14} {pct(pre_gemm, pre_gpu):>6}"
    f"  {fmt(dec_gemm):>15} {pct(dec_gemm, dec_gpu):>6}"
)

# ── GPU idle gap analysis ─────────────────────────────────────────────────────
# gap  = idle time between consecutive kernels (>2μs threshold)
# sync = subset of gap time adjacent to a gpu_memcpy (DtoH/HtoD sync proxy)
print(f"\n{'GPU IDLE GAP ANALYSIS  (gaps >2μs between consecutive kernels)':─<{W}}")
print(f"  {'Component':<14}"
      f"  {'PRE gap':>10} {'% GPU':>6} {'sync%':>6}"
      f"  {'DEC gap avg':>12} {'% GPU':>6} {'sync%':>6}")
print(f"  {'─'*14}"
      f"  {'─'*10} {'─'*6} {'─'*6}"
      f"  {'─'*12} {'─'*6} {'─'*6}")

for comp_name, _ in present_components:
    p_gap  = agg[comp_name][0]["gap"]
    p_sync = agg[comp_name][0]["sync_gap"]
    p_gpu  = agg[comp_name][0]["gpu"]
    d_gap  = decode_avg(comp_name, "gap")
    d_sync = decode_avg(comp_name, "sync_gap")
    d_gpu  = decode_avg(comp_name, "gpu")
    print(
        f"  {comp_name:<14}"
        f"  {fmt(p_gap):>10} {pct(p_gap, p_gpu):>6} {pct(p_sync, p_gap):>6}"
        f"  {fmt(d_gap):>12} {pct(d_gap, d_gpu):>6} {pct(d_sync, d_gap):>6}"
    )

# phase totals row
pre_gap  = pass_total(0, "gap")
pre_sync = pass_total(0, "sync_gap")
dec_gap  = decode_total_avg("gap")
dec_sync = decode_total_avg("sync_gap")
print(
    f"  {'TOTAL':<14}"
    f"  {fmt(pre_gap):>10} {pct(pre_gap, pre_gpu):>6} {pct(pre_sync, pre_gap):>6}"
    f"  {fmt(dec_gap):>12} {pct(dec_gap, dec_gpu):>6} {pct(dec_sync, dec_gap):>6}"
)

# ── CPU vs GPU ratios ─────────────────────────────────────────────────────────
print(f"\n{'CPU vs GPU RATIO  (CPU / GPU — >1 means CPU is the bottleneck)':─<{W}}")
print(f"  {'Component':<14}  {'PREFILL CPU/GPU':>16}  {'DECODE CPU/GPU avg':>18}")
print(f"  {'─'*14}  {'─'*16}  {'─'*18}")
for comp_name, _ in present_components:
    p_cpu = agg[comp_name][0]["cpu"]
    p_gpu = agg[comp_name][0]["gpu"]
    d_cpu = decode_avg(comp_name, "cpu")
    d_gpu = decode_avg(comp_name, "gpu")
    p_ratio = f"{p_cpu/p_gpu:.2f}x" if p_gpu > 0 else "n/a"
    d_ratio = f"{d_cpu/d_gpu:.2f}x" if d_gpu > 0 else "n/a"
    print(f"  {comp_name:<14}  {p_ratio:>16}  {d_ratio:>18}")

# ── Bottleneck metrics (section 4.2.3) ───────────────────────────────────────
# Pipeline efficiency  = kernel_time / gpu_window   (fraction of GPU window active)
# AIMC ceiling         = gpu / (gpu - gemm)          (speedup if matmuls were instant)
# Effective AIMC ceil  = gpu / (gpu - gemm - sync)   (also removes sync barriers)
def _ceiling(gpu, gemm, sync=0.0):
    denom = gpu - gemm - sync
    return f"{gpu/denom:.2f}x" if denom > 0 else "n/a"

def _pipe_eff(kernel, gpu):
    return pct(kernel, gpu) if gpu > 0 else "   n/a"

print(f"\n{'BOTTLENECK METRICS':─<{W}}")
print(f"  {'Component':<14}"
      f"  {'PRE pipe-eff':>13} {'PRE AIMC ceil':>14} {'PRE eff-ceil':>13}"
      f"  {'DEC pipe-eff':>13} {'DEC AIMC ceil':>14} {'DEC eff-ceil':>13}")
print(f"  {'─'*14}"
      f"  {'─'*13} {'─'*14} {'─'*13}"
      f"  {'─'*13} {'─'*14} {'─'*13}")

for comp_name, _ in present_components:
    p_gpu  = agg[comp_name][0]["gpu"]
    p_gemm = agg[comp_name][0]["gemm"]
    p_knl  = agg[comp_name][0]["kernel"]
    p_sync = agg[comp_name][0]["sync_gap"]
    d_gpu  = decode_avg(comp_name, "gpu")
    d_gemm = decode_avg(comp_name, "gemm")
    d_knl  = decode_avg(comp_name, "kernel")
    d_sync = decode_avg(comp_name, "sync_gap")
    print(
        f"  {comp_name:<14}"
        f"  {_pipe_eff(p_knl, p_gpu):>13} {_ceiling(p_gpu, p_gemm):>14} {_ceiling(p_gpu, p_gemm, p_sync):>13}"
        f"  {_pipe_eff(d_knl, d_gpu):>13} {_ceiling(d_gpu, d_gemm):>14} {_ceiling(d_gpu, d_gemm, d_sync):>13}"
    )

p_knl_tot  = pass_total(0, "kernel")
p_sync_tot = pass_total(0, "sync_gap")
d_knl_tot  = decode_total_avg("kernel")
d_sync_tot = decode_total_avg("sync_gap")
print(
    f"  {'TOTAL':<14}"
    f"  {_pipe_eff(p_knl_tot, pre_gpu):>13} {_ceiling(pre_gpu, pre_gemm):>14} {_ceiling(pre_gpu, pre_gemm, p_sync_tot):>13}"
    f"  {_pipe_eff(d_knl_tot, dec_gpu):>13} {_ceiling(dec_gpu, dec_gemm):>14} {_ceiling(dec_gpu, dec_gemm, d_sync_tot):>13}"
)
print(f"  (pipe-eff: % of GPU window running a kernel  |  AIMC ceil: gpu/(gpu-gemm)  |  eff-ceil: also removes sync)")

# ── Per-step decode breakdown (optional) ─────────────────────────────────────
if args.show_decode_steps and n_decode > 0:
    print(f"\n{'PER-STEP DECODE DETAIL':─<{W}}")
    header = f"  {'Step':<6}"
    for comp_name, _ in present_components:
        header += f"  {comp_name+' CPU':>14} {comp_name+' GPU':>14} {comp_name+' GEMM':>14}"
    print(header)
    sep = f"  {'─'*6}"
    for _ in present_components:
        sep += f"  {'─'*14} {'─'*14} {'─'*14}"
    print(sep)
    for step in range(1, n_passes):
        row = f"  {step:<6}"
        for comp_name, _ in present_components:
            c = agg[comp_name][step]["cpu"]
            g = agg[comp_name][step]["gpu"]
            m = agg[comp_name][step]["gemm"]
            row += f"  {fmt(c):>14} {fmt(g):>14} {fmt(m):>14}"
        print(row)

# ── GPU miss warning ──────────────────────────────────────────────────────────
total_miss = sum(
    agg[comp][p]["gpu_miss"]
    for comp, _ in present_components
    for p in range(n_passes)
)
if total_miss > 0:
    print(f"\n  ⚠  {total_miss} annotation(s) had no matching GPU event "
          "(GPU and GEMM time shown as 0 for those)")

print(f"\n{'='*W}\n")
