#!/usr/bin/env python3
"""
gantt_timeline.py  —  Execution timeline Gantt charts for prefill and decode.

Figure layout (2 × 2):
  Top row    : CPU module timeline  (prefill | decode)
  Bottom row : GPU kernel timeline  (prefill | decode)

CPU row  — one horizontal track per component type; each bar = one layer occurrence.
GPU row  — one track per kernel category (GEMM / dispatch / other / memcpy).

Both rows share the same x-axis reference: time from CPU pass start in ms.

Usage:
  python3 scripts/gantt_timeline.py --trace results/olmoe_profile_2/inference_trace.json
  python3 scripts/gantt_timeline.py --trace ... --out fig.pdf --no-gpu
"""

import json
import sys
import argparse
import re
import bisect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Component classification ──────────────────────────────────────────────────

COMP_ORDER  = ["attention", "expert", "router", "lm_head"]
COMP_COLORS = {
    "attention": "#4C72B0",
    "expert":    "#DD8452",
    "router":    "#55A868",
    "lm_head":   "#8172B2",
}
COMP_LABELS = {
    "attention": "Attention",
    "expert":    "Expert / MLP",
    "router":    "Router / Gate",
    "lm_head":   "LM Head",
}

def _classify_ann(name):
    if ".attention (" in name:
        return "attention"
    if re.search(r"\.experts?\s+\(", name):
        return "expert"
    m = re.search(r"\.expert\d+\s+\((\w+)\)", name)
    if m and ("MLP" in m.group(1) or "Expert" in m.group(1)):
        return "expert"
    if re.search(r"\.router\s+\(", name):
        return "router"
    m2 = re.search(r"\.mlp\s+\((\w+)\)", name)
    if m2 and "MLP" in m2.group(1):
        return "expert"   # treat dense MLP as expert for timeline purposes
    if name.startswith("lm_head ("):
        return "lm_head"
    return None

# ── Kernel classification ─────────────────────────────────────────────────────

GEMM_KW     = ("cutlass", "gemm", "sgemm", "gemv")
DISPATCH_KW = ("gather", "scatter", "fill", "cub", "scan", "sort",
                "topk", "index_select", "unique")

KRN_ORDER  = ["gemm", "dispatch", "other", "memcpy"]
KRN_COLORS = {
    "gemm":     "#1f77b4",
    "dispatch": "#ff7f0e",
    "other":    "#C0C0C0",
    "memcpy":   "#d62728",
}
KRN_LABELS = {
    "gemm":     "GEMM (static matmul — AIMC candidate)",
    "dispatch": "Dispatch / gather / scatter",
    "other":    "Other (elementwise, norm, …)",
    "memcpy":   "Memcpy / sync (DtoH)",
}

def _classify_krn(e):
    if e.get("cat") == "gpu_memcpy":
        return "memcpy"
    name = e.get("name", "").lower()
    if any(x in name for x in GEMM_KW):
        return "gemm"
    if any(x in name for x in DISPATCH_KW):
        return "dispatch"
    return "other"

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gantt timeline chart for prefill and decode")
parser.add_argument("--trace",        required=True, help="Path to inference_trace.json")
parser.add_argument("--out",          default="gantt.png",
                    help="Output image path (.png or .pdf)")
parser.add_argument("--decode-step",  type=int, default=-1,
                    help="Decode step to show: 1=first, -1=last (default)")
parser.add_argument("--no-gpu",       action="store_true",
                    help="Skip GPU kernel timeline row")
parser.add_argument("--dpi",          type=int, default=150)
args = parser.parse_args()

# ── Load trace ────────────────────────────────────────────────────────────────
print(f"Loading {args.trace}...")
with open(args.trace) as f:
    data = json.load(f)
events = data.get("traceEvents", data)

# ── Pass boundaries ───────────────────────────────────────────────────────────
layer0_anns = sorted(
    [e for e in events
     if e.get("cat") == "user_annotation" and e.get("ph") == "X"
     and re.match(r"^layer0\s+\(", e.get("name", ""))],
    key=lambda e: e["ts"],
)
if not layer0_anns:
    print("ERROR: no 'layer0 (...)' annotations found.")
    sys.exit(1)

n_passes    = len(layer0_anns)
n_decode    = n_passes - 1
pass_starts = [e["ts"] for e in layer0_anns]

# Pass end = start of next pass (or last layer's end for the final pass)
all_layer_anns = sorted(
    [e for e in events
     if e.get("cat") == "user_annotation" and e.get("ph") == "X"
     and re.match(r"^layer\d+\s+\(", e.get("name", ""))],
    key=lambda e: e["ts"],
)
pass_ends = []
for p in range(n_passes):
    if p + 1 < n_passes:
        pass_ends.append(pass_starts[p + 1])
    else:
        in_pass = [e for e in all_layer_anns if e["ts"] >= pass_starts[p]]
        last    = in_pass[-1] if in_pass else layer0_anns[-1]
        pass_ends.append(last["ts"] + last.get("dur", 0))

print(f"Found {n_passes} passes: 1 prefill + {n_decode} decode step(s)")

# Choose which decode step to show
if n_decode == 0:
    decode_idx = None
elif args.decode_step == -1:
    decode_idx = n_passes - 1
else:
    decode_idx = max(1, min(args.decode_step, n_passes - 1))

phases = [("Prefill", 0)]
if decode_idx is not None:
    lbl = "Decode" if n_decode == 1 else f"Decode step {decode_idx}"
    phases.append((lbl, decode_idx))

# ── GPU annotation index ──────────────────────────────────────────────────────
gpu_by_ext_id = {}
for e in events:
    if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X":
        ext_id = e.get("args", {}).get("External id")
        if ext_id is not None and ext_id not in gpu_by_ext_id:
            gpu_by_ext_id[ext_id] = e

sorted_kernels = sorted(
    [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy") and e.get("ph") == "X"],
    key=lambda e: e["ts"],
)
kernel_ts_list = [e["ts"] for e in sorted_kernels]

def _gpu_window_for_pass(pass_idx):
    """GPU time extent for a pass, from the union of all layer GPU annotations."""
    p_start, p_end = pass_starts[pass_idx], pass_ends[pass_idx]
    g_starts, g_ends = [], []
    for e in all_layer_anns:
        if e["ts"] < p_start or e["ts"] >= p_end:
            continue
        ext_id = e.get("args", {}).get("External id")
        if ext_id:
            gpu_ev = gpu_by_ext_id.get(ext_id)
            if gpu_ev:
                g_starts.append(gpu_ev["ts"])
                g_ends.append(gpu_ev["ts"] + gpu_ev["dur"])
    return (min(g_starts), max(g_ends)) if g_starts else (None, None)

def _kernels_for_pass(pass_idx):
    gpu_start, gpu_end = _gpu_window_for_pass(pass_idx)
    if gpu_start is None:
        return []
    lo = bisect.bisect_left(kernel_ts_list, gpu_start - 5)
    hi = bisect.bisect_right(kernel_ts_list, gpu_end + 5)
    return [e for e in sorted_kernels[lo:hi]
            if e["ts"] + e.get("dur", 0) <= gpu_end + 5]

# ── Determine which component tracks are present ──────────────────────────────
present_comps = set()
for _, p in phases:
    p_start, p_end = pass_starts[p], pass_ends[p]
    for e in events:
        if e.get("cat") != "user_annotation" or e.get("ph") != "X":
            continue
        if e["ts"] < p_start or e["ts"] >= p_end:
            continue
        c = _classify_ann(e.get("name", ""))
        if c:
            present_comps.add(c)

comp_tracks = [c for c in COMP_ORDER if c in present_comps]
# y-position: top of list = highest y value
comp_y = {c: i for i, c in enumerate(reversed(comp_tracks))}

# ── Figure setup ──────────────────────────────────────────────────────────────
n_cols = len(phases)
n_rows = 1 if args.no_gpu else 2
height_ratios = ([3, 2] if not args.no_gpu else [1])

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(9 * n_cols, 3.5 * n_rows + 0.5),
    gridspec_kw={"height_ratios": height_ratios},
    squeeze=False,
)
fig.subplots_adjust(hspace=0.45, wspace=0.06)

BAR_H   = 0.55
US_TO_MS = 1 / 1000   # convert μs → ms for readable x-axis ticks

# ── Plot each column (phase) ──────────────────────────────────────────────────
for col, (phase_label, pass_idx) in enumerate(phases):
    p_start = pass_starts[pass_idx]
    p_end   = pass_ends[pass_idx]

    # ── Row 0: CPU module timeline ────────────────────────────────────────────
    ax = axes[0][col]

    for e in events:
        if e.get("cat") != "user_annotation" or e.get("ph") != "X":
            continue
        if e["ts"] < p_start or e["ts"] >= p_end:
            continue
        c = _classify_ann(e.get("name", ""))
        if c is None:
            continue
        x_ms = (e["ts"] - p_start) * US_TO_MS
        w_ms = e.get("dur", 0)  * US_TO_MS
        ax.barh(comp_y[c], w_ms, left=x_ms, height=BAR_H,
                color=COMP_COLORS[c], alpha=0.85, linewidth=0)

    ax.set_yticks(range(len(comp_tracks)))
    ax.set_yticklabels(
        [COMP_LABELS.get(c, c) for c in reversed(comp_tracks)],
        fontsize=9,
    )
    ax.set_ylim(-0.6, len(comp_tracks) - 0.4)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_title(f"{phase_label}  —  CPU module timeline", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if col > 0:
        ax.set_yticklabels([])

    # ── Row 1: GPU kernel timeline ────────────────────────────────────────────
    if not args.no_gpu:
        ax_g = axes[1][col]
        kernels = _kernels_for_pass(pass_idx)

        # Reference: use CPU pass start so the two rows share the same time origin
        for e in kernels:
            k    = _classify_krn(e)
            x_ms = (e["ts"] - p_start) * US_TO_MS
            w_ms = e.get("dur", 0) * US_TO_MS
            y    = KRN_ORDER.index(k)
            ax_g.barh(y, w_ms, left=x_ms, height=BAR_H,
                      color=KRN_COLORS[k], alpha=0.85, linewidth=0)

        ax_g.set_yticks(range(len(KRN_ORDER)))
        ax_g.set_yticklabels(
            [KRN_LABELS.get(k, k) for k in KRN_ORDER],
            fontsize=8,
        )
        ax_g.set_ylim(-0.6, len(KRN_ORDER) - 0.4)
        ax_g.set_xlabel("Time (ms)", fontsize=9)
        ax_g.set_title(f"{phase_label}  —  GPU kernel timeline", fontsize=11, fontweight="bold")
        ax_g.spines[["top", "right"]].set_visible(False)
        ax_g.tick_params(axis="y", length=0)
        if col > 0:
            ax_g.set_yticklabels([])

        # Align x-axis with the CPU row above
        ax_g.set_xlim(axes[0][col].get_xlim())

# ── Legends ───────────────────────────────────────────────────────────────────
cpu_patches = [
    mpatches.Patch(color=COMP_COLORS[c], label=COMP_LABELS.get(c, c))
    for c in comp_tracks
]
axes[0][-1].legend(handles=cpu_patches, loc="upper right",
                   fontsize=8, framealpha=0.8, edgecolor="none")

if not args.no_gpu:
    gpu_patches = [
        mpatches.Patch(color=KRN_COLORS[k], label=KRN_LABELS.get(k, k))
        for k in KRN_ORDER
    ]
    axes[1][-1].legend(handles=gpu_patches, loc="upper right",
                       fontsize=8, framealpha=0.8, edgecolor="none")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path(args.out)
fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
print(f"Saved → {out_path}")
