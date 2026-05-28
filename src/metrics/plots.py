"""
Publication-quality plots for MoE AIMC workload metrics.

Reads from the metrics/raw/*.json files produced by workload_metrics.py and
outputs PDF + PNG plots suitable for a thesis or paper.

Usage (CLI):
    python -m src.metrics.plots metrics/raw/ --output-dir metrics/plots/ --style thesis_print

Usage (API):
    from src.metrics.plots import plot_all
    plot_all(Path("metrics/raw"), Path("metrics/plots"), config={"style": "thesis_print"})
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    warnings.warn("seaborn not installed — heatmaps will use matplotlib directly", stacklevel=2)

# ─── Color palette (Okabe-Ito, universally colorblind-safe) ──────────────────

CB = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "sky":        "#56B4E9",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "black":      "#000000",
    "gray":       "#999999",
    "light_gray": "#CCCCCC",
    "red":        "#BB3322",
}

# Op-type → color  (consistent across all plots)
OP_COLOR = {
    "attention":        CB["sky"],
    "attention_matmul": CB["sky"],
    "expert_ffn":       CB["purple"],
    "gate":             CB["orange"],
    "lm_head":          CB["green"],
    "other":            CB["gray"],
    "other_matmul":     CB["gray"],
    "analog":           CB["blue"],
    "digital":          CB["gray"],
}

# ─── Style definitions ────────────────────────────────────────────────────────

_BASE = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
}

STYLES: dict[str, dict[str, Any]] = {
    "thesis_print": {
        **_BASE,
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.linewidth":    1.2,
        "lines.linewidth":   1.5,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         False,
    },
    "thesis_digital": {
        **_BASE,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.linewidth":    1.0,
        "lines.linewidth":   1.5,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f7f7f7",
        "axes.grid":         True,
        "axes.grid.alpha":   0.4,
        "axes.grid.color":   "#cccccc",
    },
    "presentation": {
        **_BASE,
        "font.family":       "sans-serif",
        "font.size":         14,
        "axes.titlesize":    16,
        "axes.labelsize":    14,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "axes.linewidth":    1.8,
        "lines.linewidth":   2.5,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "axes.grid.alpha":   0.3,
        "savefig.pad_inches": 0.2,
    },
}

DEFAULT_PLOT_CONFIG: dict[str, Any] = {
    "style":                    "thesis_print",
    "dpi":                      300,
    "min_viable_batch":         4,          # tokens/expert below which AIMC efficiency degrades
    "max_gantt_layers":         16,         # layers shown in Gantt chart (None = all)
    "max_bar_layers":           16,         # layers shown in grouped bar chart
    # AIMC bubble chart (plot_numels_flops_per_step)
    "aimc_ridge_point":         21.8,       # FLOP/numel ridge (peak_compute / peak_bw)
    "aimc_min_flops_threshold": None,       # None = auto (median total FLOPs)
    # AIMC candidates plot (plot_aimc_candidates)
    "ridge_point":              21.8,       # FLOP/numel ridge point
    "min_tile_numels":          500_000,    # weights smaller than this skip AIMC
    "model_name":               "MoE Model",
    "hardware_name":            "Target Hardware",
}

# ─── Shared helpers ───────────────────────────────────────────────────────────

def apply_style(config: dict[str, Any]) -> None:
    """Update global rcParams for the chosen style."""
    name = config.get("style", "thesis_print")
    plt.rcParams.update(STYLES.get(name, STYLES["thesis_print"]))


def _save(fig: plt.Figure, name: str, out: Path, cfg: dict[str, Any]) -> None:
    dpi = cfg.get("dpi", 300)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)


def _load(raw_dir: Path, name: str) -> dict[str, Any] | None:
    p = raw_dir / f"{name}.json"
    if not p.exists():
        print(f"  [skip] {name}.json not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# HuggingFace SwiGLU names → canonical llama.cpp / paper names
_WEIGHT_ALIASES: dict[str, str] = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj":   "w3",
}


def _weight_suffix(module: str) -> str:
    """Extract terminal weight name from a module path, normalizing HF names to w1/w2/w3."""
    m = re.search(r"\.(w\d+|gate_proj|up_proj|down_proj|gate|q_proj|k_proj|v_proj|o_proj|lm_head)$", module)
    name = m.group(1) if m else module.rsplit(".", 1)[-1]
    return _WEIGHT_ALIASES.get(name, name)


_LAYER_RE_PLOT  = re.compile(r"layers\.(\d+)")
_EXPERT_RE_PLOT = re.compile(r"experts\.(\d+)")


def _point_label(module: str) -> str:
    """Compact annotation: 'L13 E5 w1', 'L0 q_proj', 'lm_head', etc."""
    layer_m  = _LAYER_RE_PLOT.search(module)
    expert_m = _EXPERT_RE_PLOT.search(module)
    parts = []
    if layer_m:
        parts.append(f"L{layer_m.group(1)}")
    if expert_m:
        parts.append(f"E{expert_m.group(1)}")
    parts.append(_weight_suffix(module))
    return " ".join(parts)


def _layer_cmap(n: int) -> list[str]:
    """Return n distinct colors for layer-level coloring."""
    base = plt.get_cmap("tab20")
    return [mcolors.to_hex(base(i / max(n - 1, 1))) for i in range(n)]


# ─── Plot 1 — Arithmetic Intensity Distribution ───────────────────────────────

def plot_arithmetic_intensity(data: dict, cfg: dict, out: Path) -> None:
    """Histogram of arithmetic intensity per op category with AIMC threshold line."""
    threshold = data["config"]["aimc_roofline_threshold"]
    cats_to_show = ["attention_matmul", "expert_matmul", "gate"]
    cats = {k: v for k, v in data["categories"].items() if k in cats_to_show and v["values"]}
    if not cats:
        cats = {k: v for k, v in data["categories"].items() if v["values"]}

    n = len(cats)
    if n == 0:
        print("  [skip] arithmetic_intensity: no data"); return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    cat_labels = {
        "attention_matmul": "Attention Matmuls",
        "expert_matmul":    "Expert FFN Matmuls",
        "gate":             "Gate / Routing",
        "other":            "Other",
        "other_matmul":     "Other Matmuls",
    }

    for ax, (cat, d) in zip(axes, cats.items()):
        vals = np.array(d["values"], dtype=float)
        bins = np.logspace(np.log10(max(vals.min(), 1e-3)), np.log10(vals.max() * 1.05), 30)

        below = vals[vals < threshold]
        above = vals[vals >= threshold]
        if len(below):
            ax.hist(below, bins=bins, color=CB["red"],    alpha=0.85, label=f"< {threshold}")
        if len(above):
            ax.hist(above, bins=bins, color=CB["green"],  alpha=0.85, label=f"≥ {threshold}")

        ax.axvline(threshold, color=CB["black"], ls="--", lw=1.4, label=f"Threshold ({threshold})")
        ax.set_xscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/numel)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(cat_labels.get(cat, cat), fontsize=13)
        ax.legend(fontsize=9)

        s = d["stats"]
        ax.text(0.97, 0.97,
                f"mean={s['mean']:.2f}\np50={s['p50']:.2f}\np95={s['p95']:.2f}\n"
                f"below={d['below_threshold_fraction']*100:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.suptitle("Arithmetic Intensity Distribution by Op Category", fontsize=13, y=1.01)
    _save(fig, "arithmetic_intensity", out, cfg)


# ─── Plot 2 — Tile Utilization Heatmap ───────────────────────────────────────

def plot_tile_utilization(data: dict, cfg: dict, out: Path) -> None:
    """
    Heatmap of mean tile utilization: rows = weight type, columns = layer.
    Cells annotated with total tile count.
    """
    ops = data["ops"]
    if not ops:
        print("  [skip] tile_mapping: no ops"); return

    # Build (weight_type, layer) → (util_sum, tiles_sum, count)
    table: dict[tuple[str, int], list[float]] = {}
    tile_count: dict[tuple[str, int], int] = {}

    for op in ops:
        lid = op.get("layer_id")
        if lid is None:
            continue
        wt = _weight_suffix(op["module"])
        key = (wt, int(lid))
        table.setdefault(key, []).append(op["tile_utilization"])
        tile_count[key] = tile_count.get(key, 0) + op["tiles_needed"]

    # Desired row order: w1, w2, w3 first, then attention, then others
    wt_order = ["w1", "w2", "w3", "gate", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
    all_wts = sorted({k[0] for k in table},
                     key=lambda w: wt_order.index(w) if w in wt_order else 99)
    all_lids = sorted({k[1] for k in table})

    n_rows, n_cols = len(all_wts), len(all_lids)
    util_mat = np.full((n_rows, n_cols), np.nan)
    tile_mat = np.zeros((n_rows, n_cols), dtype=int)

    for ri, wt in enumerate(all_wts):
        for ci, lid in enumerate(all_lids):
            key = (wt, lid)
            if key in table:
                util_mat[ri, ci] = float(np.mean(table[key]))
                tile_mat[ri, ci] = tile_count[key]

    util_cmap = mcolors.LinearSegmentedColormap.from_list(
        "util_wr", ["white", CB["vermillion"]], N=256
    )

    fig_w = max(8, n_cols * 0.35 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(util_mat, cmap=util_cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells with tile count (skip NaN)
    for ri in range(n_rows):
        for ci in range(n_cols):
            if not np.isnan(util_mat[ri, ci]):
                tc = tile_mat[ri, ci]
                txt = f"{tc:,}" if tc < 10_000 else f"{tc//1000}k"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=6,
                        color="black" if util_mat[ri, ci] < 0.7 else "white")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(l) for l in all_lids], fontsize=8,
                       rotation=90 if n_cols > 20 else 0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(all_wts, fontsize=9)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Weight Type", fontsize=11)
    ax.set_title(f"Tile Utilization per Weight Type × Layer  (tile size {data['config']['tile_size']})", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Tile Utilization", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    _save(fig, "tile_utilization", out, cfg)


# ─── Plot 3 — Load Imbalance per Layer ───────────────────────────────────────

def plot_load_imbalance(data: dict, cfg: dict, out: Path) -> None:
    """
    Grouped bar chart (subset of layers) + secondary imbalance ratio line across all layers.
    """
    per_layer = data["per_layer"]
    if not per_layer:
        print("  [skip] load_imbalance: no data"); return

    sorted_ids = sorted(per_layer.keys(), key=lambda x: int(x))
    max_bars = cfg.get("max_bar_layers", DEFAULT_PLOT_CONFIG["max_bar_layers"])

    # Sample layers for bar chart
    if len(sorted_ids) > max_bars:
        step = len(sorted_ids) // max_bars
        bar_ids = sorted_ids[::step][:max_bars]
        subset_note = f" (every {step}th layer)"
    else:
        bar_ids = sorted_ids
        subset_note = ""

    # Expert ids
    expert_ids = sorted({int(k) for lid in bar_ids
                         for k in per_layer[lid]["expert_activation_rate"].keys()})
    n_exp = len(expert_ids)
    if n_exp == 0:
        print("  [skip] load_imbalance: no experts"); return

    fig, (ax_bar, ax_ratio) = plt.subplots(2, 1, figsize=(12, 7),
                                            gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.35)

    # Bar chart
    x = np.arange(len(bar_ids))
    bar_w = 0.8 / n_exp
    colors = _layer_cmap(n_exp)

    for ei, eid in enumerate(expert_ids):
        rates = [
            per_layer[lid]["expert_activation_rate"].get(str(eid), 0.0)
            for lid in bar_ids
        ]
        ax_bar.bar(x + ei * bar_w, rates, bar_w * 0.9,
                   color=colors[ei], label=f"Expert {eid}", alpha=0.85)

    mean_vals = [per_layer[lid]["mean_rate"] for lid in bar_ids]
    ax_bar.step(x + (n_exp - 1) * bar_w / 2, mean_vals,
                where="mid", color=CB["black"], lw=1.5, ls="--", label="Mean rate")

    ax_bar.set_xticks(x + (n_exp - 1) * bar_w / 2)
    ax_bar.set_xticklabels([f"L{lid}" for lid in bar_ids],
                           fontsize=8, rotation=45 if len(bar_ids) > 10 else 0)
    ax_bar.set_xlabel(f"Layer{subset_note}", fontsize=11)
    ax_bar.set_ylabel("Activation Rate (activations / routings)", fontsize=11)
    ax_bar.set_title("Expert Activation Rate per Layer", fontsize=13)
    ax_bar.legend(fontsize=8, ncol=min(n_exp, 8), loc="upper right")

    # Imbalance ratio across ALL layers
    all_ratios = [per_layer[lid]["load_imbalance_ratio"] for lid in sorted_ids]
    all_x = range(len(sorted_ids))
    ax_ratio.plot(all_x, all_ratios, color=CB["vermillion"], lw=1.8, marker="o",
                  ms=3, label="Imbalance ratio")
    ax_ratio.axhline(1.0, color=CB["black"], ls="--", lw=1.0, alpha=0.6, label="Perfect balance")
    ax_ratio.set_xticks(range(0, len(sorted_ids), max(1, len(sorted_ids) // 10)))
    ax_ratio.set_xticklabels(sorted_ids[::max(1, len(sorted_ids) // 10)], fontsize=8)
    ax_ratio.set_xlabel("Layer Index", fontsize=11)
    ax_ratio.set_ylabel("max / mean", fontsize=11)
    ax_ratio.set_title("Load Imbalance Ratio (all layers)", fontsize=11)
    ax_ratio.legend(fontsize=9)

    _save(fig, "load_imbalance", out, cfg)


# ─── Plot 4 — Digital vs Analog FLOP Split ───────────────────────────────────

def plot_digital_analog_split(data: dict, cfg: dict, out: Path) -> None:
    """Stacked horizontal bar chart: analog (blue) vs digital (gray) per layer + overall."""
    per_layer = data["per_layer"]
    overall   = data["overall"]
    if not per_layer:
        print("  [skip] digital_analog_split: no data"); return

    sorted_ids = sorted(per_layer.keys(), key=lambda x: int(x))
    # Add 'Overall' at the bottom
    labels    = [f"L{lid}" for lid in sorted_ids] + ["Overall"]
    analog_v  = [per_layer[lid]["analog_flops"]  for lid in sorted_ids] + [overall["analog_flops"]]
    digital_v = [per_layer[lid]["digital_flops"] for lid in sorted_ids] + [overall["digital_flops"]]
    ratios    = [per_layer[lid]["analog_ratio"]  for lid in sorted_ids] + [overall["analog_ratio"]]
    totals    = [a + d for a, d in zip(analog_v, digital_v)]

    # Normalise to 100%
    analog_pct  = [a / t * 100 if t else 0 for a, t in zip(analog_v, totals)]
    digital_pct = [d / t * 100 if t else 0 for d, t in zip(digital_v, totals)]

    n = len(labels)
    fig_h = max(5, n * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y = np.arange(n)
    ax.barh(y, analog_pct,  height=0.7, color=CB["blue"],  label="Analog (static-weight matmul)", alpha=0.9)
    ax.barh(y, digital_pct, height=0.7, color=CB["gray"],  label="Digital (other ops)",           alpha=0.9,
            left=analog_pct)

    # Annotate analog %
    for yi, (ap, r) in enumerate(zip(analog_pct, ratios)):
        if ap > 5:
            ax.text(ap / 2, yi, f"{r*100:.1f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Share of FLOPs (%)", fontsize=11)
    ax.set_title("Digital vs Analog FLOP Split per Layer", fontsize=13)
    ax.set_xlim(0, 100)
    ax.axhline(len(sorted_ids) - 0.5, color=CB["black"], lw=1.0, ls="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    # Invert so layer 0 is at the top
    ax.invert_yaxis()
    _save(fig, "digital_analog_split", out, cfg)


# ─── Plot 5 — Batch Size CDF ─────────────────────────────────────────────────

def plot_batch_size_cdf(data: dict, cfg: dict, out: Path) -> None:
    """CDF of tokens per expert activation with percentile markers and AIMC viability shade."""
    cdf_pts = data.get("cdf", [])
    pcts    = data.get("percentiles", {})
    if not cdf_pts:
        print("  [skip] batch_size_distribution: no CDF data"); return

    min_viable = cfg.get("min_viable_batch", DEFAULT_PLOT_CONFIG["min_viable_batch"])
    xs = [pt[0] for pt in cdf_pts]
    ys = [pt[1] for pt in cdf_pts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color=CB["blue"], lw=2.0, label="CDF")

    # AIMC viability shade
    ax.axvspan(min(xs) - 0.5, min_viable, alpha=0.12, color=CB["red"],
               label=f"Below AIMC viable batch ({min_viable} tokens)")

    # Percentile lines
    pct_styles = {
        "p10": (CB["gray"],       ":"),
        "p25": (CB["orange"],     "--"),
        "p50": (CB["vermillion"], "-"),
        "p75": (CB["orange"],     "--"),
        "p90": (CB["gray"],       ":"),
    }
    for pname, (color, ls) in pct_styles.items():
        v = pcts.get(pname)
        if v is not None:
            ax.axvline(v, color=color, ls=ls, lw=1.2, alpha=0.9, label=f"{pname.upper()}={v:.0f}")

    ax.set_xlabel("Tokens per Expert Activation", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("Effective Batch Size CDF (Expert Activations)", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=max(0, min(xs) - 0.5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, "batch_size_cdf", out, cfg)


# ─── Plot 6 — Expert Activation Heatmap ──────────────────────────────────────

def plot_expert_activation_heatmap(data: dict, cfg: dict, out: Path) -> None:
    """Heatmap of activation frequency [n_layers × n_experts]."""
    n_layers  = data["n_layers"]
    n_experts = data["n_experts"]
    freq_mat  = np.array(data["frequency"], dtype=float)   # (n_layers, n_experts)
    layer_names = data["layer_names"]

    if n_layers == 0 or n_experts == 0:
        print("  [skip] expert_activation_frequency: empty matrix"); return

    # Use raw heatmap counts (already as float in frequency = count / n_requests)
    fig_w = max(7, n_experts * 0.55 + 2.5)
    fig_h = max(5, n_layers * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(min(fig_w, 18), min(fig_h, 20)))

    purple_cmap = plt.get_cmap("Purples")

    if _HAS_SNS:
        # Annotate only when the matrix is small enough to be readable
        annot = (n_layers * n_experts) <= 200
        sns.heatmap(freq_mat, ax=ax,
                    cmap=purple_cmap, linewidths=0.3 if annot else 0,
                    annot=annot, fmt=".0f", annot_kws={"size": 7},
                    cbar_kws={"label": "Activations / Request"},
                    xticklabels=range(n_experts),
                    yticklabels=layer_names if n_layers <= 40 else False)
    else:
        im = ax.imshow(freq_mat, cmap=purple_cmap, aspect="auto")
        fig.colorbar(im, ax=ax, label="Activations / Request")
        ax.set_xticks(range(n_experts))
        ax.set_yticks(range(n_layers))
        if n_layers <= 40:
            ax.set_yticklabels(layer_names, fontsize=8)

    ax.set_xlabel("Expert ID", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        f"Expert Activation Frequency  ({n_layers} layers × {n_experts} experts, "
        f"n={data['total_requests']} requests)",
        fontsize=13
    )
    _save(fig, "expert_activation_heatmap", out, cfg)


# ─── Plot 7 — Roofline Plot ───────────────────────────────────────────────────

def plot_roofline(data: dict, cfg: dict, out: Path) -> None:
    """
    Traditional roofline plot.

    Each op is placed at its performance ceiling:
        y = min(peak_compute, peak_bw × intensity)

    Ops left of the ridge point are memory-bound; right are compute-bound.
    Dots sitting on the line is expected — without measured execution time this
    shows the theoretical maximum each op can achieve on this hardware.
    """
    ops          = data.get("ops", [])
    peak_compute = data["config"]["peak_compute_tops"]
    peak_bw      = data["config"]["peak_bandwidth_tbs"]
    ridge        = data["ridge_point"]

    if not ops:
        print("  [skip] roofline: no ops"); return

    cat_data: dict[str, tuple[list[float], list[float]]] = {
        "attention":  ([], []),
        "expert_ffn": ([], []),
        "gate":       ([], []),
        "lm_head":    ([], []),
        "other":      ([], []),
    }
    cat_labels = {
        "attention":  "Attention",
        "expert_ffn": "Expert FFN (w1/w2/w3)",
        "gate":       "Gate/Routing",
        "lm_head":    "LM Head",
        "other":      "Other",
    }

    for op in ops:
        intensity = op["intensity"]
        flops     = op.get("flops", 0)
        if intensity <= 0 or flops <= 0:
            continue
        role = op["role_category"]

        if role == "expert_ffn":
            key = "expert_ffn"
        elif role == "attention":
            key = "attention"
        elif role == "gate":
            key = "gate"
        elif role == "lm_head":
            key = "lm_head"
        else:
            key = "other"

        cat_data[key][0].append(intensity)
        cat_data[key][1].append(flops)

    all_intensities = [x for xs, _ in cat_data.values() for x in xs]
    all_flops       = [y for _, ys in cat_data.values() for y in ys]
    if not all_intensities:
        print("  [skip] roofline: all zero intensities"); return

    x_min = 10 ** (np.floor(np.log10(min(all_intensities))) - 0.3)
    x_max = 10 ** (np.ceil(np.log10(max(all_intensities)))  + 0.5)
    y_min = 10 ** (np.floor(np.log10(min(all_flops)))       - 0.3)
    y_max = 10 ** (np.ceil(np.log10(max(all_flops)))        + 0.3)

    # Roofline boundary curve — drawn as a vertical guide at the ridge point
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    y_line = np.minimum(peak_compute, peak_bw * x_line)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(x_line, y_line, color=CB["black"], lw=2.0, zorder=2, label="Roofline")

    # Ridge marker
    ax.axvline(ridge, color=CB["black"], lw=0.9, ls=":", alpha=0.5, zorder=4)
    ax.text(ridge * 1.06, y_max * 0.6,
            f"Ridge\n{ridge:.1f} FLOP/el", fontsize=8, color=CB["black"], alpha=0.7)

    # Region labels
    ax.text(x_min * 1.5, y_min * 2, "Memory-bound", fontsize=9, color=CB["black"], alpha=0.5)
    ax.text(ridge * 1.1,  y_min * 2, "Compute-bound", fontsize=9, color=CB["black"], alpha=0.5)

    for key, (xs, ys) in cat_data.items():
        if not xs:
            continue
        color = OP_COLOR.get(key, CB["gray"])
        ax.scatter(xs, ys, c=color, s=8, alpha=0.35, rasterized=True,
                   label=f"{cat_labels[key]} (n={len(xs)})", zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOP/numel)", fontsize=11)
    ax.set_ylabel("Total FLOPs", fontsize=11)
    ax.set_title(
        f"Roofline  —  peak={peak_compute} TOPS,  BW={peak_bw} TB/s,  "
        f"ridge={ridge:.1f} FLOP/numel",
        fontsize=12,
    )
    ax.legend(fontsize=9, markerscale=3, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    _save(fig, "roofline", out, cfg)


# ─── Plot 7b — AIMC Candidate Bubble Chart ───────────────────────────────────

def plot_numels_flops_per_step(data: dict, cfg: dict, out: Path) -> None:
    """
    Bubble scatter: one dot per unique weight matrix.

    x  = arithmetic intensity (FLOP/numel, log) = flops_per_call / numels_per_call
    y  = total FLOPs across all forward passes  (log)
    sz = number of times the weight was called  (bubble size, 20–500 px²)

    Shaded quadrants show AIMC mapping quality:
      top-left   (memory-bound, high compute) — best AIMC candidates
      top-right  (compute-bound, high compute) — ideal AIMC
      bottom-left  (memory-bound, low compute) — poor candidates
      bottom-right (compute-bound, low compute) — compute-bound low usage
    """
    ops = data.get("ops", [])
    if not ops:
        print("  [skip] numels_flops_per_step: no ops"); return

    valid = [op for op in ops
             if op.get("arithmetic_intensity", 0) > 0 and op.get("total_flops", 0) > 0]
    if not valid:
        print("  [skip] numels_flops_per_step: no valid ops"); return

    ridge      = float(cfg.get("aimc_ridge_point", 21.8))
    flops_thresh = cfg.get("aimc_min_flops_threshold", None)

    intensities = np.array([op["arithmetic_intensity"] for op in valid])
    total_flops = np.array([op["total_flops"]          for op in valid])
    n_calls_arr = np.array([op["n_calls"]              for op in valid], dtype=float)
    roles       = [op.get("role_category", "other")    for op in valid]
    modules     = [op.get("module", "")                for op in valid]

    # Auto-threshold at median total FLOPs if not configured
    if flops_thresh is None:
        flops_thresh = float(np.median(total_flops))

    # Bubble size: log-normalised to [20, 500]
    log_n = np.log1p(n_calls_arr)
    if log_n.max() > log_n.min():
        sizes = 20.0 + 480.0 * (log_n - log_n.min()) / (log_n.max() - log_n.min())
    else:
        sizes = np.full(len(valid), 150.0)

    x_min = 10 ** (np.floor(np.log10(intensities.min())) - 0.3)
    x_max = 10 ** (np.ceil( np.log10(intensities.max())) + 0.5)
    y_min = 10 ** (np.floor(np.log10(total_flops.min())) - 0.3)
    y_max = 10 ** (np.ceil( np.log10(total_flops.max())) + 0.3)

    # Clamp threshold so quadrant fill stays inside axes
    thresh_y = max(y_min * 2, min(flops_thresh, y_max / 2))

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── Shaded quadrants ────────────────────────────────────────────────────
    ax.fill_between([x_min, ridge],  thresh_y, y_max,
                    color="#FF8C00", alpha=0.08, zorder=0)   # top-left  orange
    ax.fill_between([ridge,  x_max], thresh_y, y_max,
                    color="#00AA00", alpha=0.08, zorder=0)   # top-right green
    ax.fill_between([x_min, ridge],  y_min, thresh_y,
                    color="#DD2222", alpha=0.08, zorder=0)   # btm-left  red
    ax.fill_between([ridge,  x_max], y_min, thresh_y,
                    color="#888888", alpha=0.08, zorder=0)   # btm-right gray

    # Quadrant text — geometric centre of each region
    def _gx(lo, hi):
        return 10 ** ((np.log10(lo) + np.log10(hi)) / 2)
    def _gy(lo, hi):
        return 10 ** ((np.log10(lo) + np.log10(hi)) / 2)

    for (tx, ty, label, color) in [
        (_gx(x_min, ridge),  _gy(thresh_y, y_max),  "Memory Bound\nHigh Value",    "#AA5500"),
        (_gx(ridge,  x_max), _gy(thresh_y, y_max),  "Ideal AIMC",                  "#005500"),
        (_gx(x_min, ridge),  _gy(y_min, thresh_y),  "Poor Candidates",             "#AA0000"),
        (_gx(ridge,  x_max), _gy(y_min, thresh_y),  "Compute Bound\nLow Usage",    "#555555"),
    ]:
        ax.text(tx, ty, label, fontsize=8, color=color, alpha=0.75,
                ha="center", va="center", zorder=1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5))

    # ── Reference lines ─────────────────────────────────────────────────────
    ax.axvline(ridge, color=CB["black"], lw=1.4, ls="--", alpha=0.6, zorder=2)
    ax.axhline(flops_thresh, color=CB["black"], lw=1.0, ls="--", alpha=0.4, zorder=2)

    ax.text(ridge * 1.06, y_min * 2.5,
            f"Ridge  {ridge:.1f} FLOP/el\n← memory  |  compute →",
            fontsize=8, color=CB["black"], alpha=0.65, va="bottom")

    # ── Scatter by category ─────────────────────────────────────────────────
    cat_order  = ["attention", "expert_ffn", "gate", "lm_head", "other"]
    cat_labels = {
        "attention":  "Attention (q/k/v/o)",
        "expert_ffn": "Expert FFN (w1/w2/w3)",
        "gate":       "Gate / Routing",
        "lm_head":    "LM Head",
        "other":      "Other",
    }
    cat_handles = []
    for cat in cat_order:
        idx = [i for i, r in enumerate(roles)
               if r == cat or (cat == "other" and r not in cat_labels)]
        if not idx:
            continue
        color = OP_COLOR.get(cat, CB["gray"])
        sc = ax.scatter(
            intensities[idx], total_flops[idx],
            s=sizes[idx], c=color, alpha=0.72,
            edgecolors="white", linewidths=0.5, zorder=3,
            label=cat_labels.get(cat, cat),
        )
        cat_handles.append(sc)

    # ── Annotate extreme / high-value points ────────────────────────────────
    annotated: set[int] = set()

    def _label(i: int) -> None:
        if i in annotated:
            return
        annotated.add(i)
        ax.annotate(
            _point_label(modules[i]),
            xy=(intensities[i], total_flops[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color=CB["black"], alpha=0.85,
        )

    _label(int(np.argmax(total_flops)))
    _label(int(np.argmin(intensities)))
    _label(int(np.argmax(intensities)))
    # Best AIMC candidates: top-left, sorted by total FLOPs
    top_left = sorted(
        [i for i in range(len(valid))
         if intensities[i] < ridge and total_flops[i] > flops_thresh],
        key=lambda i: -total_flops[i],
    )
    for i in top_left[:5]:
        _label(i)

    # ── Axes & labels ────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOP/numel)", fontsize=11)
    ax.set_ylabel("Total FLOPs Across All Forward Passes", fontsize=11)
    ax.set_title("AIMC Mapping Candidates — Intensity vs Total Compute", fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.2)

    # Color legend (upper-left)
    color_leg = ax.legend(
        handles=cat_handles, fontsize=9, loc="upper left",
        title="Weight type", title_fontsize=9,
        framealpha=0.85,
    )
    ax.add_artist(color_leg)

    # Bubble-size legend (lower-right)
    n_min = int(n_calls_arr.min())
    n_max = int(n_calls_arr.max())
    size_handles = []
    for n_val, lbl in [
        (n_min, f"{n_min} calls"),
        (int(np.sqrt(n_min * n_max)) if n_min < n_max else n_min, ""),
        (n_max, f"{n_max} calls"),
    ]:
        if log_n.max() > log_n.min():
            norm = (np.log1p(n_val) - log_n.min()) / (log_n.max() - log_n.min())
            sz = 20.0 + 480.0 * float(np.clip(norm, 0, 1))
        else:
            sz = 150.0
        if lbl:
            size_handles.append(
                plt.scatter([], [], s=sz, c=CB["gray"], alpha=0.65, label=lbl)
            )
    if size_handles:
        ax.legend(
            handles=size_handles, fontsize=8, loc="lower right",
            title="Calls per trace", title_fontsize=8,
            framealpha=0.85,
        )

    _save(fig, "numels_flops_per_step", out, cfg)


# ─── Plot 7c — AIMC Tile Mapping Candidates ──────────────────────────────────

def plot_aimc_candidates(data: dict, cfg: dict, out: Path) -> None:
    """
    Bubble scatter identifying weight matrices worth mapping to AIMC tiles.

    Data source: roofline.json (per-weight profile with call counts).

    x  = total FLOPs across all forward passes  (log scale)
    y  = arithmetic intensity = total_flops / weight_numels  (FLOP/numel, log scale)
    sz = n_calls (bubble size, 20–500 px²)

    A weight is an AIMC candidate if ALL three hold:
      intensity > ridge_point     (high compute per numel of weight — worth tiling)
      total_flops > median_flops  (high-value computation)
      weight_numels > min_tile_numels  (large enough to justify tile area)

    Candidates drawn with a thick green border.
    Three quadrant labels placed in corners (no background shading).
    """
    ops = data.get("ops", [])
    if not ops:
        print("  [skip] aimc_candidates: no ops"); return

    valid = [
        op for op in ops
        if op.get("intensity", 0) > 0
        and op.get("flops", 0) > 0
        and op.get("n_calls", 0) > 0
    ]
    if not valid:
        print("  [skip] aimc_candidates: no valid ops (missing n_calls — re-run metrics)"); return

    model_name    = cfg.get("model_name", "MoE Model")
    hardware_name = cfg.get("hardware_name", "Target Hardware")
    n_sequences   = max(int(data.get("n_sequences", 1)), 1)

    # x: total_flops / weight_numels  (cumulative intensity, how much compute per element provisioned)
    # Both divided by n_sequences to give per-inference averages.
    intensities      = np.array([op["intensity"] / n_sequences for op in valid])
    total_flops      = np.array([op["flops"]     / n_sequences for op in valid])
    n_calls_arr       = np.array([op["n_calls"]               for op in valid], dtype=float)
    weight_numels_arr = np.array([op.get("weight_numels", 0) for op in valid])
    roles             = [op.get("role_category", "other")    for op in valid]
    modules          = [op.get("module", "")                for op in valid]

    n_fwd_passes = float(n_calls_arr.max()) if len(n_calls_arr) > 0 else 1.0
    pct_arr = n_calls_arr / n_fwd_passes * 100.0

    # Bubble sizes: log-normalised to [20, 500] using call-frequency percentage
    log_pct = np.log1p(pct_arr)
    if log_pct.max() > log_pct.min():
        sizes = 20.0 + 480.0 * (log_pct - log_pct.min()) / (log_pct.max() - log_pct.min())
    else:
        sizes = np.full(len(valid), 150.0)

    x_min = 10 ** (np.floor(np.log10(intensities.min()))  - 0.3)
    x_max = 10 ** (np.ceil( np.log10(intensities.max()))  + 0.5)
    y_min = 10 ** (np.floor(np.log10(total_flops.min()))  - 0.3)
    y_max = 10 ** (np.ceil( np.log10(total_flops.max()))  + 0.3)

    _CAT_COLOR = {
        "attention":  "#4C72B0",
        "expert_ffn": "#FF6EB4",
        "gate":       "#EF9F27",
        "lm_head":    CB["green"],
        "other":      "#999999",
    }
    _CAT_LABEL = {
        "attention":  "Attention (q/k/v/o)",
        "expert_ffn": "Expert FFN (w1/w2/w3)",
        "gate":       "Gate / Routing",
        "lm_head":    "LM Head",
        "other":      "Other",
    }
    # expert_ffn drawn before attention so attention sits on top
    cat_order = ["gate", "other", "lm_head", "expert_ffn", "attention"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True, which="both", ls="--", alpha=0.25, zorder=0)

    plotted_cats: set[str] = set()

    def _role_key(r: str) -> str:
        return r if r in _CAT_COLOR else "other"

    for cat in cat_order:
        idx = [i for i in range(len(valid)) if _role_key(roles[i]) == cat]
        if not idx:
            continue
        ax.scatter(
            intensities[idx], total_flops[idx],
            s=sizes[idx], c=_CAT_COLOR[cat], alpha=0.7,
            edgecolors="#BBBBBB", linewidths=0.5, zorder=2,
        )
        plotted_cats.add(cat)

    # ── Annotate top 3 by total FLOPs per weight type ───────────────────────
    annotated_c: set[int] = set()

    def _clabel(i: int) -> None:
        if i in annotated_c:
            return
        annotated_c.add(i)
        ax.annotate(
            _point_label(modules[i]),
            xy=(intensities[i], total_flops[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color=CB["black"], alpha=0.85,
        )

    by_cat: dict[str, list[int]] = {}
    for i, role in enumerate(roles):
        key = _role_key(role)
        by_cat.setdefault(key, []).append(i)

    for idx_list in by_cat.values():
        top3 = sorted(idx_list, key=lambda i: -total_flops[i])[:7]
        for i in top3:
            _clabel(i)

    # ── Quadrant text labels ─────────────────────────────────────────────────
    def _gmx(lo: float, hi: float) -> float:
        return 10 ** ((np.log10(lo) + np.log10(hi)) / 2)

    x_mid = 10 ** ((np.log10(x_min) + np.log10(x_max)) / 2)
    y_mid = 10 ** ((np.log10(y_min) + np.log10(y_max)) / 2)

    for (tx, ty, label, color) in [
        (_gmx(x_mid, x_max), _gmx(y_mid, y_max), "high intensity\nhigh compute",  "#666666"),
        (_gmx(x_min, x_mid), _gmx(y_mid, y_max), "low intensity\nhigh compute",   "#E69F00"),
        (_gmx(x_min, x_mid), _gmx(y_min, y_mid), "low intensity\nlow compute",    "#E69F00"),
    ]:
        ax.text(
            tx, ty, label,
            fontsize=8, color=color, alpha=0.85, ha="center", va="center",
            zorder=1,
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity per Inference (FLOPs / weight numels)", fontsize=11)
    ax.set_ylabel("FLOPs per Inference", fontsize=11)
    ax.set_title(
        f"AIMC Tile Mapping Candidates\n{model_name} on {hardware_name}"
        f"  (avg over {n_sequences} inference{'s' if n_sequences != 1 else ''})",
        fontsize=13,
    )

    # ── Color legend (upper-left) ─────────────────────────────────────────────
    color_patches = [
        mpatches.Patch(facecolor=_CAT_COLOR[cat], alpha=0.85,
                       label=_CAT_LABEL[cat])
        for cat in cat_order if cat in plotted_cats
    ]
    color_leg = ax.legend(
        handles=color_patches, fontsize=9, loc="upper left",
        title="Weight type", title_fontsize=9, framealpha=0.9,
    )
    ax.add_artist(color_leg)

    # ── Bubble-size legend (lower-right) ─────────────────────────────────────
    size_handles = []
    for pct_val in [1.0, 10.0, 100.0]:
        if log_pct.max() > log_pct.min():
            norm = (np.log1p(pct_val) - log_pct.min()) / (log_pct.max() - log_pct.min())
            sz = 20.0 + 480.0 * float(np.clip(norm, 0.0, 1.0))
        else:
            sz = 150.0
        size_handles.append(
            plt.scatter([], [], s=sz, c="#999999", alpha=0.65, label=f"{pct_val:.0f}%")
        )
    ax.legend(
        handles=size_handles, fontsize=8, loc="lower right",
        title="Call frequency", title_fontsize=8, framealpha=0.9,
    )

    _save(fig, "aimc_candidates", out, cfg)


# ─── Plot 8 — Tile Activation Timeline (Gantt) ───────────────────────────────

def plot_tile_timeline(data: dict, cfg: dict, out: Path) -> None:
    """
    Gantt chart of expert activation windows per layer.
    Duration proportional to total FLOPs (normalised within each layer).
    """
    per_layer = data.get("per_layer", {})
    if not per_layer:
        print("  [skip] tile_timeline: no data"); return

    max_show = cfg.get("max_gantt_layers", DEFAULT_PLOT_CONFIG["max_gantt_layers"])
    sorted_ids = sorted(per_layer.keys(), key=int)
    if max_show is not None:
        sorted_ids = sorted_ids[:max_show]
        note = f" (first {len(sorted_ids)} of {len(per_layer)} layers)"
    else:
        note = ""

    # Unique expert IDs across all shown layers
    all_eids = sorted({e["expert_id"] for lid in sorted_ids
                       for e in per_layer[lid]["experts"]})
    n_exp = len(all_eids)
    if n_exp == 0:
        print("  [skip] tile_timeline: no experts"); return

    n_layers  = len(sorted_ids)
    bar_h     = 0.70
    gap       = 0.50           # vertical gap between layer groups
    stride    = n_exp * (bar_h + 0.08) + gap
    layer_colors = _layer_cmap(n_layers)

    fig_h = min(20, max(6, n_layers * stride / 6 + 1))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ytick_pos, ytick_labels = [], []

    for li, lid in enumerate(sorted_ids):
        experts = per_layer[lid]["experts"]
        color   = layer_colors[li]
        base_y  = li * stride

        ytick_pos.append(base_y + (n_exp - 1) * (bar_h + 0.08) / 2)
        ytick_labels.append(f"Layer {lid}")

        for ei_idx, e in enumerate(sorted(experts, key=lambda x: x["expert_id"])):
            y_center = base_y + ei_idx * (bar_h + 0.08)
            ax.barh(y_center, e["relative_end"] - e["relative_start"],
                    left=e["relative_start"], height=bar_h,
                    color=color, alpha=0.85, edgecolor="white", lw=0.4)
            if n_exp <= 12:
                ax.text(-0.02, y_center, f"E{e['expert_id']}",
                        ha="right", va="center", fontsize=6.5, color=CB["black"])

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("Relative Compute Time (normalised per layer)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"Tile Activation Timeline{note}", fontsize=13)
    ax.axvline(1.0, color=CB["gray"], lw=0.8, ls="--", alpha=0.5)
    ax.invert_yaxis()

    # Compact layer-colour legend (max 10 entries)
    legend_step = max(1, n_layers // 10)
    handles = [
        mpatches.Patch(color=layer_colors[i], label=f"Layer {sorted_ids[i]}")
        for i in range(0, n_layers, legend_step)
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              ncol=min(5, len(handles)), title="Layer colour")

    _save(fig, "tile_timeline", out, cfg)


# ─── Plot 9 — Precision Sensitivity ──────────────────────────────────────────

_BITS = {"fp16": 16, "int8": 8, "int4": 4}


def plot_precision_sensitivity(data: dict, cfg: dict, out: Path) -> None:
    """
    Two-panel plot: routing agreement and output MSE vs quantisation bits,
    broken down by weight type (selective int8) and full-model.
    """
    results      = data.get("results", {})
    by_wt        = data.get("by_weight_type", {})

    if not results:
        print("  [skip] precision_sensitivity: no results"); return

    fig, (ax_ra, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── Left: routing agreement ──
    def _plot_curve(ax: plt.Axes, prec_dict: dict, label: str,
                    color: str, ls: str, key: str) -> None:
        pts = sorted(
            ((bits, v[key]) for p, v in prec_dict.items()
             if "error" not in v and v.get(key) is not None
             for bits in [_BITS.get(p, None)] if bits is not None),
            key=lambda x: x[0]
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, ls=ls, lw=1.8, marker="o", ms=5, label=label)

    # Full-model curves
    _plot_curve(ax_ra, results, "Full model", CB["blue"], "-", "routing_agreement")
    _plot_curve(ax_mse, results, "Full model", CB["blue"], "-", "output_mse")

    # fp16 baseline reference
    ax_ra.axhline(1.0, color=CB["black"], ls="--", lw=1.0, alpha=0.6, label="fp16 baseline")
    ax_mse.axhline(0.0, color=CB["black"], ls="--", lw=1.0, alpha=0.6, label="fp16 baseline")

    # Per-weight-type selective int8 curves
    wtype_colors = {"gate_only": CB["orange"], "expert_only": CB["purple"],
                    "attention_only": CB["sky"]}
    wtype_labels = {"gate_only": "Gate only (int8)",
                    "expert_only": "Expert only (int8)",
                    "attention_only": "Attention only (int8)"}
    for wt, wdata in by_wt.items():
        if "error" in wdata:
            continue
        bits = 8
        color = wtype_colors.get(wt, CB["gray"])
        ra = wdata.get("routing_agreement")
        mse = wdata.get("output_mse")
        if ra is not None:
            ax_ra.scatter([bits], [ra], color=color, s=60, zorder=5,
                          marker="^", label=wtype_labels.get(wt, wt))
        if mse is not None:
            ax_mse.scatter([bits], [mse], color=color, s=60, zorder=5,
                           marker="^", label=wtype_labels.get(wt, wt))

    for ax in (ax_ra, ax_mse):
        ax.set_xlabel("Quantisation Bits", fontsize=11)
        ax.set_xticks([4, 8, 16])
        ax.set_xticklabels(["int4\n(4b)", "int8\n(8b)", "fp16\n(16b)"])
        ax.invert_xaxis()
        ax.legend(fontsize=9)

    ax_ra.set_ylabel("Routing Agreement with fp16 (%)", fontsize=11)
    ax_ra.set_title("Expert Routing Agreement", fontsize=13)
    ax_ra.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_ra.set_ylim(0, 1.05)

    ax_mse.set_ylabel("Output MSE vs fp16", fontsize=11)
    ax_mse.set_title("Output Divergence (MSE)", fontsize=13)
    if ax_mse.get_ylim()[0] < 0:
        ax_mse.set_ylim(bottom=0)

    fig.suptitle("Precision Sensitivity Analysis", fontsize=14, y=1.01)
    _save(fig, "precision_sensitivity", out, cfg)


# ─── Plot 10 — Summary Dashboard ─────────────────────────────────────────────

def plot_summary_dashboard(all_data: dict[str, dict], cfg: dict, out: Path) -> None:
    """
    2×2 overview figure:
      top-left:     Roofline (simplified)
      top-right:    Load imbalance ratio per layer
      bottom-left:  Digital vs analog pie (overall)
      bottom-right: Batch size CDF
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)

    # ── Top-left: simplified roofline ──────────────────────────────────────
    ax_rf = fig.add_subplot(gs[0, 0])
    rf = all_data.get("roofline")
    if rf and rf.get("ops"):
        peak_compute = rf["config"]["peak_compute_tops"]
        peak_bw      = rf["config"]["peak_bandwidth_tbs"]
        ridge        = rf["ridge_point"]
        ops          = rf["ops"]

        xs_all = [o["intensity"] for o in ops if o["intensity"] > 0]
        if xs_all:
            x_min = 10 ** (np.floor(np.log10(min(xs_all))) - 0.3)
            x_max = 10 ** (np.ceil(np.log10(max(xs_all)))  + 0.5)
            x_ln  = np.logspace(np.log10(x_min), np.log10(x_max), 200)
            ax_rf.loglog(x_ln, np.minimum(peak_compute, peak_bw * x_ln),
                         color=CB["black"], lw=1.5)

            seen: dict[str, tuple[list, list]] = {}
            for op in ops:
                if op["intensity"] <= 0:
                    continue
                key = op["role_category"]
                perf = min(peak_compute, peak_bw * op["intensity"])
                seen.setdefault(key, ([], []))
                seen[key][0].append(op["intensity"])
                seen[key][1].append(perf)
            for key, (xi, yi) in seen.items():
                ax_rf.scatter(xi, yi, c=OP_COLOR.get(key, CB["gray"]),
                              s=3, alpha=0.2, rasterized=True,
                              label=key.replace("_", " "))

        ax_rf.axvline(ridge, color=CB["gray"], ls=":", lw=0.8, alpha=0.6)
        ax_rf.set_xlabel("Intensity (FLOP/numel)", fontsize=9)
        ax_rf.set_ylabel("Performance (TOPS)", fontsize=9)
        ax_rf.set_title("Roofline", fontsize=11)
        ax_rf.tick_params(labelsize=8)
        ax_rf.grid(True, which="both", ls="--", alpha=0.25)
    else:
        ax_rf.text(0.5, 0.5, "No roofline data", ha="center", va="center",
                   transform=ax_rf.transAxes)

    # ── Top-right: load imbalance ratio per layer ────────────────────────
    ax_li = fig.add_subplot(gs[0, 1])
    li = all_data.get("load_imbalance")
    if li and li.get("per_layer"):
        sorted_ids = sorted(li["per_layer"].keys(), key=int)
        ratios = [li["per_layer"][lid]["load_imbalance_ratio"] for lid in sorted_ids]
        ax_li.plot(range(len(sorted_ids)), ratios, color=CB["vermillion"],
                   lw=1.6, marker="o", ms=3)
        ax_li.axhline(1.0, color=CB["black"], ls="--", lw=0.9, alpha=0.5,
                      label="Perfect balance")
        ax_li.fill_between(range(len(sorted_ids)), ratios, 1.0,
                           alpha=0.15, color=CB["vermillion"])
        ax_li.set_xlabel("Layer Index", fontsize=9)
        ax_li.set_ylabel("max / mean tokens", fontsize=9)
        ax_li.set_title("Load Imbalance Ratio", fontsize=11)
        ax_li.tick_params(labelsize=8)
        ax_li.legend(fontsize=8)
    else:
        ax_li.text(0.5, 0.5, "No imbalance data", ha="center", va="center",
                   transform=ax_li.transAxes)

    # ── Bottom-left: analog vs digital pie (overall) ─────────────────────
    ax_pie = fig.add_subplot(gs[1, 0])
    da = all_data.get("digital_analog_split")
    if da and da.get("overall"):
        ov = da["overall"]
        total = ov["total_flops"]
        slices = [ov["analog_flops"], ov["digital_flops"]]
        labels_pie = [
            f"Analog\n{ov['analog_ratio']*100:.1f}%",
            f"Digital\n{(1-ov['analog_ratio'])*100:.1f}%",
        ]
        colors_pie = [CB["blue"], CB["gray"]]
        explode = (0.03, 0)
        wedges, texts = ax_pie.pie(
            slices, labels=labels_pie, colors=colors_pie,
            explode=explode, startangle=90,
            textprops={"fontsize": 9},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax_pie.set_title("FLOP Split (Overall)", fontsize=11)
        total_str = f"{total/1e12:.2f} TFLOP" if total >= 1e12 else f"{total/1e9:.1f} GFLOP"
        ax_pie.text(0, -1.25, f"Total: {total_str}", ha="center", fontsize=8)
    else:
        ax_pie.text(0.5, 0.5, "No split data", ha="center", va="center",
                    transform=ax_pie.transAxes)

    # ── Bottom-right: batch size CDF ─────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 1])
    bs = all_data.get("batch_size_distribution")
    if bs and bs.get("cdf"):
        cdf_pts = bs["cdf"]
        pcts = bs.get("percentiles", {})
        min_viable = cfg.get("min_viable_batch", DEFAULT_PLOT_CONFIG["min_viable_batch"])
        xs = [pt[0] for pt in cdf_pts]
        ys = [pt[1] for pt in cdf_pts]
        ax_cdf.plot(xs, ys, color=CB["blue"], lw=1.8)
        ax_cdf.axvspan(min(xs) - 0.5, min_viable,
                       alpha=0.12, color=CB["red"],
                       label=f"< viable ({min_viable}t)")
        for pname, color in [("p50", CB["vermillion"]), ("p90", CB["orange"])]:
            v = pcts.get(pname)
            if v is not None:
                ax_cdf.axvline(v, color=color, ls="--", lw=1.2, label=f"{pname.upper()}={v:.0f}")
        ax_cdf.set_xlabel("Tokens per Expert", fontsize=9)
        ax_cdf.set_ylabel("Cumulative Probability", fontsize=9)
        ax_cdf.set_title("Batch Size CDF", fontsize=11)
        ax_cdf.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_cdf.set_ylim(0, 1.02)
        ax_cdf.tick_params(labelsize=8)
        ax_cdf.legend(fontsize=8)
    else:
        ax_cdf.text(0.5, 0.5, "No CDF data", ha="center", va="center",
                    transform=ax_cdf.transAxes)

    fig.suptitle("MoE AIMC Workload — Summary Dashboard", fontsize=14, y=1.01)
    _save(fig, "summary_dashboard", out, cfg)


# ─── Orchestration ────────────────────────────────────────────────────────────

_PLOT_REGISTRY = [
    ("roofline",                   "aimc_candidates",          plot_aimc_candidates),
]


def plot_all(
    raw_dir: Path,
    out_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Load all JSON files from raw_dir and generate all plots in out_dir.

    Args:
        raw_dir:  Directory containing the *.json files from run_all_metrics().
        out_dir:  Destination directory for PDF and PNG outputs.
        config:   Override any DEFAULT_PLOT_CONFIG key, including 'style'.
    """
    cfg = {**DEFAULT_PLOT_CONFIG, **(config or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style(cfg)

    # Load all raw data upfront
    all_data: dict[str, dict | None] = {
        name: _load(raw_dir, name)
        for name, _, _ in _PLOT_REGISTRY
    }

    total = len(_PLOT_REGISTRY)
    for i, (json_name, plot_name, fn) in enumerate(_PLOT_REGISTRY, 1):
        data = all_data[json_name]
        if data is None:
            continue
        print(f"  [{i}/{total}] {plot_name}...")
        try:
            fn(data, cfg, out_dir)
        except Exception as exc:
            print(f"    ERROR in {plot_name}: {exc}")
            plt.close("all")


    print(f"  Done. Plots saved to {out_dir}/")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate thesis-quality plots from workload metrics JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("raw_dir", help="Directory containing the metrics/raw/*.json files.")
    p.add_argument("--output-dir", default="metrics/plots",
                   help="Destination directory for PDF/PNG outputs.")
    p.add_argument("--style", default="thesis_print",
                   choices=list(STYLES), help="Visual style preset.")
    p.add_argument("--dpi", type=int, default=300, help="Export DPI.")
    p.add_argument("--min-viable-batch", type=int, default=4,
                   metavar="N",
                   help="Token threshold for AIMC viable batch shading in CDF plot.")
    p.add_argument("--max-gantt-layers", type=int, default=16,
                   metavar="N",
                   help="Max layers shown in Gantt chart (0 = all).")
    p.add_argument("--max-bar-layers", type=int, default=16,
                   metavar="N",
                   help="Max layers in load-imbalance bar chart.")
    args = p.parse_args()

    plot_all(
        raw_dir=Path(args.raw_dir).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        config={
            "style":            args.style,
            "dpi":              args.dpi,
            "min_viable_batch": args.min_viable_batch,
            "max_gantt_layers": args.max_gantt_layers or None,
            "max_bar_layers":   args.max_bar_layers,
        },
    )


if __name__ == "__main__":
    _cli()
