import json
import sys
import argparse

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trace",  required=True, help="Path to inference_trace.json")
parser.add_argument("--layer",  type=int, help="Layer number e.g. 0")
parser.add_argument(
    "--kind",
    default="experts",
    choices=["experts", "mlp", "layer", "lm_head", "attention", "router"],
    help="Layer shorthand to inspect when --target is not provided",
)
parser.add_argument(
    "--target",
    help="Custom substring to match in user_annotation names, e.g. model.language_model.layers.0.mlp",
)
parser.add_argument(
    "--nlayer",
    type=int,
    help="Deprecated alias for --occurrence",
)
parser.add_argument(
    "--occurrence",
    default=1,
    type=int,
    help="Nth matching annotation to inspect, 1-based; later occurrences correspond to later forwards/decode steps",
)
parser.add_argument(
    "--last",
    action="store_true",
    help="Inspect the last matching annotation",
)
parser.add_argument(
    "--list-matches",
    action="store_true",
    help="List matching annotations and exit",
)
parser.add_argument("--sort",   default="ts", choices=["ts", "dur"], help="Sort by timestamp or duration")
parser.add_argument("--top",    default=0, type=int, help="Show only top N kernels by duration (0 = all)")
args = parser.parse_args()

if args.nlayer is not None:
    if args.occurrence != 1:
        parser.error("--nlayer and --occurrence cannot both be set")
    args.occurrence = args.nlayer

if args.occurrence < 1:
    parser.error("--occurrence must be >= 1")

GLOBAL_KINDS = {"lm_head"}  # kinds that don't need --layer

if args.target:
    TARGET = args.target
elif args.kind == "lm_head":
    TARGET = "lm_head ("
elif args.layer is None:
    parser.error("--layer is required unless --target is provided or --kind is one of: " +
                 ", ".join(sorted(GLOBAL_KINDS)))
elif args.kind == "experts":
    TARGET = f"layer{args.layer}.experts"
elif args.kind == "mlp":
    TARGET = f"layers.{args.layer}.mlp ("
elif args.kind == "attention":
    TARGET = f"layer{args.layer}.attention ("
elif args.kind == "router":
    TARGET = f"layer{args.layer}.router ("
else:
    TARGET = f"layer{args.layer} ("

AVAILABLE_FILTERS = {
    "experts":   lambda name: "expert" in name.lower(),
    "mlp":       lambda name: ".mlp" in name.lower(),
    "layer":     lambda name: name.startswith("layer"),
    "lm_head":   lambda name: name.startswith("lm_head"),
    "attention": lambda name: ".attention" in name.lower(),
    "router":    lambda name: ".router" in name.lower(),
}
available_filter = AVAILABLE_FILTERS.get(args.kind, lambda name: True)

# ── Load trace ────────────────────────────────────────────────────────────────
print(f"\nLoading {args.trace}...")
with open(args.trace) as f:
    data = json.load(f)
events = data.get("traceEvents", data)

# ── Find cpu user_annotation ──────────────────────────────────────────────────
cpu_matches = [
    e for e in events
    if (e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
            and TARGET in e.get("name", ""))
]
if args.list_matches:
    if not cpu_matches:
        print(f"No annotations matched '{TARGET}'")
    else:
        print(f"Found {len(cpu_matches)} matching '{TARGET}' annotations:")
        for idx, e in enumerate(cpu_matches, start=1):
            print(f"  #{idx:<3} ts={e.get('ts', 0):.3f} dur={e.get('dur', 0):.3f}μs  {e['name']}")
    sys.exit(0)

match_index = len(cpu_matches) if args.last and cpu_matches else args.occurrence
cpu_event = cpu_matches[match_index - 1] if match_index <= len(cpu_matches) else None

if cpu_event is None:
    print(f"ERROR: could not find user_annotation #{match_index} matching '{TARGET}'")
    if cpu_matches:
        print(f"Found {len(cpu_matches)} matching '{TARGET}' annotations:")
        for idx, e in enumerate(cpu_matches, start=1):
            print(f"  #{idx:<3} ts={e.get('ts', 0):.3f} dur={e.get('dur', 0):.3f}μs  {e['name']}")
        sys.exit(1)
    print(f"Available {args.kind} annotations:")
    for idx, e in enumerate(
        (
            event for event in events
            if event.get("cat") == "user_annotation"
            and available_filter(event.get("name", ""))
        ),
        start=1,
    ):
        print(f"  #{idx:<3} ts={e.get('ts', 0):.3f} dur={e.get('dur', 0):.3f}μs  {e['name']}")
    sys.exit(1)

cpu_ext_id = cpu_event["args"]["External id"]
cpu_dur    = cpu_event["dur"]
cpu_start  = cpu_event["ts"]
cpu_end    = cpu_start + cpu_dur
cpu_name   = cpu_event["name"]
parent_cpu_ext_id = cpu_ext_id
parent_cpu_name = cpu_name

def module_path(name):
    return name.split(" (", 1)[0]

def find_gpu_window(external_id):
    for event in events:
        if (
            event.get("cat") == "gpu_user_annotation"
            and event.get("ph") == "X"
            and event.get("args", {}).get("External id") == external_id
        ):
            return event
    return None

def contains_event(parent_start, parent_end, event):
    ts = event.get("ts", 0)
    dur = event.get("dur", 0)
    return ts >= parent_start - 5 and ts + dur <= parent_end + 5

def intervals_from(events_to_use):
    return [
        (event["ts"], event["ts"] + event["dur"])
        for event in events_to_use
    ]

def event_inside_any_interval(event, intervals):
    ts = event.get("ts", 0)
    dur = event.get("dur", 0)
    return any(ts >= start and ts + dur <= end + 5 for start, end in intervals)

cpu_windows = [cpu_event]
gpu_windows = []
aggregation_note = None

selected_module_path = module_path(cpu_name)
if selected_module_path.endswith(".mlp"):
    child_cpu_windows = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and event.get("ph") == "X"
        and module_path(event.get("name", "")).startswith(selected_module_path + ".")
        and contains_event(cpu_start, cpu_end, event)
    ]
    child_cpu_windows.sort(key=lambda event: event.get("ts", 0))
    child_gpu_windows = [
        gpu_window
        for gpu_window in (
            find_gpu_window(event.get("args", {}).get("External id"))
            for event in child_cpu_windows
        )
        if gpu_window is not None
    ]
    parent_gpu_window = find_gpu_window(parent_cpu_ext_id)
    if child_cpu_windows and (child_gpu_windows or parent_gpu_window is not None):
        gpu_windows = [
            gpu_window for gpu_window in [parent_gpu_window, *child_gpu_windows]
            if gpu_window is not None
        ]
        cpu_ext_id = ",".join(
            str(event.get("args", {}).get("External id"))
            for event in [cpu_event, *child_cpu_windows]
        )
        cpu_name = f"{cpu_name} [unfolded children]"
        aggregation_note = (
            f"unfolded {len(child_cpu_windows)} child annotations inside parent MLP"
        )

# ── Find matching gpu_user_annotation ─────────────────────────────────────────
if not gpu_windows:
    gpu_window = find_gpu_window(cpu_event["args"]["External id"])
    if gpu_window is not None:
        gpu_windows = [gpu_window]

if not gpu_windows:
    print(f"ERROR: no gpu_user_annotation found for External id {cpu_ext_id}")
    print("CUDA activity may not have been captured. Check ProfilerActivity.CUDA is enabled.")
    sys.exit(1)

gpu_start = min(event["ts"] for event in gpu_windows)
gpu_end   = max(event["ts"] + event["dur"] for event in gpu_windows)
gpu_dur   = gpu_end - gpu_start
gpu_intervals = intervals_from(gpu_windows)
cpu_intervals = intervals_from(cpu_windows)

# ── Collect GPU kernel events ─────────────────────────────────────────────────
gpu_events = []
for e in events:
    if e.get("cat") not in ("kernel", "gpu_memcpy"):
        continue
    if e.get("ph") != "X":
        continue
    ts  = e.get("ts", 0)
    dur = e.get("dur", 0)
    if event_inside_any_interval(e, gpu_intervals):
        gpu_events.append({
            "side": "GPU",
            "cat":  e["cat"],
            "name": e["name"],
            "ts":   ts,
            "dur":  dur,
            "rel":  ts - gpu_start,
        })

# ── Collect CPU op events inside the cpu annotation window ───────────────────
cpu_ops = []
for e in events:
    if e.get("cat") not in ("cpu_op", "user_annotation"):
        continue
    if e.get("ph") != "X":
        continue
    if (
        e.get("name") == parent_cpu_name
        and e.get("args", {}).get("External id") == parent_cpu_ext_id
    ):
        continue  # skip the parent annotation itself
    ts  = e.get("ts", 0)
    dur = e.get("dur", 0)
    if event_inside_any_interval(e, cpu_intervals):
        cpu_ops.append({
            "side": "CPU",
            "cat":  e["cat"],
            "name": e["name"],
            "ts":   ts,
            "dur":  dur,
            "rel":  ts - cpu_start,
        })

# ── Sort ──────────────────────────────────────────────────────────────────────
sort_key = (lambda x: -x["dur"]) if args.sort == "dur" else (lambda x: x["ts"])
gpu_events.sort(key=sort_key)
cpu_ops.sort(key=sort_key)

if args.top > 0:
    gpu_events = gpu_events[:args.top]
    cpu_ops    = cpu_ops[:args.top]

# ── Compute GPU metrics ───────────────────────────────────────────────────────
total_kernel_time = sum(k["dur"] for k in gpu_events)
compute_ratio     = total_kernel_time / gpu_dur if gpu_dur > 0 else 0
gpu_overhead      = gpu_dur - total_kernel_time
gemm_time         = sum(
    k["dur"] for k in gpu_events
    if any(x in k["name"].lower() for x in ("cutlass", "gemm", "sgemm", "gemv"))
)
dispatch_time     = sum(
    k["dur"] for k in gpu_events
    if any(x in k["name"].lower() for x in ("gather", "memcpy", "scatter", "fill", "scan", "cub"))
)

# ── Compute CPU metrics ───────────────────────────────────────────────────────
total_cpu_op_time = sum(e["dur"] for e in cpu_ops)
cpu_gap           = cpu_dur - total_cpu_op_time
sync_time         = sum(
    e["dur"] for e in cpu_ops
    if any(x in e["name"].lower() for x in ("dtoh", "copy_", "clone", "empty"))
)
dispatch_cpu_time = sum(
    e["dur"] for e in cpu_ops
    if any(x in e["name"].lower() for x in ("index", "gather", "scatter", "reshape",
                                              "clamp", "repeat", "chunk", "split",
                                              "flatten", "view", "expand"))
)

# ── Identify GPU idle gaps ────────────────────────────────────────────────────
sorted_gpu = sorted(gpu_events, key=lambda x: x["ts"])
gaps = []
for i in range(1, len(sorted_gpu)):
    prev_end = sorted_gpu[i-1]["ts"] + sorted_gpu[i-1]["dur"]
    curr_start = sorted_gpu[i]["ts"]
    gap_dur = curr_start - prev_end
    if gap_dur > 2:  # only gaps > 2μs
        # find CPU ops that overlap this gap
        gap_cpu = [
            e for e in cpu_ops
            if e["ts"] < curr_start and e["ts"] + e["dur"] > prev_end
        ]
        cause = "unknown"
        if any("dtoh" in e["name"].lower() or "dtoH" in e["name"] for e in gap_cpu):
            cause = "sync_barrier"
        elif any("launch" in e["name"].lower() for e in gap_cpu):
            cause = "launch_latency"
        elif gap_cpu:
            cause = "cpu_overhead"
        gaps.append({
            "rel_start": prev_end - gpu_start,
            "dur":       gap_dur,
            "cause":     cause,
            "cpu_ops":   [e["name"] for e in gap_cpu[:3]],
        })

total_gap_time  = sum(g["dur"] for g in gaps)
sync_gap_time   = sum(g["dur"] for g in gaps if g["cause"] == "sync_barrier")
launch_gap_time = sum(g["dur"] for g in gaps if g["cause"] == "launch_latency")
cpu_gap_time    = sum(g["dur"] for g in gaps if g["cause"] == "cpu_overhead")
unknown_gap     = sum(g["dur"] for g in gaps if g["cause"] == "unknown")

# ── Helper to shorten kernel names ───────────────────────────────────────────
def short_name(name, width=65):
    if "cutlass" in name.lower():
        parts = name.split("<")
        s = parts[1].split(">")[0] if len(parts) > 1 else name
        return f"cutlass::{s[:width]}"
    if "gemv" in name.lower() and "cutlass" not in name.lower() and "gemm" not in name.lower():
        return f"gemv::kernel (GEMV){'':<11}"
    if "::" in name:
        parts = name.split("::")
        return "::".join(parts[-2:])[:width]
    return name[:width]

# ══════════════════════════════════════════════════════════════════════════════
# PRINT
# ══════════════════════════════════════════════════════════════════════════════

W = 90

def fmt_pct(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"

print(f"\n{'='*W}")
print(f"  Block:  {cpu_name}")
print(f"  Match:  #{match_index} of {len(cpu_matches)} matching '{TARGET}' annotations")
print(f"  Ext id: {cpu_ext_id}")
if aggregation_note:
    print(f"  Mode:   {aggregation_note}")
print(f"{'='*W}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'SUMMARY':─<{W}}")
print(f"  {'Metric':<40} {'CPU':>12}  {'GPU':>12}")
print(f"  {'─'*40} {'─'*12}  {'─'*12}")
print(f"  {'Window duration':<40} {cpu_dur:>10.1f}μs  {gpu_dur:>10.1f}μs")
print(f"  {'Active op time':<40} {total_cpu_op_time:>10.1f}μs  {total_kernel_time:>10.1f}μs")
print(f"  {'Idle/gap time':<40} {cpu_gap:>10.1f}μs  {gpu_overhead:>10.1f}μs")
print(f"  {'Active ratio':<40} {total_cpu_op_time/cpu_dur:>10.1%}  {compute_ratio:>10.1%}")
print(f"  {'Num operations':<40} {len(cpu_ops):>12}  {len(gpu_events):>12}")

# ── GPU breakdown ─────────────────────────────────────────────────────────────
print(f"\n{'GPU BREAKDOWN':─<{W}}")
print(f"  {'AIMC analog (gemm/cutlass/gemv)':<50} {gemm_time:>8.1f}μs  {gemm_time/gpu_dur:>6.1%}")
print(f"  {'Digital dispatch (gather/scatter/fill/scan)':<50} {dispatch_time:>8.1f}μs  {dispatch_time/gpu_dur:>6.1%}")
other_gpu = total_kernel_time - gemm_time - dispatch_time
print(f"  {'Other kernels (elementwise/reduce/etc)':<50} {other_gpu:>8.1f}μs  {other_gpu/gpu_dur:>6.1%}")
print(f"  {'GPU idle gaps':<50} {gpu_overhead:>8.1f}μs  {gpu_overhead/gpu_dur:>6.1%}")

# ── CPU breakdown ─────────────────────────────────────────────────────────────
print(f"\n{'CPU BREAKDOWN':─<{W}}")
print(f"  {'Dispatch/index ops':<50} {dispatch_cpu_time:>8.1f}μs  {dispatch_cpu_time/cpu_dur:>6.1%}")
print(f"  {'Sync/copy ops':<50} {sync_time:>8.1f}μs  {sync_time/cpu_dur:>6.1%}")
other_cpu = total_cpu_op_time - dispatch_cpu_time - sync_time
print(f"  {'Other cpu ops':<50} {other_cpu:>8.1f}μs  {other_cpu/cpu_dur:>6.1%}")
print(f"  {'CPU idle/python overhead':<50} {cpu_gap:>8.1f}μs  {cpu_gap/cpu_dur:>6.1%}")

# ── Gap analysis ──────────────────────────────────────────────────────────────
print(f"\n{'GPU IDLE GAP ANALYSIS':─<{W}}")
print(f"  Total gap time:    {total_gap_time:>8.1f}μs")
print(f"  Sync barriers:     {sync_gap_time:>8.1f}μs  ({fmt_pct(sync_gap_time, total_gap_time)} of gaps)  ← CPU-GPU sync points")
print(f"  Launch latency:    {launch_gap_time:>8.1f}μs  ({fmt_pct(launch_gap_time, total_gap_time)} of gaps)  ← CUDA kernel launch overhead")
print(f"  CPU overhead:      {cpu_gap_time:>8.1f}μs  ({fmt_pct(cpu_gap_time, total_gap_time)} of gaps)  ← Python/dispatch logic")
print(f"  Unattributed:      {unknown_gap:>8.1f}μs  ({fmt_pct(unknown_gap, total_gap_time)} of gaps)")

if gaps:
    print(f"\n  Top 5 largest gaps:")
    print(f"  {'rel_start':>12}  {'dur':>10}  {'cause':<16}  cpu ops during gap")
    print(f"  {'─'*12}  {'─'*10}  {'─'*16}  {'─'*30}")
    for g in sorted(gaps, key=lambda x: -x["dur"])[:5]:
        cpu_during = ", ".join(g["cpu_ops"]) if g["cpu_ops"] else "none found"
        print(f"  {g['rel_start']:>12.1f}  {g['dur']:>10.1f}  {g['cause']:<16}  {cpu_during[:40]}")

# ── GPU kernel timeline ───────────────────────────────────────────────────────
print(f"\n{'GPU KERNELS':─<{W}}")
print(f"  {'rel_ts(μs)':>12}  {'dur(μs)':>10}  {'type':<12}  kernel")
print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*50}")
for k in gpu_events:
    print(f"  {k['rel']:>12.1f}  {k['dur']:>10.3f}  {k['cat']:<12}  {k['name']}")

# ── CPU op timeline ───────────────────────────────────────────────────────────
print(f"\n{'CPU OPS':─<{W}}")
print(f"  {'rel_ts(μs)':>12}  {'dur(μs)':>10}  {'type':<16}  op")
print(f"  {'─'*12}  {'─'*10}  {'─'*16}  {'─'*50}")
for e in cpu_ops:
    print(f"  {e['rel']:>12.1f}  {e['dur']:>10.3f}  {e['cat']:<16}  {short_name(e['name'])}")

# ── AIMC suitability ─────────────────────────────────────────────────────────
print(f"\n{'AIMC SUITABILITY':─<{W}}")
analog_pct  = gemm_time / gpu_dur
digital_pct = 1 - analog_pct
print(f"  Analog replaceable (matmuls):     {gemm_time:>8.1f}μs  ({analog_pct:.1%} of GPU window)")
print(f"  Remaining digital overhead:       {gpu_dur-gemm_time:>8.1f}μs  ({digital_pct:.1%} of GPU window)")
print(f"  Theoretical max AIMC speedup:     {gpu_dur/(gpu_dur-gemm_time):>8.2f}x  (if matmuls were instant)")
print(f"  Sync barrier cost:                {sync_gap_time:>8.1f}μs  ← eliminated on AIMC")
print(f"  Effective AIMC speedup (no sync): {gpu_dur/(gpu_dur-gemm_time-sync_gap_time):>8.2f}x")
print(f"{'='*W}\n")
