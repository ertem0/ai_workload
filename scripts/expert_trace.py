import json
import sys
import argparse

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trace",  required=True, help="Path to inference_trace.json")
parser.add_argument("--layer",  required=True, type=int, help="Layer number e.g. 0")
parser.add_argument("--sort",   default="ts", choices=["ts", "dur"], help="Sort by timestamp or duration")
parser.add_argument("--top",    default=0, type=int, help="Show only top N kernels by duration (0 = all)")
args = parser.parse_args()

TARGET = f"layer{args.layer}.experts"

# ── Load trace ────────────────────────────────────────────────────────────────
print(f"\nLoading {args.trace}...")
with open(args.trace) as f:
    data = json.load(f)
events = data.get("traceEvents", data)

# ── Find gpu_user_annotation for this block via External id ───────────────────
cpu_ext_id = None
cpu_dur    = None
for e in events:
    if (e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
            and TARGET in e.get("name", "")):
        cpu_ext_id = e["args"]["External id"]
        cpu_dur    = e["dur"]
        cpu_name   = e["name"]
        break

if cpu_ext_id is None:
    print(f"ERROR: could not find user_annotation matching '{TARGET}'")
    print("Available expert blocks:")
    for e in events:
        if e.get("cat") == "user_annotation" and "expert" in e.get("name","").lower():
            print(f"  {e['name']}")
    sys.exit(1)

# ── Find matching gpu_user_annotation ─────────────────────────────────────────
gpu_window = None
for e in events:
    if (e.get("cat") == "gpu_user_annotation"
            and e.get("ph") == "X"
            and e.get("args", {}).get("External id") == cpu_ext_id):
        gpu_window = e
        break

if gpu_window is None:
    print(f"ERROR: no gpu_user_annotation found for External id {cpu_ext_id}")
    print("CUDA activity may not have been captured. Check ProfilerActivity.CUDA is enabled.")
    sys.exit(1)

gpu_start = gpu_window["ts"]
gpu_end   = gpu_start + gpu_window["dur"]
gpu_dur   = gpu_window["dur"]

# ── Find all kernel events inside the GPU window ──────────────────────────────
kernels = []
for e in events:
    if e.get("cat") not in ("kernel", "gpu_memcpy"):
        continue
    if e.get("ph") != "X":
        continue
    ts  = e.get("ts", 0)
    dur = e.get("dur", 0)
    if ts >= gpu_start and ts + dur <= gpu_end + 5:  # 5μs tolerance
        kernels.append({
            "cat":  e["cat"],
            "name": e["name"],
            "ts":   ts,
            "dur":  dur,
            "rel":  ts - gpu_start,  # relative timestamp from block start
        })

# ── Sort ──────────────────────────────────────────────────────────────────────
if args.sort == "dur":
    kernels.sort(key=lambda x: -x["dur"])
else:
    kernels.sort(key=lambda x: x["ts"])

if args.top > 0:
    kernels = kernels[:args.top]

# ── Summary ───────────────────────────────────────────────────────────────────
total_kernel_time = sum(k["dur"] for k in kernels)
compute_ratio     = total_kernel_time / gpu_dur if gpu_dur > 0 else 0
overhead          = gpu_dur - total_kernel_time

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"Block:          {cpu_name}")
print(f"External id:    {cpu_ext_id}")
print(f"{'─'*80}")
print(f"CPU duration:   {cpu_dur:>10.1f} μs")
print(f"GPU duration:   {gpu_dur:>10.1f} μs")
print(f"Kernel time:    {total_kernel_time:>10.1f} μs  ({compute_ratio:.1%} of GPU window)")
print(f"Gap/overhead:   {overhead:>10.1f} μs  ({1-compute_ratio:.1%} of GPU window)")
print(f"Num kernels:    {len(kernels)}")
print(f"{'='*80}")
print(f"\n{'rel_ts(μs)':>12}  {'dur(μs)':>10}  {'type':<12}  kernel")
print(f"{'─'*12}  {'─'*10}  {'─'*12}  {'─'*50}")

for k in kernels:
    # Shorten kernel name for readability
    name = k["name"]
    if "cutlass" in name.lower():
        # Extract the meaningful part
        parts = name.split("<")
        short = parts[1].split(">")[0] if len(parts) > 1 else name
        short = f"cutlass::{short[:60]}"
    elif "::" in name:
        # Take last meaningful namespace
        parts = name.split("::")
        short = "::".join(parts[-2:])[:70]
    else:
        short = name[:70]

    print(f"{k['rel']:>12.1f}  {k['dur']:>10.3f}  {k['cat']:<12}  {short}")

print(f"\n{'─'*80}")
print(f"AIMC analog candidates (cutlass gemm):  ", end="")
gemm_time = sum(k["dur"] for k in kernels if "cutlass" in k["name"].lower() or "gemm" in k["name"].lower() or "sgemm" in k["name"].lower())
print(f"{gemm_time:.1f}μs ({gemm_time/gpu_dur:.1%} of GPU window)")
print(f"Digital overhead (dispatch/gather/misc): {gpu_dur - gemm_time:.1f}μs ({(gpu_dur-gemm_time)/gpu_dur:.1%} of GPU window)")