"""Heavy 15-minute PyTorch GPU stress test (power + thermal + memory + numerical).

Sustained large bf16 matmuls (max tensor-core power draw) + a multi-GB resident
buffer hammered each iteration (memory-bandwidth stress). Every 30 s it logs
temp / power / util / VRAM / SM-clock via nvidia-smi and checks the results for
non-finite values (NaN/Inf = a sign of unstable compute/memory). Prints a final
STABLE_OK / UNSTABLE verdict with peak temperature and power.
"""
import datetime
import subprocess
import sys
import time

import torch

assert torch.cuda.is_available(), "CUDA not available"
dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0  # seconds
N = 8192
BUF_GB = 6
dtype = torch.bfloat16
flops_per_mm = 2.0 * N ** 3


def smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm",
             "--format=csv,noheader,nounits"], text=True).strip()
    except Exception as e:
        return f"smi_err:{e}"


print(f"GPU: {name}  cap {cap}  torch {torch.__version__}", flush=True)

# Resident memory-stress buffer (~BUF_GB), plus working matmul matrices.
buf = torch.randn(int(BUF_GB * 1024 ** 3 // 2), device=dev, dtype=dtype)
a = torch.randn(N, N, device=dev, dtype=dtype)
b = torch.randn(N, N, device=dev, dtype=dtype)
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"VRAM after alloc: used={(total - free) / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)

t0 = time.time()
last = t0
it = 0
errs = 0
peak_t = 0.0
peak_p = 0.0
print(f"=== STRESS START {datetime.datetime.now():%H:%M:%S}  dur={DUR:.0f}s  "
      f"matmul {N}x{N} bf16 + {BUF_GB}GB mem ===", flush=True)
print("elapsed  iters  TFLOP/s | temp,power,util,memMiB,smclk", flush=True)

while time.time() - t0 < DUR:
    for _ in range(40):
        c = torch.mm(a, b)
        a = c * 1e-4 + b          # keep values bounded (avoid expected bf16 overflow)
        it += 1
    buf.mul_(1.0000001).add_(1e-6)  # memory-bandwidth stress
    torch.cuda.synchronize()
    if not (torch.isfinite(a).all() and torch.isfinite(buf[:4096]).all()):
        errs += 1
        print(f"[ERROR] non-finite values at iter {it} t={time.time() - t0:.0f}s", flush=True)
        a = torch.randn(N, N, device=dev, dtype=dtype)
        buf.normal_()
    now = time.time()
    if now - last >= 30:
        el = now - t0
        tf = it * flops_per_mm / el / 1e12
        s = smi()
        print(f"{el:7.0f} {it:7d} {tf:8.1f} | {s}", flush=True)
        f = [x.strip() for x in s.split(",")]
        if len(f) >= 2:
            try:
                peak_t = max(peak_t, float(f[0]))
                peak_p = max(peak_p, float(f[1]))
            except ValueError:
                pass
        last = now

torch.cuda.synchronize()
el = time.time() - t0
tf = it * flops_per_mm / el / 1e12
print(f"=== STRESS DONE el={el:.0f}s iters={it} avg={tf:.1f} TFLOP/s "
      f"peak_temp={peak_t:.0f}C peak_power={peak_p:.0f}W errors={errs} ===", flush=True)
print("VERDICT:", "STABLE_OK" if errs == 0 else f"UNSTABLE_{errs}_ERRORS", flush=True)
