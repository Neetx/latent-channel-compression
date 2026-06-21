import torch

print("torch", torch.__version__)
print("avail", torch.cuda.is_available())
print("name", torch.cuda.get_device_name(0))
print("cap", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())

x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("bf16 matmul OK", y.dtype, tuple(y.shape), round(float(y.float().abs().mean()), 3))

free, total = torch.cuda.mem_get_info()
print("vram_free_GB", round(free / 1e9, 2), "vram_total_GB", round(total / 1e9, 2))
