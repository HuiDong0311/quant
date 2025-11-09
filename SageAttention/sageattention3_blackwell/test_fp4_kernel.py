"""
✅ scaled_fp4_quant ok
你改的 CHECK_DTYPE 宏是可用的；

fp4quant_cuda 的接口我们已经按源码真实签名搞对了；

当前环境下 FP4 pack + FP8 scale 写入是能正常跑起来的。
"""
import torch, time
import fp4quant_cuda

torch.manual_seed(0)
device = "cuda"

# 按源码默认 BLOCK_SIZE=128，选一个合理维度
B, N, H, D = 1, 64, 8, 128   # (batch, seq, heads, head_dim)

# 源码里支持 fp16 / bf16，这里用 bf16
x = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16)

# 按实现：FP4 pack 后每 byte 存两个值 → D/2
# output_sf 存 scale，通常是每 16 个通道一个 FP8 → D/16
out = torch.empty(B, N, H, D // 2, device=device, dtype=torch.uint8)
out_sf = torch.empty(B, N, H, D // 16, device=device, dtype=torch.float8_e4m3fn)

print("[DEBUG] x    :", x.shape, x.dtype)
print("[DEBUG] out  :", out.shape, out.dtype, out.is_contiguous())
print("[DEBUG] out_sf:", out_sf.shape, out_sf.dtype, out_sf.is_contiguous())

# tensor_layout = 0: (B, N, H, D) 格式，对应源码里的一种分支
start = time.time()
fp4quant_cuda.scaled_fp4_quant(x, out, out_sf, 0)
torch.cuda.synchronize()
t = (time.time() - start) * 1000

print(f"✅ scaled_fp4_quant ok, time = {t:.3f} ms")
print("out sample:", out.flatten()[:8].tolist())
print("sf sample :", out_sf.flatten()[:8].tolist())
