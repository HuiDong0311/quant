# test_fp4_matmul.py
import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda"

# 1. 创建输入矩阵
M, K, N = 128, 128, 128
a_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
b_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)

# 2. 模拟FP4量化（E2M1 or E3M0, 这里用最常见E2M1）
def quantize_fp4(x):
    # clamp 到可表示范围，大约 ±1.875
    x = torch.clamp(x, -1.875, 1.875)
    scale = 1.875 / 7.0
    q = torch.round(x / scale)
    return q * scale  # 模拟反量化（即 dequant 后的值）

a_fp4 = quantize_fp4(a_fp16)
b_fp4 = quantize_fp4(b_fp16)

# 3. 矩阵乘（FP4 仿真）
out_sim = a_fp4 @ b_fp4

# 4. 对比FP16基准
ref = a_fp16 @ b_fp16

err = (out_sim - ref).abs().mean()
rel_err = err / ref.abs().mean()

print(f"Mean Abs Error: {err.item():.6f}")
print(f"Relative Error: {rel_err.item():.6f}")
print("FP4 模拟矩阵乘测试完成 ✅")
