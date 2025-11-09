import torch
from sageattn3 import patch_blackwell_capability # 自己打包的
patch_blackwell_capability()

import fp4quant_cuda
import fp4attn_cuda

torch.manual_seed(0)
device = "cuda"

B, H, N, D = 1, 8, 128, 128
dtype = torch.bfloat16
blk = 128

# 原始 Q/K/V: (B, H, N, D)
q = torch.randn(B, H, N, D, device=device, dtype=dtype)
k = torch.randn(B, H, N, D, device=device, dtype=dtype)
v = torch.randn(B, H, N, D, device=device, dtype=dtype)

# ---------------- Q/K: (B, H, N, D) -> (B, H, N, D/2) + scale (B,H,N,D/16) ----------------
q_fp4 = torch.empty(B, H, N, D // 2, dtype=torch.uint8, device=device)
k_fp4 = torch.empty(B, H, N, D // 2, dtype=torch.uint8, device=device)

q_sf = torch.empty(B, H, N, D // 16, dtype=torch.float8_e4m3fn, device=device)
k_sf = torch.empty(B, H, N, D // 16, dtype=torch.float8_e4m3fn, device=device)

# 正确顺序: (input, output_fp4, output_sf, blk)
fp4quant_cuda.scaled_fp4_quant(q, q_fp4, q_sf, blk)
fp4quant_cuda.scaled_fp4_quant(k, k_fp4, k_sf, blk)

# ---------------- V: 用 trans 版本 -> (B, H, D, N/2) + scale (B,H,D,D/16) ----------------
v_fp4 = torch.empty(B, H, D, N // 2, dtype=torch.uint8, device=device)
v_sf = torch.empty(B, H, D, D // 16, dtype=torch.float8_e4m3fn, device=device)

fp4quant_cuda.scaled_fp4_quant_trans(v, v_fp4, v_sf, blk)

print("[DEBUG] q_fp4:", q_fp4.shape, q_fp4.dtype)
print("[DEBUG] k_fp4:", k_fp4.shape, k_fp4.dtype)
print("[DEBUG] v_fp4:", v_fp4.shape, v_fp4.dtype)
print("[DEBUG] q_sf :", q_sf.shape, q_sf.dtype)
print("[DEBUG] v_sf :", v_sf.shape, v_sf.dtype)

# ---------------- delta_s / p_sf 占位 ----------------
# 根据 api.cu 里的用法，delta_s 是 per-block softmax 调整 buffer
delta_s = torch.zeros(B, H, max(N // 128, 1), N, dtype=torch.bfloat16, device=device)

softmax_scale = 1.0 / (D ** 0.5)
is_causal = False
per_block_mean = False
is_training = False

# 你在 api.cu 里看到的 fwd 签名是：
# fwd(q, k, v, q_sf, k_sf, v_sf, delta_s,
#     seqlen_k, out, softmax_scale, is_causal, per_block_mean, is_training)

out, softmax_lse = fp4attn_cuda.fwd(
    q_fp4,
    k_fp4,
    v_fp4,
    q_sf,
    k_sf,
    v_sf,
    delta_s,
    N,              # seqlen_k
    None,           # out (让 kernel 自己 alloc / 返回)
    softmax_scale,
    is_causal,
    per_block_mean,
    is_training,
)

print("✅ fp4attn_cuda.fwd ok")
print("out:", out.shape, out.dtype)
print("softmax_lse:", softmax_lse.shape, softmax_lse.dtype)
