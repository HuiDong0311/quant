# import torch, time
# import fp4quant_cuda
# import fp4attn_cuda

# torch.manual_seed(0)
# device = "cuda"

# B, H, N, D = 1, 8, 128, 128
# dtype = torch.bfloat16
# blk = 128

# # ---------------- BF16 基线 ----------------
# q = torch.randn(B, H, N, D, device=device, dtype=dtype)
# k = torch.randn(B, H, N, D, device=device, dtype=dtype)
# v = torch.randn(B, H, N, D, device=device, dtype=dtype)

# softmax_scale = 1.0 / (D ** 0.5)

# torch.cuda.synchronize()
# t0 = time.time()
# out_bf16 = torch.nn.functional.scaled_dot_product_attention(
#     q, k, v, attn_mask=None, is_causal=False, scale=softmax_scale
# )
# torch.cuda.synchronize()
# t1 = time.time()

# print(f"[BF16] time = {(t1 - t0)*1000:.3f} ms, out shape = {out_bf16.shape}")

# # ---------------- FP4 量化 ----------------
# q_fp4 = torch.empty(B, H, N, D // 2, dtype=torch.uint8, device=device)
# k_fp4 = torch.empty(B, H, N, D // 2, dtype=torch.uint8, device=device)
# v_fp4 = torch.empty(B, H, D, N // 2, dtype=torch.uint8, device=device)

# q_sf = torch.empty(B, H, N, D // 16, dtype=torch.float8_e4m3fn, device=device)
# k_sf = torch.empty(B, H, N, D // 16, dtype=torch.float8_e4m3fn, device=device)
# v_sf = torch.empty(B, H, D, D // 16, dtype=torch.float8_e4m3fn, device=device)

# fp4quant_cuda.scaled_fp4_quant(q, q_fp4, q_sf, blk)
# fp4quant_cuda.scaled_fp4_quant(k, k_fp4, k_sf, blk)
# fp4quant_cuda.scaled_fp4_quant_trans(v, v_fp4, v_sf, blk)

# # ---------------- FP4 Attention ----------------
# delta_s = torch.zeros(B, H, max(N // 128, 1), N, dtype=torch.bfloat16, device=device)

# torch.cuda.synchronize()
# t2 = time.time()
# out_fp4, softmax_lse_fp4 = fp4attn_cuda.fwd(
#     q_fp4, k_fp4, v_fp4,
#     q_sf, k_sf, v_sf,
#     delta_s,
#     N,
#     None,
#     softmax_scale,
#     False,  # is_causal
#     False,  # per_block_mean
#     False,  # is_training
# )
# torch.cuda.synchronize()
# t3 = time.time()

# print(f"[FP4]  time = {(t3 - t2)*1000:.3f} ms, out shape = {out_fp4.shape}")

# # ---------------- 误差分析 ----------------
# out_fp4 = out_fp4.to(torch.bfloat16)
# diff = (out_bf16 - out_fp4).abs()
# l2 = diff.pow(2).mean().sqrt().item()
# max_err = diff.max().item()

# print(f"[DIFF] L2 = {l2:.6f}, MAX = {max_err:.6f}")

# # softmax LSE 误差（如果输出了）
# if softmax_lse_fp4 is not None:
#     with torch.no_grad():
#         qk = torch.einsum("bhqd,bhkd->bhqk", q, k) * softmax_scale
#         ref_lse = torch.logsumexp(qk, dim=-1)
#         lse_diff = (softmax_lse_fp4 - ref_lse).abs().mean().item()
#         print(f"[SOFTMAX LSE] mean abs diff = {lse_diff:.6f}")

# print("✅ Benchmark done.")


# compare_fp4_fp16.py
import torch
from sageattn3.blackwell import fp4_attention_test  # 假设模块暴露此接口

out_fp4, out_fp16 = fp4_attention_test(compare_fp16=True)
diff = (out_fp4 - out_fp16).float()
print(f"L2 error: {diff.pow(2).mean().sqrt():.6f}")
print(f"Max abs error: {diff.abs().max():.6f}")
print(f"Cosine sim: {torch.nn.functional.cosine_similarity(out_fp4.flatten(), out_fp16.flatten(), dim=0).item():.6f}")

