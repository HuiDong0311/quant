"""
patch_b200.py — Force-correct PyTorch's device capability reporting
for NVIDIA B200 (Blackwell) GPUs when it incorrectly reports (10,0).
"""

import torch

def patch_blackwell_capability():
    """Force PyTorch to recognize B200 (Blackwell) as SM120 (12,0)."""
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        if "B200" in name and cap[0] < 12:
            print(f"[PATCH] Detected {name} with reported capability={cap}, forcing to (12,0)")
            torch.cuda.get_device_capability = lambda device=0: (12, 0)
            torch._C._cuda_getDeviceCapability = lambda device=0: (12, 0)
        else:
            print(f"[INFO] GPU={name}, capability={cap} (no patch needed).")
    except Exception as e:
        print(f"[WARN] Blackwell capability patch skipped: {e}")
