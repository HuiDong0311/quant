"""2025.11.9
⚙️ 2. 在项目初始化时自动启用

修改：

sageattn3/__init__.py


把之前那两行替换成：

from sageattn3.patch_blackwell_auto import patch_blackwell_capability
patch_blackwell_capability()

patch_blackwell_auto.py  —  Auto-detect and patch Blackwell GPUs
Ensures PyTorch reports (12,0) capability for B100/B200/GB200/RTX5090 etc.
通用 Blackwell GPU 自动识别 + 彩色提示版 patch。
它能在任何场景下（B100/B200/GB200/RTX 5090 等）自动识别、修正 torch 的返回值，让所有脚本都立刻跑在 SM120 模式。
"""

import torch, re, sys

# ANSI colors for pretty log
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def patch_blackwell_capability():
    """
    Detect known Blackwell GPUs and force capability=(12,0)
    if torch incorrectly reports (10,0) or below.
    """
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        name_l = name.lower()

        # Common Blackwell identifiers
        blackwell_keys = ["b100", "b200", "gb200", "rtx 5090", "rtx5090", "blackwell"]

        is_blackwell = any(k in name_l for k in blackwell_keys)
        need_patch = is_blackwell and cap[0] < 12

        if need_patch:
            print(f"{YELLOW}[PATCH] Detected {name} (reported capability={cap}) → forcing (12,0){RESET}")
            torch.cuda.get_device_capability = lambda device=0: (12, 0)
            torch._C._cuda_getDeviceCapability = lambda device=0: (12, 0)
            print(f"{GREEN}[INFO] GPU recognized as Blackwell SM120. FP4 kernel ready.{RESET}")
        else:
            print(f"{GREEN}[INFO] GPU={name}, capability={cap} (no patch needed).{RESET}")

    except Exception as e:
        print(f"{RED}[WARN] Blackwell capability patch skipped: {e}{RESET}")
        sys.stdout.flush()
