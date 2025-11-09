# sageattn3/patch_blackwell_auto.py

import torch, sys

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

BLACKWELL_KEYS = ["b100", "b200", "gb200", "rtx 5090", "rtx5090", "blackwell"]

def patch_blackwell_capability():
    """
    如果是 Blackwell 家族卡但 capability < (12,0)，
    强制把 torch 的 get_device_capability / _cuda_getDeviceCapability 改成 (12,0)。
    """
    try:
        if not torch.cuda.is_available():
            print(f"{YELLOW}[WARN] CUDA not available, skip Blackwell patch.{RESET}")
            return

        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        name_l = name.lower()

        is_blackwell = any(k in name_l for k in BLACKWELL_KEYS)
        need_patch = is_blackwell and cap[0] < 12

        if need_patch:
            print(f"{YELLOW}[PATCH] Detected {name} with reported capability={cap}, forcing to (12,0){RESET}")
            torch.cuda.get_device_capability = lambda device=0: (12, 0)
            try:
                torch._C._cuda_getDeviceCapability = lambda device=0: (12, 0)
            except Exception:
                pass
            print(f"{GREEN}[INFO] GPU recognized as Blackwell SM120. FP4 kernel allowed.{RESET}")
        else:
            print(f"{GREEN}[INFO] GPU={name}, capability={cap} (no Blackwell patch needed).{RESET}")

    except Exception as e:
        print(f"{RED}[WARN] Blackwell capability patch skipped: {e}{RESET}")
        sys.stdout.flush()
