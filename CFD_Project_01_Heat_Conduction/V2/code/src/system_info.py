import platform
import psutil
import cpuinfo
import json


# =============================================================================
# System Setup
# =============================================================================


def get_system_info():
    cpu = cpuinfo.get_cpu_info()

    info = {
        "processor": cpu.get("brand_raw", "Unknown"),
        "architecture": platform.machine(),
        "system": platform.system(),
        "os_version": platform.version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

    return info


def save_system_info(filename="system_info.json"):
    info = get_system_info()

    with open(filename, "w") as f:
        json.dump(info, f, indent=4)

    print("System info saved to", filename)
    return info