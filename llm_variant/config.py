from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass


@dataclass
class SystemProfile:
    os_name: str
    machine: str
    cpu_count: int
    ram_gb: float | None


def detect_system_profile() -> SystemProfile:
    return SystemProfile(
        os_name=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_detect_ram_gb(),
    )


def recommended_ollama_model(profile: SystemProfile) -> str:
    ram = profile.ram_gb
    if ram is None:
        return "qwen2.5:3b-instruct"
    if ram >= 24:
        return "qwen2.5:14b-instruct"
    if ram >= 16:
        return "qwen2.5:7b-instruct"
    return "qwen2.5:3b-instruct"


def recommended_openai_model() -> str:
    return "gpt-4.1-mini"


def _detect_ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        return None
    return None

