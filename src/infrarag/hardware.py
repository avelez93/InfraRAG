"""Detect local hardware capacity for model-tier selection."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of machine resources relevant to local LLM sizing."""

    total_ram_gb: float
    gpu_vram_gb: float | None
    apple_silicon: bool
    effective_gb: float
    notes: str


def _read_meminfo() -> str:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        return handle.read()


def _total_ram_gb() -> float:
    """Return total system RAM in GiB."""
    system = platform.system()
    try:
        if system == "Linux":
            text = _read_meminfo()
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
            if match:
                return int(match.group(1)) / (1024 * 1024)
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024**3)
        if system == "Windows":
            out = subprocess.check_output(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024**3)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except (AttributeError, OSError, ValueError):
        return 8.0


def _nvidia_vram_gb() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024.0)  # MiB -> GiB
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def detect_hardware() -> HardwareInfo:
    """Probe RAM / GPU and compute effective_gb for tier selection."""
    total_ram_gb = round(_total_ram_gb(), 2)
    gpu_vram_gb = _nvidia_vram_gb()
    if gpu_vram_gb is not None:
        gpu_vram_gb = round(gpu_vram_gb, 2)

    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    ram_budget = total_ram_gb * 0.45
    vram_budget = gpu_vram_gb or 0.0
    # Apple unified memory: treat a larger fraction of RAM as usable when no NVIDIA.
    if apple_silicon and gpu_vram_gb is None:
        ram_budget = total_ram_gb * 0.55

    effective_gb = round(max(ram_budget, vram_budget), 2)
    parts = [f"{total_ram_gb:g} GB RAM"]
    if gpu_vram_gb is not None:
        parts.append(f"{gpu_vram_gb:g} GB NVIDIA VRAM")
    if apple_silicon:
        parts.append("Apple Silicon")
    notes = ", ".join(parts)
    return HardwareInfo(
        total_ram_gb=total_ram_gb,
        gpu_vram_gb=gpu_vram_gb,
        apple_silicon=apple_silicon,
        effective_gb=effective_gb,
        notes=notes,
    )
