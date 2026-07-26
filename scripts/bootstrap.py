#!/usr/bin/env python3
"""Cross-platform InfraRAG bootstrap: venv, deps, tier choice, Ollama models."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrarag.config import write_local_ollama_config  # noqa: E402
from infrarag.hardware import detect_hardware  # noqa: E402
from infrarag.model_tiers import (  # noqa: E402
    EMBED_MODEL,
    TIERS,
    format_tiers_table,
    get_tier,
    select_tier,
)


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def _ensure_python() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit(f"Python 3.11+ required; found {sys.version}")


def _ensure_venv() -> Path:
    py = _venv_python()
    if not py.is_file():
        print("Creating virtualenv at .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")])
    return py


def _pip_install(py: Path) -> None:
    print("Installing InfraRAG package (editable + dev) ...")
    subprocess.check_call([str(py), "-m", "pip", "install", "-U", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-e", ".[dev]"], cwd=REPO_ROOT)


def _noninteractive(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    return os.environ.get("INFRARAG_BOOTSTRAP_NONINTERACTIVE", "").strip() in {
        "1",
        "true",
        "True",
        "yes",
        "YES",
    }


def _prompt_tier(recommended_id: str, *, noninteractive: bool) -> str:
    if noninteractive:
        print(f"Non-interactive mode: using recommended tier '{recommended_id}'.")
        return recommended_id

    print()
    print(format_tiers_table(recommended_id=recommended_id))
    print()
    print("Choose a tier by name (nano..xlarge), number 1-7, Enter, or 'default'.")
    while True:
        raw = input(f"Choose tier [default={recommended_id}]: ").strip()
        if raw == "" or raw.lower() == "default":
            return recommended_id
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(TIERS):
                return TIERS[idx - 1].id
            print(f"Enter a number between 1 and {len(TIERS)}.")
            continue
        try:
            return get_tier(raw).id
        except KeyError:
            print(f"Unknown tier '{raw}'. Try again.")


def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        import httpx
    except ImportError:
        return False
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _which_ollama() -> str | None:
    return shutil.which("ollama")


def _try_install_ollama() -> None:
    system = platform.system()
    if system in {"Linux", "Darwin"}:
        print("Attempting Ollama install via official script ...")
        proc = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            check=False,
        )
        if proc.returncode != 0:
            print(
                "Ollama auto-install failed. Install manually: https://ollama.com",
                file=sys.stderr,
            )
        return
    if system == "Windows":
        winget = shutil.which("winget")
        if winget:
            print("Attempting Ollama install via winget ...")
            proc = subprocess.run(
                [winget, "install", "-e", "--id", "Ollama.Ollama"],
                check=False,
            )
            if proc.returncode == 0:
                return
        print(
            "Install Ollama manually from https://ollama.com/download then re-run bootstrap.",
            file=sys.stderr,
        )
        return
    print("Unsupported OS for auto Ollama install. See https://ollama.com", file=sys.stderr)


def _ensure_ollama() -> None:
    if _ollama_available():
        print("Ollama is reachable.")
        return
    if not _which_ollama():
        _try_install_ollama()
    # Try starting serve in background on Unix if binary exists but API down
    if _which_ollama() and not _ollama_available():
        print("Starting `ollama serve` in background ...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        import time

        for _ in range(20):
            time.sleep(0.5)
            if _ollama_available():
                print("Ollama is reachable.")
                return
    if not _ollama_available():
        print(
            "WARNING: Ollama API not reachable. Install/start Ollama, then run:\n"
            "  ollama pull nomic-embed-text\n"
            "  ollama pull <your-chat-model>",
            file=sys.stderr,
        )


def _pull_models(chat_model: str, embed_model: str) -> None:
    if not _which_ollama():
        print("Skipping model pull (ollama CLI not found).", file=sys.stderr)
        return
    for model in (embed_model, chat_model):
        print(f"Pulling {model} ...")
        proc = subprocess.run(["ollama", "pull", model], check=False)
        if proc.returncode != 0:
            print(f"WARNING: failed to pull {model}", file=sys.stderr)


def _seed_env(chat_model: str, embed_model: str) -> None:
    env_path = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if env_path.exists() or not example.is_file():
        return
    text = example.read_text(encoding="utf-8")
    text = text.replace("qwen2.5:3b", chat_model)
    text = text.replace("nomic-embed-text", embed_model)
    env_path.write_text(text, encoding="utf-8")
    print(f"Created {env_path} from .env.example")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap InfraRAG")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive: use hardware-recommended tier",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip venv/pip (only configure models / Ollama)",
    )
    args = parser.parse_args(argv)

    _ensure_python()
    os.chdir(REPO_ROOT)

    if not args.skip_install:
        py = _ensure_venv()
        _pip_install(py)
    else:
        print("Skipping package install (--skip-install).")

    hw = detect_hardware()
    recommended = select_tier(hw.effective_gb)
    print()
    print(
        f"Detected: {hw.notes} -> effective_gb={hw.effective_gb:g} "
        f"-> recommended: {recommended.id} ({recommended.chat_model})"
    )

    chosen_id = _prompt_tier(recommended.id, noninteractive=_noninteractive(args))
    chosen = get_tier(chosen_id)
    print(f"Selected tier: {chosen.id} ({chosen.chat_model})")

    local_path = write_local_ollama_config(
        chat_model=chosen.chat_model,
        embed_model=EMBED_MODEL,
    )
    print(f"Wrote {local_path}")
    _seed_env(chosen.chat_model, EMBED_MODEL)

    _ensure_ollama()
    _pull_models(chosen.chat_model, EMBED_MODEL)

    print()
    print("Bootstrap complete. Start the UI with:")
    print("  source .venv/bin/activate   # Windows: .venv\\Scripts\\activate")
    print("  infrarag")
    print("  # or: python -m infrarag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
