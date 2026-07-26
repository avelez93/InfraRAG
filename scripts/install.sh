#!/usr/bin/env bash
# InfraRAG one-command installer (Linux / macOS)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/scripts/bootstrap.py" "$@"
