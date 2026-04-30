#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export UV_CACHE_DIR="$SCRIPT_DIR/.uv_cache"
export LMDB_PURE=1
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
unset PYTHONHOME PYTHONPATH
uv sync "$@"