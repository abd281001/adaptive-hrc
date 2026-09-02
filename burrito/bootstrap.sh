#!/usr/bin/env bash
set -euo pipefail

integration_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${integration_root}/.." && pwd)"
talents_root="${integration_root}/third_party/talents-zsc"
venv_root="${integration_root}/.venv"

git -C "${project_root}" submodule update --init --recursive

python_binary="${BURRITO_PYTHON:-}"
if [[ -z "${python_binary}" ]] && command -v python3.10 >/dev/null 2>&1 \
    && python3.10 -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' >/dev/null 2>&1; then
    python_binary="$(command -v python3.10)"
fi
if [[ -z "${python_binary}" ]] && command -v pyenv >/dev/null 2>&1 \
    && PYENV_VERSION=3.10.20 pyenv which python >/dev/null 2>&1; then
    python_binary="$(PYENV_VERSION=3.10.20 pyenv which python)"
fi
if [[ -z "${python_binary}" ]]; then
    echo "A Python 3.10 interpreter is required. Set BURRITO_PYTHON=/path/to/python3.10." >&2
    exit 2
fi
"${python_binary}" -c 'import sys; expected=(3, 10); actual=sys.version_info[:2]; assert actual == expected, f"Burrito requires Python {expected[0]}.{expected[1]}, found {actual[0]}.{actual[1]}"'

if [[ ! -x "${venv_root}/bin/python" ]]; then
    "${python_binary}" -m venv "${venv_root}"
fi

"${venv_root}/bin/python" -m pip install --upgrade \
    pip==24.3.1 setuptools==72.1.0 wheel==0.45.1
"${venv_root}/bin/python" -m pip install \
    -r "${integration_root}/requirements-runtime.txt"
"${venv_root}/bin/python" -m pip install --no-deps \
    -e "${integration_root}/wrapper"
"${venv_root}/bin/python" -m adaptive_hrc_burrito verify
