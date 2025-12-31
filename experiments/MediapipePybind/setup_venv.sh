#!/usr/bin/env bash
set -euo pipefail

# Скрипт: создать venv и поставить requirements.txt.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${project_dir}/venv"
requirements_file="${project_dir}/requirements.txt"

python_create_cmd="python3.12"
if ! command -v "${python_create_cmd}" >/dev/null 2>&1; then
  python_create_cmd="python3"
fi

if [[ ! -d "${venv_dir}" ]]; then
  "${python_create_cmd}" -m venv "${venv_dir}"
fi

venv_python="${venv_dir}/bin/python"
"${venv_python}" -m pip install -r "${requirements_file}"
