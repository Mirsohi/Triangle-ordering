#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${project_root}/vendor/volt}"
commit="31ea49141cc75d9a547c16a5c954c881676ae109"

if [[ -e "${destination}" ]]; then
  echo "destination already exists: ${destination}" >&2
  exit 2
fi

git clone https://github.com/lecfab/volt.git "${destination}"
git -C "${destination}" checkout --detach "${commit}"
git -C "${destination}" apply "${project_root}/third_party/volt_reproducibility.patch"
make -C "${destination}" -j2

echo "built ${destination}/volt at pinned commit ${commit}"
