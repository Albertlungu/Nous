#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${REPO_ROOT}/electron-app/backend-dist"
BUILD_DIR="${REPO_ROOT}/electron-app/backend-build"

if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Unable to find a Python interpreter."
  exit 1
fi

echo "Building backend executable with ${PYTHON_BIN}"
rm -rf "${DIST_DIR}" "${BUILD_DIR}"
mkdir -p "${DIST_DIR}" "${BUILD_DIR}"

"${PYTHON_BIN}" -m pip install --upgrade pyinstaller
"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name nous-api-server \
  --distpath "${DIST_DIR}" \
  --workpath "${BUILD_DIR}/work" \
  --specpath "${BUILD_DIR}" \
  --paths "${REPO_ROOT}" \
  --additional-hooks-dir "${REPO_ROOT}/electron-app/pyinstaller-hooks" \
  --hidden-import tiktoken_ext \
  --hidden-import tiktoken_ext.openai_public \
  --collect-all tiktoken_ext \
  "${REPO_ROOT}/api/server.py"

chmod +x "${DIST_DIR}/nous-api-server"
echo "Backend executable generated at ${DIST_DIR}/nous-api-server"
