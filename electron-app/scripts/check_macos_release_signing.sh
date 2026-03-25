#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This release check is for macOS only."
  exit 1
fi

if ! command -v security >/dev/null 2>&1; then
  echo "Unable to find macOS security tool in PATH."
  exit 1
fi

IDENTITIES="$(security find-identity -v -p codesigning 2>/dev/null || true)"
if ! grep -q "Developer ID Application:" <<<"${IDENTITIES}"; then
  echo "Missing required signing certificate: Developer ID Application"
  echo "Available signing identities:"
  if [[ -n "${IDENTITIES}" ]]; then
    echo "${IDENTITIES}"
  else
    echo "(none)"
  fi
  echo ""
  echo "Install a Developer ID Application certificate in Keychain before running build:mac:release."
  exit 1
fi

missing_env=0
for var_name in APPLE_ID APPLE_APP_SPECIFIC_PASSWORD APPLE_TEAM_ID; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing environment variable: ${var_name}"
    missing_env=1
  fi
done

if [[ "${missing_env}" -ne 0 ]]; then
  echo ""
  echo "Set notarization environment variables and retry build:mac:release."
  exit 1
fi

echo "Release signing precheck passed. Developer ID Application certificate found."
