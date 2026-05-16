#!/usr/bin/env bash

set -euo pipefail

REPO="${VIGREP_REPO:-KerrKerrKerr/vigrep}"
VERSION="${VIGREP_VERSION:-latest}"
PREFIX="${VIGREP_INSTALL_DIR:-$HOME/.local/bin}"
BIN_NAME="vigrep"

usage() {
    cat <<'EOF'
Usage: install.sh [--repo OWNER/REPO] [--version TAG] [--prefix DIR]

Install the latest released vigrep binary into a user-writable bin directory.

Options:
  --repo OWNER/REPO   GitHub repository to download from (default: KerrKerrKerr/vigrep)
  --version TAG       Release tag to install (default: latest)
  --prefix DIR        Install directory (default: ~/.local/bin)
  -h, --help          Show this help text

Environment:
  VIGREP_REPO         Same as --repo
  VIGREP_VERSION      Same as --version
  VIGREP_INSTALL_DIR  Same as --prefix
EOF
}

while (($# > 0)); do
    case "$1" in
        --repo)
            REPO="${2:?Missing value for --repo}"
            shift 2
            ;;
        --version)
            VERSION="${2:?Missing value for --version}"
            shift 2
            ;;
        --prefix)
            PREFIX="${2:?Missing value for --prefix}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$(uname -s)" in
    Linux) ;;
    *)
        echo "vigrep installer currently supports Linux only." >&2
        exit 1
        ;;
esac

case "$(uname -m)" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    *)
        echo "vigrep installer currently supports Linux x86_64 only." >&2
        exit 1
        ;;
esac

if [[ "$VERSION" == "latest" ]]; then
    DOWNLOAD_BASE="https://github.com/${REPO}/releases/latest/download"
else
    DOWNLOAD_BASE="https://github.com/${REPO}/releases/download/${VERSION}"
fi

ASSET_NAME="vigrep-linux-${ARCH}.tar.gz"
CHECKSUM_NAME="${ASSET_NAME}.sha256"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ARCHIVE_PATH="${TMP_DIR}/${ASSET_NAME}"
CHECKSUM_PATH="${TMP_DIR}/${CHECKSUM_NAME}"

echo "Downloading ${ASSET_NAME} from ${DOWNLOAD_BASE}..."
curl -fsSL "${DOWNLOAD_BASE}/${ASSET_NAME}" -o "${ARCHIVE_PATH}"
curl -fsSL "${DOWNLOAD_BASE}/${CHECKSUM_NAME}" -o "${CHECKSUM_PATH}"

echo "Verifying archive checksum..."
(cd "${TMP_DIR}" && sha256sum -c "${CHECKSUM_NAME}")

echo "Extracting vigrep..."
tar -C "${TMP_DIR}" -xzf "${ARCHIVE_PATH}"

mkdir -p "${PREFIX}"
cp "${TMP_DIR}/${BIN_NAME}" "${PREFIX}/${BIN_NAME}"
chmod 755 "${PREFIX}/${BIN_NAME}"

if [[ ":${PATH}:" != *":${PREFIX}:"* ]]; then
    echo "Installed to ${PREFIX}/${BIN_NAME}. Add ${PREFIX} to PATH if it is not already there." >&2
else
    echo "Installed to ${PREFIX}/${BIN_NAME}."
fi
