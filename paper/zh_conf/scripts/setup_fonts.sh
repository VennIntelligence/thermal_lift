#!/usr/bin/env bash
# Download Fandol open-source fonts (CTAN) for zh_conf typography.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FONT_DIR="$ROOT/fonts"

if compgen -G "$FONT_DIR/FandolSong-Regular.otf" > /dev/null; then
  echo "Fandol fonts already present in $FONT_DIR"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$FONT_DIR"
curl -fsSL -o "$TMP/fandol.zip" "https://mirrors.ctan.org/fonts/fandol.zip"
unzip -qo "$TMP/fandol.zip" -d "$TMP/extract"
find "$TMP/extract" -name '*.otf' -exec cp {} "$FONT_DIR/" \;

echo "Installed Fandol fonts to $FONT_DIR"
ls -lh "$FONT_DIR"
