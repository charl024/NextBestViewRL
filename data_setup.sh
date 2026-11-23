#!/usr/bin/env bash
# Set up data/: download ModelNet40, extract the 20 classes the project uses, and fix the
# broken .off headers. ModelNet is the only dataset.
#
# Usage:  ./data_setup.sh [--keep-zip] [--prune]
#   --keep-zip  keep data/raw/ModelNet40.zip after extracting (2 GB)
#   --prune     delete ModelNet40 class folders that the project does not use
#
# Environment overrides:
#   DATA_DIR      default: <repo>/data
#   MODELNET_URL  default: https://modelnet.cs.princeton.edu/ModelNet40.zip
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$REPO/data}"
MODELNET_URL="${MODELNET_URL:-https://modelnet.cs.princeton.edu/ModelNet40.zip}"

KEEP_ZIP=0
PRUNE=0
for arg in "$@"; do
    case "$arg" in
        --keep-zip) KEEP_ZIP=1 ;;
        --prune)    PRUNE=1 ;;
        -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Training and seen-test classes (ModelNet10), then the held-out unseen-test classes.
SEEN=(bathtub bed chair desk dresser monitor night_stand sofa table toilet)
UNSEEN=(airplane bench bookshelf bottle car cup guitar lamp piano vase)
CLASSES=("${SEEN[@]}" "${UNSEEN[@]}")

RAW="$DATA_DIR/raw"
MN="$RAW/ModelNet40"
ZIP="$RAW/ModelNet40.zip"
mkdir -p "$RAW" "$DATA_DIR/splits" "$DATA_DIR/cache"

# ---- 1. ModelNet40: download and extract the needed classes ----------------
missing=()
for c in "${CLASSES[@]}"; do
    [[ -d "$MN/$c/train" && -d "$MN/$c/test" ]] || missing+=("$c")
done

if (( ${#missing[@]} == 0 )); then
    echo "[modelnet] all ${#CLASSES[@]} classes present, skipping download"
else
    echo "[modelnet] missing ${#missing[@]} classes: ${missing[*]}"
    if [[ ! -f "$ZIP" ]]; then
        echo "[modelnet] downloading $MODELNET_URL (about 2 GB)"
        curl -L --fail -C - -o "$ZIP.part" "$MODELNET_URL"
        mv "$ZIP.part" "$ZIP"
    fi
    patterns=()
    for c in "${missing[@]}"; do patterns+=("ModelNet40/$c/*"); done
    echo "[modelnet] extracting"
    unzip -q -n "$ZIP" "${patterns[@]}" -d "$RAW"
    if (( KEEP_ZIP == 0 )); then
        rm -f "$ZIP"
    fi
fi

# ---- 2. Fix .off headers ----------------------------------------------------
# Some ModelNet files put the counts on the first line ("OFF1234 5678 0").
# Rewrite them as "OFF" followed by the counts on their own line.
dirs=()
for c in "${CLASSES[@]}"; do dirs+=("$MN/$c"); done
mapfile -t bad < <(find "${dirs[@]}" -name '*.off' -exec \
    awk 'FNR == 1 { if ($0 ~ /^OFF[ \t]*[0-9]/) print FILENAME; nextfile }' {} +)
for f in "${bad[@]}"; do
    sed -i '1s/^OFF[ \t]*/OFF\n/' "$f"
done
echo "[modelnet] fixed ${#bad[@]} .off headers"

# ---- 3. Optional: delete unused classes ------------------------------------
if (( PRUNE == 1 )); then
    for d in "$MN"/*/; do
        name="$(basename "$d")"
        keep=0
        for c in "${CLASSES[@]}"; do [[ "$name" == "$c" ]] && keep=1; done
        if (( keep == 0 )); then
            rm -rf "$d"
            echo "[modelnet] pruned $name"
        fi
    done
fi

# ---- 4. Summary ------------------------------------------------------------
echo
echo "classes   train  test"
for c in "${CLASSES[@]}"; do
    printf "%-12s %5d %5d\n" "$c" \
        "$(find "$MN/$c/train" -name '*.off' | wc -l)" \
        "$(find "$MN/$c/test" -name '*.off' | wc -l)"
done
du -sh "$DATA_DIR"
