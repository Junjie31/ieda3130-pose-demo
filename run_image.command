#!/bin/bash
# IEDA 3130 pose demo - single photo (macOS)
# Drag a photo onto this file, or run:  bash run_image.command "photo.jpg"
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Usage: drag a photo file onto run_image.command,"
    echo "       or in Terminal:  bash run_image.command \"photo.jpg\""
    echo
    echo "Expected photo: side view, whole body, plain background."
    read -r -p "Press Enter to close..."
    exit 1
fi

find_conda() {
    CONDA_BASE=""
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/bin/conda" ]; then
            return 0
        fi
    fi
    for base in \
        "$HOME/anaconda3" "$HOME/miniconda3" \
        "$HOME/opt/anaconda3" "$HOME/opt/miniconda3" \
        "/opt/anaconda3" "/opt/miniconda3" \
        "/usr/local/anaconda3" "/usr/local/miniconda3"; do
        if [ -x "$base/bin/conda" ]; then
            CONDA_BASE="$base"
            return 0
        fi
    done
    return 1
}

if ! find_conda; then
    echo "[ERROR] Anaconda / Miniconda not found. Run setup_mac.command first."
    read -r -p "Press Enter to close..."
    exit 1
fi
export PATH="$CONDA_BASE/bin:$PATH"
eval "$("$CONDA_BASE/bin/conda" shell.bash hook 2>/dev/null)" || true

if ! conda activate ieda3130 2>/dev/null; then
    echo "[ERROR] Conda environment 'ieda3130' not found. Run setup_mac.command first."
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "Running the pose demo on: $1"
echo "(A window opens with the skeleton; press any key there to close it.)"
echo
python pose_demo.py --image "$1"
echo
echo "Results are in the outputs/ folder."
read -r -p "Press Enter to close..."
