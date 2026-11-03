#!/bin/bash
# IEDA 3130 pose demo - live camera demo (macOS)
# Double-click this file, or run:  bash run_webcam.command
set -e
cd "$(dirname "$0")"

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

echo "Starting the live pose demo ..."
echo "Press S to save a snapshot, Q (or Esc) to quit the window."
echo "If the camera window stays black: System Settings > Privacy & Security > Camera -> allow Terminal."
echo
python pose_demo.py --webcam
echo
echo "The demo has stopped. Results (if you pressed S) are in the outputs/ folder."
read -r -p "Press Enter to close..."
