#!/bin/bash
# IEDA 3130 pose demo - first-time setup (macOS)
# Double-click this file, or run:  bash setup_mac.command
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  IEDA 3130 pose demo - first-time setup (macOS)"
echo "============================================================"
echo

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
    echo "[ERROR] Anaconda or Miniconda was not found on this Mac."
    echo
    echo "  1. Download the macOS installer from:"
    echo "       https://www.anaconda.com/download"
    echo "     (pick the Apple Silicon or Intel installer that matches your Mac)"
    echo "  2. Double-click the downloaded .pkg and accept the default options."
    echo "  3. Open a NEW Terminal window and double-click this file again."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi
echo "[ok] Found Conda at: $CONDA_BASE"

export PATH="$CONDA_BASE/bin:$PATH"
# make `conda activate` work inside this script even if conda init was never run
eval "$("$CONDA_BASE/bin/conda" shell.bash hook 2>/dev/null)" || true

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "ieda3130"; then
    echo "[ok] Conda environment 'ieda3130' already exists."
else
    echo "[info] Creating Conda environment 'ieda3130' with Python 3.12 ..."
    conda create -n ieda3130 python=3.12 -y
fi

echo "[info] Activating 'ieda3130' ..."
conda activate ieda3130

echo "[info] Installing Python packages (about 30 s, ~350 MB disk) ..."
python -m pip install --disable-pip-version-check -r requirements.txt

echo "[info] Running the self-test (downloads the pose model, ~6 MB) ..."
python check_env.py

echo
echo "============================================================"
echo "  Setup complete!"
echo "    Double-click run_webcam.command      - live camera demo"
echo "    Drag a photo onto run_image.command  - single photo demo"
echo
echo "  First live run: allow camera access for Terminal when macOS asks,"
echo "  or enable it in System Settings > Privacy & Security > Camera."
echo "============================================================"
read -r -p "Press Enter to close..."
