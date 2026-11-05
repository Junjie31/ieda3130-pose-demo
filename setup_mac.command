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
    echo "     Apple Silicon (M1 and later) only - Intel Macs have no official"
    echo "     Anaconda installer any more; use Route B (python.org Python) or Colab."
    echo "  2. Double-click the downloaded .pkg and accept the default options."
    echo "     The graphical installer puts Anaconda in /opt/anaconda3 and asks for"
    echo "     an administrator password; the terminal .sh installer is admin-free"
    echo "     and installs to ~/anaconda3. This script finds either location."
    echo "  3. Open a NEW Terminal window and double-click this file again."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi
echo "[ok] Found Conda at: $CONDA_BASE"

export PATH="$CONDA_BASE/bin:$PATH"
# make `conda activate` work inside this script even if conda init was never run
eval "$("$CONDA_BASE/bin/conda" shell.bash hook 2>/dev/null)" || true

# Anaconda's Terms of Service must be accepted before `conda create` works with a
# fresh 2025+ Anaconda installed from the terminal .sh installer (the graphical
# .pkg accepts them automatically; these calls are harmless there).
"$CONDA_BASE/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"$CONDA_BASE/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "ieda3130"; then
    echo "[ok] Conda environment 'ieda3130' already exists."
else
    echo "[info] Creating Conda environment 'ieda3130' with Python 3.12 ..."
    conda create -n ieda3130 python=3.12 -y
fi

if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "ieda3130"; then
    echo "[ERROR] The environment 'ieda3130' was not created."
    echo "        Run this script again; if it keeps failing, create it by hand:"
    echo "            conda create -n ieda3130 python=3.12 -y"
    read -r -p "Press Enter to close..."
    exit 1
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
echo "    To type commands yourself: open Terminal and run:  conda activate ieda3130"
echo "    No admin rights / no Anaconda? Use the Colab notebook instead:"
echo "      https://colab.research.google.com/github/Junjie31/ieda3130-pose-demo/blob/main/IEDA3130_Pose_Demo_Colab.ipynb"
echo
echo "  First live run: allow camera access for Terminal when macOS asks,"
echo "  or enable it in System Settings > Privacy & Security > Camera."
echo "============================================================"
read -r -p "Press Enter to close..."
