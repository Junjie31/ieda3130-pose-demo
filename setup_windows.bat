@echo off
setlocal EnableExtensions
title IEDA 3130 pose demo - first-time setup
cd /d "%~dp0"

echo ============================================================
echo   IEDA 3130 pose demo - first-time setup (Windows)
echo ============================================================
echo.

call :find_conda
if not defined CONDA_ROOT (
    echo [ERROR] Anaconda or Miniconda was not found on this computer.
    echo.
    echo   1. Download "Anaconda Distribution" from:
    echo        https://www.anaconda.com/download
    echo   2. Run the installer ^(default options are fine^).
    echo   3. Close this window, then double-click setup_windows.bat again.
    echo.
    pause
    exit /b 1
)
echo [ok] Found Conda at: %CONDA_ROOT%

call :find_env
if "%ENV_FOUND%"=="1" (
    echo [ok] Conda environment "ieda3130" already exists.
) else (
    echo [info] Creating Conda environment "ieda3130" with Python 3.12 ...
    call "%CONDA_BAT%" create -n ieda3130 python=3.12 -y
    if errorlevel 1 (
        echo [ERROR] Could not create the Conda environment. See the messages above.
        pause
        exit /b 1
    )
)

echo [info] Activating "ieda3130" ...
call "%CONDA_ROOT%\Scripts\activate.bat" ieda3130
if errorlevel 1 (
    echo [ERROR] Could not activate the environment. Run this script again.
    pause
    exit /b 1
)

echo [info] Installing Python packages - about 30 s, ~350 MB disk ...
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection and run again.
    pause
    exit /b 1
)

echo [info] Running the self-test - downloads the pose model, ~6 MB ...
python check_env.py
if errorlevel 1 (
    echo [ERROR] The self-test failed. Send the messages above to your TA.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete!
echo     Double-click run_webcam.bat     - live camera demo
echo     Drag a photo onto run_image.bat - single photo demo
echo     open_shell.bat                  - prompt with ieda3130 env active
echo ============================================================
echo.
pause
exit /b 0

REM ---------------------------------------------------------------
:find_conda
set "CONDA_ROOT="
set "CONDA_BAT="
for %%R in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%PROGRAMDATA%\anaconda3"
    "%PROGRAMDATA%\miniconda3"
    "C:\ProgramData\Anaconda3"
    "C:\ProgramData\Miniconda3"
    "C:\Anaconda3"
    "C:\Miniconda3"
) do (
    if not defined CONDA_ROOT if exist "%%~R\Scripts\activate.bat" set "CONDA_ROOT=%%~R"
)
if defined CONDA_ROOT (
    if exist "%CONDA_ROOT%\condabin\conda.bat" (
        set "CONDA_BAT=%CONDA_ROOT%\condabin\conda.bat"
    ) else (
        set "CONDA_BAT=%CONDA_ROOT%\Scripts\conda.exe"
    )
)
exit /b 0

:find_env
set "ENV_FOUND=0"
call "%CONDA_BAT%" env list > "%TEMP%\ieda_envlist.txt" 2>nul
findstr /C:"ieda3130" "%TEMP%\ieda_envlist.txt" >nul 2>nul && set "ENV_FOUND=1"
del "%TEMP%\ieda_envlist.txt" >nul 2>nul
exit /b 0
