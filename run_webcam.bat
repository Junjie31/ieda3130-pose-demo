@echo off
setlocal EnableExtensions
title IEDA 3130 pose demo - live camera
cd /d "%~dp0"

call :find_conda
if not defined CONDA_ROOT (
    echo [ERROR] Anaconda / Miniconda not found.
    echo         Double-click setup_windows.bat first.
    pause
    exit /b 1
)
if not exist "%CONDA_ROOT%\envs\ieda3130\python.exe" (
    echo [ERROR] Conda environment "ieda3130" was not found in:
    echo         %CONDA_ROOT%\envs\ieda3130
    echo.
    echo         Double-click setup_windows.bat first - it creates the environment.
    pause
    exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" ieda3130 >nul 2>nul

echo Starting the live pose demo ...
echo Press S to save a snapshot, Q (or Esc) to quit the window.
echo.
python pose_demo.py --webcam
echo.
echo The demo has stopped. Results (if you pressed S) are in the outputs\ folder.
pause
exit /b 0

:find_conda
set "CONDA_ROOT="
for %%R in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%PROGRAMDATA%\anaconda3"
    "%PROGRAMDATA%\miniconda3"
    "C:\ProgramData\Anaconda3"
    "C:\ProgramData\Miniconda3"
    "C:\Anaconda3"
    "C:\Miniconda3"
) do (
    if not defined CONDA_ROOT if exist "%%~R\Scripts\activate.bat" set "CONDA_ROOT=%%~R"
)
exit /b 0
