@echo off
setlocal EnableExtensions
title IEDA 3130 - Command Prompt with the ieda3130 conda environment
cd /d "%~dp0"

call :find_conda
if not defined CONDA_ROOT (
    echo [ERROR] Anaconda or Miniconda was not found on this computer.
    echo.
    echo   Install it from https://www.anaconda.com/download
    echo   ^(default options^), then run setup_windows.bat.
    echo.
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

echo ============================================================
echo   The "ieda3130" conda environment is ACTIVE in this window.
echo   You can now type commands such as:
echo.
echo      python check_env.py
echo      python pose_demo.py --image samples\sample_pose.jpg
echo      python pose_demo.py --webcam
echo      conda list
echo.
echo   Type "exit" to close this window.
echo ============================================================
echo.
cmd /k

exit /b 0

:find_conda
set "CONDA_ROOT="
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
exit /b 0
