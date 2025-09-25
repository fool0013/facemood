@echo off
setlocal
REM always run from this .bat's folder
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM venv paths
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

REM create venv on first run
if not exist "%PY%" (
  echo [INFO] Creating virtual environment...
  py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
)

REM upgrade pip
"%PY%" -m pip install --upgrade pip

REM install from requirements.txt if present
if exist "%ROOT%requirements.txt" (
  echo [INFO] Installing dependencies from requirements.txt...
  "%PY%" -m pip install -r "%ROOT%requirements.txt"
) else (
  echo [WARN] requirements.txt not found next to this .bat
)

REM ensure critical packages are present (auto-fix if missing)
echo [INFO] Verifying key modules...
"%PY%" -c "import cv2"           1>nul 2>nul || "%PIP%" install opencv-python
"%PY%" -c "import deepface"      1>nul 2>nul || "%PIP%" install deepface retina-face
"%PY%" -c "import tensorflow"    1>nul 2>nul || "%PIP%" install tensorflow==2.19.1 tf-keras==2.19.0
"%PY%" -c "import mediapipe"     1>nul 2>nul || "%PIP%" install mediapipe
"%PY%" -c "import pynput"        1>nul 2>nul || "%PIP%" install pynput
"%PY%" -c "import numpy"         1>nul 2>nul || "%PIP%" install numpy

REM run the app with the venv's python (no 'activate' needed)
set TF_ENABLE_ONEDNN_OPTS=0
"%PY%" facemood_deepface.py

echo.
pause
endlocal
