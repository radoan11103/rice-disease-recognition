@echo off
REM ===========================================================================
REM  Rice Disease Recognition -- one-time environment setup (Windows)
REM  Creates a virtual environment and installs every dependency, including a
REM  CUDA build of PyTorch for the RTX 3050.
REM ===========================================================================
setlocal

echo.
echo [1/4] Creating virtual environment (.venv) ...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: could not create the virtual environment.
    echo        Make sure Python 3.10/3.11 (64-bit) is installed and on PATH.
    exit /b 1
)

echo [2/4] Activating the environment ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/4] Installing PyTorch with CUDA (for the RTX 3050) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ERROR: PyTorch installation failed. Check your internet connection.
    exit /b 1
)

echo [4/4] Installing the remaining requirements ...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: requirements installation failed.
    exit /b 1
)

echo.
echo Verifying the GPU is visible to PyTorch ...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - check your NVIDIA driver')"

echo.
echo ===========================================================================
echo  Setup complete.
echo  Next: open run_all.bat, set RICE_RAW_DIR to your data folder, then run it.
echo ===========================================================================
endlocal
