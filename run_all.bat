@echo off
REM ===========================================================================
REM  Rice Disease Recognition -- run the FULL pipeline (Windows)
REM  Preprocessing -> 8 training runs -> Grad-CAM -> analysis report.
REM
REM  Any extra arguments are passed straight to run_pipeline.py, e.g.
REM      run_all.bat --skip-preprocess
REM      run_all.bat --models resnet50 dinov2
REM ===========================================================================
setlocal

REM --- EDIT THIS: the folder that directly contains 2021, 2022, ... 2026 ------
set RICE_RAW_DIR=R:\Anti gravity\Rice Disease Project\Rice Disease

REM --- Optional overrides (defaults are fine for most users) -----------------
REM set RICE_PROCESSED_DIR=%~dp0data\RiceCrossYear
REM set RICE_RESULTS_DIR=%~dp0results
REM set RICE_NUM_WORKERS=4

REM --- Activate the virtual environment if it exists -------------------------
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: .venv not found. Run setup.bat first, or make sure Python and
    echo          the dependencies are already installed on PATH.
)

REM --- Make sure the GPU is visible -----------------------------------------
echo Checking for a CUDA GPU ...
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ERROR: no CUDA GPU detected. Aborting.
    echo        To run on CPU anyway: python run_pipeline.py --allow-cpu
    exit /b 1
)

REM --- Run the pipeline -----------------------------------------------------
python run_pipeline.py %*
if errorlevel 1 (
    echo.
    echo Pipeline reported errors. See results\pipeline.log
    exit /b 1
)

echo.
echo ===========================================================================
echo  Done. Open results\SUMMARY.md for the report.
echo ===========================================================================
endlocal
