@echo off
REM ===========================================================================
REM  Rice Disease Recognition -- difficulty-stratified subset selection
REM
REM  Runs src/select_difficulty_subset.py: scores every image in a pool with a
REM  trained model and, per class, extracts the window of images whose accuracy
REM  falls inside a target band (default 70-80%).
REM
REM  The result is a DIFFICULTY STRATUM for analysis (curriculum, error
REM  analysis) -- NOT a test set or a performance benchmark. The model's real
REM  accuracy is recorded in the output manifest.json.
REM
REM  Any extra arguments are passed straight through, e.g.
REM      run_select_subset.bat --num-workers 2
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM ===========================================================================
REM  EDIT THESE
REM ===========================================================================

REM Path to the trained checkpoint (.pth saved by src/train.py).
set CHECKPOINT=results\resnet50_full\best.pth

REM Candidate image pool (ImageFolder layout: one subfolder per class).
REM IMPORTANT: this MUST hold MORE than PER_CLASS images per class, otherwise
REM there is nothing to select. The processed test set is capped at 1000/class,
REM so point this at a larger pool (e.g. raw 2026 images).
set DATA_DIR=data\RiceCrossYear\test

REM How many images to select per class.
set PER_CLASS=1000

REM Target accuracy band, as fractions (0-1).
set TARGET_LO=0.70
set TARGET_HI=0.80

REM Output directory for the CSV, manifest, plot and (optional) copied images.
set OUT_DIR=results\difficulty_subset

REM Copy the selected image files into OUT_DIR\images\<class>\ ?   (yes / no)
set COPY_IMAGES=yes

REM ===========================================================================
REM  (usually no need to edit below this line)
REM ===========================================================================

REM --- Activate the virtual environment if it exists ------------------------
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: .venv not found. Run setup.bat first, or make sure Python and
    echo          the dependencies are already installed on PATH.
)

REM --- Sanity check: the checkpoint must exist ------------------------------
if not exist "%CHECKPOINT%" (
    echo ERROR: checkpoint not found: %CHECKPOINT%
    echo        Place the trained .pth there, or edit CHECKPOINT in this file.
    exit /b 1
)

REM --- GPU check: fall back to CPU if none is available ---------------------
echo Checking for a CUDA GPU ...
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo WARNING: no CUDA GPU detected -- running on CPU ^(slower^).
    set ALLOW_CPU=--allow-cpu
) else (
    set ALLOW_CPU=
)

REM --- Build the optional --copy-images flag --------------------------------
set COPY_FLAG=
if /I "%COPY_IMAGES%"=="yes" set COPY_FLAG=--copy-images

REM --- Run the selection ----------------------------------------------------
echo.
echo Running difficulty-subset selection ...
echo   checkpoint : %CHECKPOINT%
echo   pool       : %DATA_DIR%
echo   per-class  : %PER_CLASS%   band: %TARGET_LO% - %TARGET_HI%
echo   output     : %OUT_DIR%
echo.
python -m src.select_difficulty_subset ^
    --checkpoint "%CHECKPOINT%" ^
    --data-dir "%DATA_DIR%" ^
    --per-class %PER_CLASS% ^
    --target-lo %TARGET_LO% ^
    --target-hi %TARGET_HI% ^
    --out "%OUT_DIR%" ^
    %COPY_FLAG% %ALLOW_CPU% %*
if errorlevel 1 (
    echo.
    echo Selection reported errors. See %OUT_DIR%\selection.log
    exit /b 1
)

echo.
echo ===========================================================================
echo  Done. Outputs in %OUT_DIR%
echo    selected_images.csv     -- the chosen images
echo    manifest.json           -- metadata + the model's TRUE accuracy
echo    selection_overview.png  -- how each window was chosen
if /I "%COPY_IMAGES%"=="yes" echo    images\^<class^>\          -- copies of the selected image files
echo ===========================================================================
endlocal
