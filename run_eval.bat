@echo off
REM ===========================================================================
REM  Rice Disease Recognition -- re-run EVALUATION ONLY (Windows)
REM
REM  Loads the trained best.pth checkpoints and scores them on the test
REM  directory (and val directory, if present). No training happens and no
REM  weights change. Results go to a SEPARATE folder so the original
REM  results\<run>\ files are left untouched.
REM
REM  Any extra arguments pass straight through, e.g.
REM      run_eval.bat --model resnet50 --mode full
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM ===========================================================================
REM  EDIT THESE
REM ===========================================================================

REM The TEST image directory (ImageFolder layout: one subfolder per class).
set TEST_DIR=results\difficulty_subset\images
REM Optional VAL directory -- used only for the generalization gap. If you
REM don't have one, leave this as-is; it is skipped automatically when absent.
set VAL_DIR=data\RiceCrossYear\val

REM Where the evaluation outputs are written.
set OUT_DIR=results\evaluation

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

REM --- Sanity check: the test directory must exist --------------------------
if not exist "%TEST_DIR%" (
    echo ERROR: test directory not found: %TEST_DIR%
    echo        Edit TEST_DIR in this file to point at your test folder.
    exit /b 1
)

REM --- GPU check: fall back to CPU if none is available ---------------------
echo Checking for a CUDA GPU ...
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo WARNING: no CUDA GPU detected -- evaluating on CPU ^(slower^).
    set ALLOW_CPU=--allow-cpu
) else (
    set ALLOW_CPU=
)

REM --- Run evaluation only --------------------------------------------------
echo.
echo Re-running evaluation on %TEST_DIR% ...
python -m src.evaluate ^
    --test-dir "%TEST_DIR%" ^
    --val-dir "%VAL_DIR%" ^
    --out "%OUT_DIR%" ^
    %ALLOW_CPU% %*
if errorlevel 1 (
    echo.
    echo Evaluation reported errors. See %OUT_DIR%\evaluate.log
    exit /b 1
)

echo.
echo ===========================================================================
echo  Done. Evaluation outputs in %OUT_DIR%
echo    summary.csv                              -- all runs: val/test metrics
echo    ^<model^>_^<mode^>\test_metrics.json         -- per-run metrics
echo    ^<model^>_^<mode^>\confusion_matrix_test.png -- per-run confusion matrix
echo ===========================================================================
endlocal
