# ===========================================================================
#  Rice Disease Recognition -- run the FULL pipeline (Windows PowerShell)
#  Preprocessing -> 8 training runs -> Grad-CAM -> analysis report.
#
#  Usage:
#      .\run_all.ps1
#      .\run_all.ps1 --skip-preprocess
#      .\run_all.ps1 --models resnet50 dinov2
#  (If scripts are blocked, run once:
#      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
# ===========================================================================

# --- EDIT THIS: the folder that directly contains 2021, 2022, ... 2026 ------
$env:RICE_RAW_DIR = "R:\Anti gravity\Rice Disease Project\Rice Disease"

# --- Optional overrides ----------------------------------------------------
# $env:RICE_PROCESSED_DIR = "$PSScriptRoot\data\RiceCrossYear"
# $env:RICE_RESULTS_DIR   = "$PSScriptRoot\results"
# $env:RICE_NUM_WORKERS   = "4"

# --- Activate the virtual environment if it exists -------------------------
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning ".venv not found. Run setup.bat first, or ensure Python and the dependencies are installed."
}

# --- Make sure the GPU is visible -----------------------------------------
Write-Host "Checking for a CUDA GPU ..."
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "No CUDA GPU detected. Aborting. (To run on CPU: python run_pipeline.py --allow-cpu)"
    exit 1
}

# --- Run the pipeline -----------------------------------------------------
python run_pipeline.py @args
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline reported errors. See results\pipeline.log"
    exit 1
}

Write-Host ""
Write-Host "==========================================================================="
Write-Host " Done. Open results\SUMMARY.md for the report."
Write-Host "==========================================================================="
