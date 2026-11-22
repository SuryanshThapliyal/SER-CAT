# RAVDESS two-model experiment (docs/ravdess-two-model-gpu-handoff.md).
# Runs every step in order and skips steps that already finished:
#   1. git pull --ff-only origin main, record commit hash
#   2. download + extract the official audio-only speech archive
#   3. build the manifest + fixed split summary
#   4. train MFCC + SVM (validation selection only)
#   5. train wav2vec2-base (validation selection only, GPU)
#   6. evaluate BOTH models once on the 112 test clips
#   7. zip everything to send back
# Everything stays inside this SER-CAT folder (work\ravdess, work\logs, work\cache).
# CREMA-D files are not touched.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\SER-CAT\run_ravdess.ps1
# If wav2vec2 runs out of GPU memory:
#   powershell -ExecutionPolicy Bypass -File D:\SER-CAT\run_ravdess.ps1 -BatchSize 2

param([int]$BatchSize = 4)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# ---- keep all caches inside the project
$Cache = Join-Path $Root "work\cache"; $Tmp = Join-Path $Cache "tmp"
New-Item -ItemType Directory -Force -Path $Cache, $Tmp | Out-Null
$env:TEMP = $Tmp; $env:TMP = $Tmp
$env:PIP_CACHE_DIR = Join-Path $Cache "pip"
$env:HF_HOME = Join-Path $Cache "huggingface"
$env:TORCH_HOME = Join-Path $Cache "torch"
$env:NUMBA_CACHE_DIR = Join-Path $Cache "numba"
$env:XDG_CACHE_HOME = $Cache
$env:PYTHONPYCACHEPREFIX = Join-Path $Cache "pycache"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$Rav  = Join-Path $Root "work\ravdess"
$Raw  = Join-Path $Rav "raw"
$Data = Join-Path $Rav "data"
$Logs = Join-Path $Root "work\logs"
New-Item -ItemType Directory -Force -Path $Rav, $Raw, $Logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw ".venv not found. Run run_wav2vec2.ps1 -SetupOnly first." }

$RunName = "wav2vec2_base_ravdess_seed_20261122"
if ($BatchSize -ne 4) { $RunName = "${RunName}_bs$BatchSize" }

function RunPy([string]$Name, [string[]]$PyArgs) {
    $log = Join-Path $Logs ("ravdess_{0}_{1}.txt" -f $Name, $Stamp)
    Write-Host "Command: .venv\Scripts\python.exe $($PyArgs -join ' ')"
    Write-Host "Log: $log"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Py @PyArgs 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -ne "System.Management.Automation.RemoteException" } | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Add-Content (Join-Path $Rav "commands.txt") (".venv\Scripts\python.exe {0}   (exit {1}, log {2})" -f ($PyArgs -join ' '), $code, (Split-Path $log -Leaf))
    if ($code -ne 0) { throw "Step '$Name' failed (exit $code). See $log" }
}

# ---- 1. pull
Step "1. Pulling latest code (git pull --ff-only origin main)"
$ErrorActionPreference = "Continue"
git pull --ff-only origin main
$pullCode = $LASTEXITCODE
$Commit = (git rev-parse HEAD).Trim()
$ErrorActionPreference = "Stop"
if ($pullCode -ne 0) { throw "git pull failed." }
Write-Host "Commit: $Commit"
Set-Content (Join-Path $Rav "git_commit.txt") $Commit
foreach ($f in "ml\ravdess_common.py", "ml\build_ravdess_manifest.py", "ml\train_ravdess_svm.py", "ml\train_ravdess_wav2vec2.py", "ml\evaluate_ravdess_test.py") {
    if (-not (Test-Path $f)) { throw "Missing $f" }
}

# ---- 2. download + extract
$Zip = Join-Path $Raw "Audio_Speech_Actors_01-24.zip"
$Url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
if (-not (Test-Path (Join-Path $Data "Actor_24"))) {
    if (-not (Test-Path $Zip) -or (Get-Item $Zip).Length -lt 200000000) {
        Step "2. Downloading Audio_Speech_Actors_01-24.zip (~208 MB) from Zenodo"
        $ErrorActionPreference = "Continue"
        curl.exe -L --fail --retry 3 -C - -o $Zip $Url
        $ErrorActionPreference = "Stop"
        if ($LASTEXITCODE -ne 0) { throw "Download failed. Download it manually from https://zenodo.org/records/1188976 into $Raw" }
    }
    Step "2. Extracting archive"
    New-Item -ItemType Directory -Force -Path $Data | Out-Null
    Expand-Archive $Zip $Data -Force
    # If the archive wrapped the actor folders in an extra directory, lift them up.
    if (-not (Test-Path (Join-Path $Data "Actor_01"))) {
        $inner = Get-ChildItem $Data -Recurse -Directory -Filter "Actor_01" | Select-Object -First 1
        if ($inner) { Get-ChildItem $inner.Parent.FullName -Directory -Filter "Actor_*" | Move-Item -Destination $Data }
    }
}
$wavCount = (Get-ChildItem $Data -Recurse -Filter *.wav).Count
Write-Host "WAV files under work\ravdess\data: $wavCount (expected 1440)"

# ---- install SVM dependencies (librosa etc.) into the same venv
$ErrorActionPreference = "Continue"
& $Py -c "import librosa, joblib" 2>$null
$haveLibrosa = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $haveLibrosa) {
    Step "Installing MFCC/SVM dependencies (ml\requirements.txt)"
    & $Py -m pip install -r ml\requirements.txt joblib
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

# ---- 3. manifest
if (-not (Test-Path (Join-Path $Rav "manifest_ravdess_speech_4class.csv"))) {
    Step "3. Building manifest and fixed split summary"
    RunPy "manifest" @("ml\build_ravdess_manifest.py")
} else { Write-Host "`n3. Manifest already built - skipping" }

# ---- 4. SVM
if (-not (Test-Path (Join-Path $Rav "svm\validation_results.json"))) {
    Step "4. Training MFCC + SVM (validation selection)"
    RunPy "svm_train" @("ml\train_ravdess_svm.py")
} else { Write-Host "`n4. SVM already trained - skipping" }

# ---- 5. wav2vec2
$RunDir = Join-Path $Rav "wav2vec2_experiments\$RunName"
if (-not (Test-Path (Join-Path $RunDir "validation_results.json"))) {
    Step "5. Training wav2vec2-base on GPU (run $RunName, batch size $BatchSize)"
    $ErrorActionPreference = "Continue"
    & $Py -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
    if (-not $ok) { throw "CUDA not available; wav2vec2 must train on the GPU." }
    RunPy "wav2vec2_train_$RunName" @("ml\train_ravdess_wav2vec2.py", "--batch-size", "$BatchSize", "--run-name", $RunName)
} else { Write-Host "`n5. wav2vec2 run $RunName already trained - skipping" }

# ---- 6. one-time test
if (-not (Test-Path (Join-Path $Rav "ravdess_comparison.md"))) {
    Step "6. One-time test evaluation of both models (112 clips, actors 16 17 19 24)"
    RunPy "test_eval" @("ml\evaluate_ravdess_test.py", "--run-name", $RunName)
} else { Write-Host "`n6. Test already evaluated - skipping (it must only run once)" }

# ---- 7. package
Step "7. Packaging results for Suryansh"
$ErrorActionPreference = "Continue"
& $Py -m pip freeze | Set-Content (Join-Path $Rav "pip_freeze.txt")
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | Set-Content (Join-Path $Rav "gpu_info.txt")
$ErrorActionPreference = "Stop"

$Out = Join-Path $Root "work\ravdess_results_for_suryansh"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force -Path "$Out\scripts", "$Out\svm", "$Out\wav2vec2", "$Out\logs" | Out-Null
Copy-Item "ml\ravdess_common.py", "ml\build_ravdess_manifest.py", "ml\train_ravdess_svm.py", "ml\train_ravdess_wav2vec2.py", "ml\evaluate_ravdess_test.py", "run_ravdess.ps1" "$Out\scripts\"
Copy-Item "$Rav\manifest_ravdess_speech_4class.csv", "$Rav\split_summary_ravdess_seed_20261122.json", "$Rav\ravdess_comparison.md", "$Rav\ravdess_comparison.json", "$Rav\git_commit.txt", "$Rav\commands.txt", "$Rav\pip_freeze.txt", "$Rav\gpu_info.txt" $Out
Copy-Item "$Rav\svm\validation_results.json", "$Rav\svm\test_results.json", "$Rav\svm\best_svm.joblib" "$Out\svm\"
Copy-Item "$RunDir\validation_results.json", "$RunDir\test_results.json" "$Out\wav2vec2\"
Get-ChildItem $Logs -Filter "ravdess_*" | Copy-Item -Destination "$Out\logs\"
$ZipOut = "$Out.zip"
if (Test-Path $ZipOut) { Remove-Item $ZipOut -Force -ErrorAction SilentlyContinue }
Compress-Archive "$Out\*" $ZipOut

Step "Done"
Get-Content (Join-Path $Rav "ravdess_comparison.md") | Select-Object -First 30
Write-Host "`nSend Suryansh: $ZipOut"
Write-Host "Send separately (large): $RunDir\best_model.pt"
