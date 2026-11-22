# CREMA-D -> RAVDESS transfer fine-tuning (docs/wav2vec2-crema-to-ravdess-transfer-handoff.md).
#   1. git pull --ff-only origin main, record commit hash
#   2. check inputs (CREMA-D best_model.pt, RAVDESS manifest/split/audio from the previous run)
#   3. fine-tune the CREMA-D checkpoint on RAVDESS train actors (validation early stopping,
#      CREMA-D validation retention check before/after)
#   4. one repeat-exploratory RAVDESS test evaluation + comparison with the previous models
#   5. zip results for Suryansh
# Earlier checkpoints/reports are never overwritten. Everything stays inside this folder.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File D:\SER-CAT\run_ravdess_transfer.ps1
# If GPU memory runs out:
#   powershell -ExecutionPolicy Bypass -File D:\SER-CAT\run_ravdess_transfer.ps1 -BatchSize 2

param([int]$BatchSize = 4)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

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
$Logs = Join-Path $Root "work\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw ".venv not found." }

$RunName = "cremad_init_ravdess_seed_20261122"
if ($BatchSize -ne 4) { $RunName = "${RunName}_bs$BatchSize" }
$RunDir = Join-Path $Rav "wav2vec2_experiments\$RunName"
$Commands = Join-Path $Rav "transfer_commands.txt"

function RunPy([string]$Name, [string[]]$PyArgs) {
    $log = Join-Path $Logs ("transfer_{0}_{1}.txt" -f $Name, $Stamp)
    Write-Host "Command: .venv\Scripts\python.exe $($PyArgs -join ' ')"
    Write-Host "Log: $log"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Py @PyArgs 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -ne "System.Management.Automation.RemoteException" } | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Add-Content $Commands (".venv\Scripts\python.exe {0}   (exit {1}, log {2})" -f ($PyArgs -join ' '), $code, (Split-Path $log -Leaf))
    if ($code -ne 0) { throw "Step '$Name' failed (exit $code). See $log" }
}

Step "1. Pulling latest code (git pull --ff-only origin main)"
$ErrorActionPreference = "Continue"
git pull --ff-only origin main
$pullCode = $LASTEXITCODE
$Commit = (git rev-parse HEAD).Trim()
$ErrorActionPreference = "Stop"
if ($pullCode -ne 0) { throw "git pull failed." }
Write-Host "Commit: $Commit"
Set-Content (Join-Path $Rav "transfer_git_commit.txt") $Commit -Encoding ascii

Step "2. Checking inputs"
$Init = Join-Path $Root "work\crema_d\wav2vec2_experiments\wav2vec2_base_seed_20261108\best_model.pt"
foreach ($p in $Init, "$Rav\manifest_ravdess_speech_4class.csv", "$Rav\split_summary_ravdess_seed_20261122.json",
               "$Rav\data\Actor_24", "$Rav\svm\test_results.json",
               "$Rav\wav2vec2_experiments\wav2vec2_base_ravdess_seed_20261122\test_results.json",
               "ml\finetune_cremad_wav2vec2_on_ravdess.py", "ml\evaluate_cremad_init_ravdess_test.py") {
    if (-not (Test-Path $p)) { throw "Missing: $p" }
}
Write-Host "All inputs present. CREMA-D audio (for the validation retention check): $(Test-Path "$Root\work\crema_d\data\AudioWAV")"

if (-not (Test-Path (Join-Path $RunDir "validation_results.json"))) {
    Step "3. Transfer fine-tuning on GPU (run $RunName, batch size $BatchSize)"
    RunPy "train_$RunName" @("ml\finetune_cremad_wav2vec2_on_ravdess.py", "--batch-size", "$BatchSize", "--run-name", $RunName)
} else { Write-Host "`n3. Run $RunName already trained - skipping" }

if (-not (Test-Path (Join-Path $RunDir "test_results.json"))) {
    Step "4. One repeat-exploratory RAVDESS test evaluation"
    RunPy "test_$RunName" @("ml\evaluate_cremad_init_ravdess_test.py", "--run-name", $RunName)
} else { Write-Host "`n4. Test already evaluated - skipping (it must only run once)" }

Step "5. Packaging results for Suryansh"
$ErrorActionPreference = "Continue"
& $Py -m pip freeze | Set-Content (Join-Path $Rav "transfer_pip_freeze.txt") -Encoding ascii
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | Set-Content (Join-Path $Rav "transfer_gpu_info.txt") -Encoding ascii
$ErrorActionPreference = "Stop"
$ZipOut = Join-Path $Root "work\ravdess_transfer_results_for_suryansh.zip"
$pack = @'
import pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1]); run = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3]); stamp = sys.argv[4]
rav = root / 'work' / 'ravdess'
items = {
    'scripts/finetune_cremad_wav2vec2_on_ravdess.py': root / 'ml' / 'finetune_cremad_wav2vec2_on_ravdess.py',
    'scripts/evaluate_cremad_init_ravdess_test.py': root / 'ml' / 'evaluate_cremad_init_ravdess_test.py',
    'scripts/run_ravdess_transfer.ps1': root / 'run_ravdess_transfer.ps1',
    'validation_results.json': run / 'validation_results.json',
    'test_results.json': run / 'test_results.json',
    'comparison_with_previous_ravdess_models.md': run / 'comparison_with_previous_ravdess_models.md',
    'comparison_with_previous_ravdess_models.json': run / 'comparison_with_previous_ravdess_models.json',
    'git_commit.txt': rav / 'transfer_git_commit.txt',
    'commands.txt': rav / 'transfer_commands.txt',
    'pip_freeze.txt': rav / 'transfer_pip_freeze.txt',
    'gpu_info.txt': rav / 'transfer_gpu_info.txt',
}
for log in sorted((root / 'work' / 'logs').glob('transfer_*.txt')):
    items['logs/' + log.name] = log
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in items.items():
        if path.is_file():
            z.write(path, name)
        else:
            print('missing, not packed:', path)
print('Wrote', out)
'@
$packFile = Join-Path $Tmp "pack_transfer.py"
Set-Content $packFile $pack -Encoding ascii
& $Py $packFile $Root $RunDir $ZipOut $Stamp
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }

Step "Done"
Get-Content (Join-Path $RunDir "comparison_with_previous_ravdess_models.md") | Select-Object -First 30
Write-Host "`nSend Suryansh: $ZipOut"
Write-Host "Send separately (large): $RunDir\best_model.pt"
