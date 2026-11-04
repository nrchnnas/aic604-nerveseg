# Windows PowerShell version of env.sh. Only needed if you call nnUNetv2_* commands directly;
# nerveseg/nnunet/run.py sets these itself.
# Usage (from any folder):   . .\nerveseg\nnunet\env.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:nnUNet_raw = Join-Path $here "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $here "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $here "nnUNet_results"
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed, $env:nnUNet_results | Out-Null
Write-Host "nnUNet_raw=$env:nnUNet_raw"
