$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"

$existing = @(Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue)
if ($existing) {
    Write-Host "MethaneEye is already running. Opening http://127.0.0.1:8787"
    Start-Process "http://127.0.0.1:8787"
    exit 0
}

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install numpy==1.26.4 pandas matplotlib Pillow
    & $python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    & $python -m pip install segmentation-models-pytorch==0.3.4 fastapi "uvicorn[standard]"
}

Write-Host "Starting MethaneEye on http://127.0.0.1:8787"
Start-Process powershell -ArgumentList @("-NoProfile", "-Command", "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:8787'") -WindowStyle Hidden
& $python -m src.server
