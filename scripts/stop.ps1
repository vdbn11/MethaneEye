$ErrorActionPreference = "SilentlyContinue"
$listeners = @(Get-NetTCPConnection -LocalPort 8787 -State Listen)
if (-not $listeners) {
    Write-Host "MethaneEye is not running on port 8787."
    exit 0
}
foreach ($listener in $listeners) {
    Stop-Process -Id $listener.OwningProcess -Force
}
Start-Sleep -Milliseconds 500
if (Get-NetTCPConnection -LocalPort 8787 -State Listen) {
    Write-Host "Could not stop MethaneEye. Close its terminal window manually."
    exit 1
}
Write-Host "MethaneEye stopped."
