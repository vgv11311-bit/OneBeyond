Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   ONE BEYOND - Spatial 3D Reconstruction" -ForegroundColor Cyan
Write-Host "   Prototype 1.0 (Localhost Server)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check Python and Node
$backendDir = Join-Path $PSScriptRoot "backend"
$frontendDir = Join-Path $PSScriptRoot "frontend"

Write-Host "`n[1/2] Starting Flask Backend API on http://localhost:5000..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python app.py
} -ArgumentList $backendDir

Start-Sleep -Seconds 2

Write-Host "`n[2/2] Starting Vite Frontend Dev Server on http://localhost:5173..." -ForegroundColor Green
Set-Location $frontendDir
npm.cmd run dev
