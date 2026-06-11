# TactiVision'u tek komutla başlatır.
# Kullanım:  .\start.ps1
# Sonra tarayıcıda:  http://localhost:8000

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==> Python bağımlılıkları kontrol ediliyor..." -ForegroundColor Cyan
python -m pip install -q -r "$root/requirements-api.txt" imageio-ffmpeg

$dist = Join-Path $root "src/WebFrontEnd/dist"
if (-not (Test-Path $dist)) {
    Write-Host "==> Frontend derleniyor (ilk sefer)..." -ForegroundColor Cyan
    Push-Location "$root/src/WebFrontEnd"
    if (-not (Test-Path "node_modules")) { npm install }
    npm run build
    Pop-Location
}

Write-Host "==> Sunucu başlıyor: http://localhost:8000" -ForegroundColor Green
Push-Location "$root/src"
python -m uvicorn BackendAPI.app:app --host 0.0.0.0 --port 8000
Pop-Location
