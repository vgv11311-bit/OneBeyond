param([switch]$CheckOnly, [switch]$WithReviewViewer, [switch]$WithHeic)
$ErrorActionPreference = 'Stop'
$toolsRoot = Join-Path $env:LOCALAPPDATA 'OneBeyond\tools'
$packages = @(
    @{ Name='brush-0.3.0'; Archive='brush-0.3.0.zip'; Url='https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip'; SHA256='b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6'; Executable='brush_app.exe' },
    @{ Name='colmap-4.2.1'; Archive='colmap-4.2.1-nocuda.zip'; Url='https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-nocuda.zip'; SHA256='c493d88cb4a43f21afba42cb55bf786e5434a0420e73db0ba1454c5d8b3686ae'; Executable='bin\colmap.exe' }
)
foreach ($package in $packages) {
    $destination = Join-Path $toolsRoot $package.Name
    $executable = Join-Path $destination $package.Executable
    if (!(Test-Path -LiteralPath $executable) -and !$CheckOnly) {
        if ((Get-PSDrive C).Free -lt 3GB) { throw 'Free at least 3 GB before installing portable reconstruction tools.' }
        $downloads = Join-Path $toolsRoot 'downloads'
        New-Item -ItemType Directory -Force -Path $downloads | Out-Null
        $archive = Join-Path $downloads $package.Archive
        if (!(Test-Path -LiteralPath $archive)) { Invoke-WebRequest -Uri $package.Url -OutFile $archive }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $package.SHA256) { throw "Checksum mismatch: $archive. Do not execute it." }
        if (Test-Path -LiteralPath $destination) { throw "Partial installation exists: $destination. Inspect it rather than overwrite it." }
        Expand-Archive -LiteralPath $archive -DestinationPath $destination
    }
    if (!(Test-Path -LiteralPath $executable)) { throw "Missing: $executable" }
    Write-Output $executable
    $helpLines = & $executable --help
    if ($LASTEXITCODE -ne 0) { throw "Tool failed to start: $executable" }
    $helpLines | Select-Object -First 3
}
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (!$ffmpeg) { throw 'FFmpeg is missing. Install a Windows build linked from https://ffmpeg.org/download.html, then rerun this check.' }
Write-Output $ffmpeg.Source
$versionLines = & $ffmpeg.Source -version
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg failed to start.' }
$versionLines | Select-Object -First 1
Write-Output 'Portable tools verified. A successful help command does not prove GPU training compatibility.'
if ($WithReviewViewer) {
    $vendor = Join-Path $toolsRoot 'spark-2.3.1'
    $module = Join-Path $vendor 'package\dist\spark.module.js'
    if (!(Test-Path -LiteralPath $module) -and !$CheckOnly) {
        New-Item -ItemType Directory -Force -Path $vendor | Out-Null
        $archive = Join-Path $vendor 'spark.tgz'
        Invoke-WebRequest -Uri 'https://registry.npmjs.org/@sparkjsdev/spark/-/spark-2.3.1.tgz' -OutFile $archive
        $hash = [Convert]::ToBase64String([Security.Cryptography.SHA512]::Create().ComputeHash([IO.File]::ReadAllBytes($archive)))
        if ($hash -ne 'K+SyIbkO/dx1TLLudK8Jp3/noS13z/kdNPo6OK2waxQaGy81HBmKGnNBT/t/K9Ew3dsIXmWqsJXxGfmAu1T5Jg==') { throw 'Spark archive integrity mismatch' }
        & tar -xf $archive -C $vendor
        if ($LASTEXITCODE -ne 0) { throw 'Spark extraction failed' }
    }
    if (!(Test-Path -LiteralPath $module)) { throw 'Optional local Spark review module is missing' }
    Write-Output $module
}
if ($WithHeic) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $localPython = Join-Path $projectRoot '.venv-local-splats\Scripts\python.exe'
    if (!$CheckOnly) {
        if (!(Test-Path -LiteralPath $localPython)) {
            $basePython = Join-Path $projectRoot '.venv-moge\Scripts\python.exe'
            & $basePython -m venv --system-site-packages (Join-Path $projectRoot '.venv-local-splats')
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated local intake environment' }
        }
        & $localPython -m pip install --only-binary=:all: -r (Join-Path $projectRoot 'backend\requirements-local-splats.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Optional HEIC dependencies failed to install' }
    }
    if (!(Test-Path -LiteralPath $localPython)) { throw 'Optional local HEIC environment is missing' }
    & $localPython -c 'import pillow_heif; import PIL; print(PIL.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'Optional HEIC decoding is unavailable' }
    Write-Output 'Local HEIC decoding available'
}
