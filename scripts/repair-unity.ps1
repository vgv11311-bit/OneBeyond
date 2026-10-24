param(
    [string]$Installer = (Join-Path $env:TEMP 'one-beyond-unity-repair\UnitySetup64-6000.6.6f1.exe'),
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$destination = 'C:\Program Files\Unity\Hub\Editor\6000.6.6f1'
$required = @(
    'Editor\Unity.exe',
    'Editor\Data\Resources\PackageManager\Server\UnityPackageManager.exe',
    'Editor\Data\Tools\UnityShaderCompiler.exe',
    'Editor\Data\PlaybackEngines\WebGLSupport\UnityEditor.WebGL.Extensions.dll'
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $destination $_)) })
if ($missing.Count -eq 0) {
    Write-Output 'Required Unity files are present.'
    if (-not $CheckOnly) { & (Join-Path $PSScriptRoot 'build-unity.ps1') }
    return
}
Write-Output ('Missing required files: ' + ($missing -join ', '))
$freeGiB = (Get-PSDrive C).Free / 1GB
Write-Output ('C: free space: {0:N2} GiB' -f $freeGiB)
# Conservative headroom for the installer temporary files and final files.
# A badge in Unity Hub is not proof that its installer succeeded.
if ($freeGiB -lt 20) { throw 'Free space until C: has at least 20 GiB before retrying the Editor repair. The downloaded installer can be reused.' }
if (-not (Test-Path -LiteralPath $Installer)) { throw "Verified installer not found: $Installer" }
if ((Get-Item -LiteralPath $Installer).Length -ne 3856567088 -or
    (Get-FileHash -LiteralPath $Installer -Algorithm MD5).Hash -ne '027F11738EA1FDA86843C60E67AFCDE9') {
    throw 'The installer does not match the Unity-published size/checksum for 6000.6.6f1 x64.'
}
if ($CheckOnly) { Write-Output 'Repair preflight passed. No installer was started.'; return }
Write-Output 'Windows may request administrator approval to repair the existing Unity Editor.'
# NSIS requires /D last and without quotes, including destinations with spaces.
$installProcess = Start-Process -FilePath $Installer -ArgumentList ('/S /D=' + $destination) -Verb RunAs -WindowStyle Hidden -PassThru -Wait
if ($installProcess.ExitCode -ne 0) { throw "Unity installer failed with exit $($installProcess.ExitCode). Do not trust Hub's installed badge." }
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $destination $_)) })
if ($missing.Count) { throw ('Unity repair is incomplete: ' + ($missing -join ', ')) }
Write-Output 'Unity required files verified. Starting the web viewer build.'
& (Join-Path $PSScriptRoot 'build-unity.ps1')
