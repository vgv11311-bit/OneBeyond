param([string]$Editor = 'C:\Program Files\Unity\Hub\Editor\6000.6.6f1\Editor\Unity.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$unityRoot = Join-Path $projectRoot 'unity'
$editorRoot = Split-Path -Parent $Editor
if (-not (Test-Path -LiteralPath $Editor)) { throw 'Install Unity 6000.6.6f1 or pass -Editor with its full path.' }
$packageManager = Join-Path $editorRoot 'Data\Resources\PackageManager\Server\UnityPackageManager.exe'
if (-not (Test-Path -LiteralPath $packageManager)) { throw "Unity installation is incomplete: missing $packageManager. Repair/reinstall this Editor using Unity Hub before building." }
if (-not (Test-Path -LiteralPath (Join-Path $editorRoot 'Data\PlaybackEngines\WebGLSupport'))) { throw 'Add Web Build Support in Unity Hub.' }
$logPath = Join-Path $projectRoot 'reports\unity-build.log'
$process = Start-Process -FilePath $Editor -WindowStyle Hidden -PassThru -ArgumentList @(
    '-batchmode', '-quit', '-projectPath', ('"' + $unityRoot + '"'), '-buildTarget', 'WebGL',
    '-executeMethod', 'OneBeyondBuild.Web', '-logFile', ('"' + $logPath + '"')
)
$process.WaitForExit()
if ($process.ExitCode -ne 0) { throw "Unity build failed (exit $($process.ExitCode)). See $logPath" }
if (-not (Select-String -LiteralPath $logPath -Pattern 'ONE_BEYOND_WEB_BUILD_OK' -Quiet)) { throw 'Unity did not confirm a successful build; existing player files were not accepted.' }
Push-Location (Join-Path $projectRoot 'frontend')
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Output 'Unity player and website build completed. Browser acceptance testing is still required.'
