param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskRoot=Join-Path $env:LOCALAPPDATA 'OneBeyond\tools'
$taskDestination=Join-Path $taskRoot 'openmvs-2.4.0'
$taskArchive=Join-Path $taskRoot 'downloads\OpenMVS_Windows_x64-2.4.0.zip'
$taskDigest='0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167'
if(!(Test-Path -LiteralPath $taskDestination) -and !$CheckOnly){
 if((Get-PSDrive C).Free -lt 2GB){throw 'At least 2 GB free space required'}
 New-Item -ItemType Directory -Force -Path (Split-Path -Parent $taskArchive) | Out-Null
 if(!(Test-Path -LiteralPath $taskArchive)){Invoke-WebRequest 'https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip' -OutFile $taskArchive}
 if((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash -ne $taskDigest){throw 'OpenMVS checksum mismatch; do not execute'}
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 $taskZip=[IO.Compression.ZipFile]::OpenRead($taskArchive)
 try{
  foreach($taskEntry in $taskZip.Entries){
   $taskResolved=[IO.Path]::GetFullPath((Join-Path $taskDestination $taskEntry.FullName))
   if(!$taskResolved.StartsWith($taskDestination+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Unsafe archive path'}
  }
 }finally{$taskZip.Dispose()}
 Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskDestination
}
if(!(Test-Path -LiteralPath $taskDestination)){throw 'OpenMVS is not installed'}
foreach($taskName in @('InterfaceCOLMAP','DensifyPointCloud','ReconstructMesh','RefineMesh','TextureMesh')){
 $taskExe=Get-ChildItem -LiteralPath $taskDestination -Recurse -File -Filter "$taskName.exe" | Select-Object -First 1
 if(!$taskExe){throw "Missing $taskName"}
 Write-Output $taskExe.FullName
 $taskSmoke=Join-Path $taskDestination 'smoke-logs'
 New-Item -ItemType Directory -Force -Path $taskSmoke | Out-Null
 Push-Location $taskSmoke
 try{& $taskExe.FullName --help 2>&1 | Out-Null;$taskExit=$LASTEXITCODE}finally{Pop-Location}
 # These official Windows executables log help to a file and return 1 for
 # --help. Check the actual banner/options rather than treating 1 as a crash.
 $taskLog=Get-ChildItem -LiteralPath $taskSmoke -Filter "$taskName-*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
 if(!$taskLog){throw "$taskName produced no diagnostic log"}
 $taskHelp=Get-Content -LiteralPath $taskLog.FullName -Raw
 if($taskExit -notin @(0,1) -or $taskHelp -notmatch 'OpenMVS x64 v2.4.0' -or $taskHelp -notmatch 'Available options:'){throw "$taskName failed its help verification"}
 Write-Output "Help verified: $($taskLog.FullName)"
}
Write-Output 'Official standalone CPU OpenMVS 2.4.0 verified. AGPL-3.0 applies; public adoption/distribution requires a separate user decision.'
