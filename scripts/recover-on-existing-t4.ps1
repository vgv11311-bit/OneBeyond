param([Parameter(Mandatory=$true)][string]$Archive,[Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference='Stop'
$taskGcloud='C:\Users\jalpa\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\GoogleCloudCLI-590.0.0\google-cloud-sdk\bin\gcloud.cmd'
$taskRemote='/home/jalpa/one-beyond-setup/cf54-recovery-20261021'
$taskCommon=@('--project=project-c7993406-5bfd-4d1f-ae4','--zone=us-west2-c','--quiet')
try {
    & $taskGcloud compute ssh onebeyondgpu @taskCommon --tunnel-through-iap --ssh-flag=-batch --command "mkdir $taskRemote"
    if($LASTEXITCODE -ne 0){throw 'Could not create a fresh recovery directory'}
    & $taskGcloud compute scp $Archive 'scripts/t4-recovery-worker.py' "onebeyondgpu:$taskRemote/" @taskCommon --tunnel-through-iap --scp-flag=-batch
    if($LASTEXITCODE -ne 0){throw 'Training upload failed'}
    $taskName=Split-Path -Leaf $Archive
    & $taskGcloud compute ssh onebeyondgpu @taskCommon --tunnel-through-iap --ssh-flag=-batch --command "python3 $taskRemote/t4-recovery-worker.py --archive $taskRemote/$taskName --brush /home/jalpa/one-beyond-setup/brush-0.3.0/brush-app-x86_64-unknown-linux-gnu/brush_app"
    if($LASTEXITCODE -ne 0){throw 'Remote training failed; do not publish an incomplete export'}
    & $taskGcloud compute scp "onebeyondgpu:$taskRemote/result.tar.gz" $Destination @taskCommon --tunnel-through-iap --scp-flag=-batch
    if($LASTEXITCODE -ne 0){throw 'Could not retrieve the final training result'}
} finally {
    & $taskGcloud compute instances stop onebeyondgpu @taskCommon --format='value(status)'
    if($LASTEXITCODE -ne 0){Write-Error 'VM stop failed. Stop onebeyondgpu immediately in Google Cloud Console.'}
}
