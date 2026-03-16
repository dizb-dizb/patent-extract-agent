# AutoDL 一键部署与运行
# 用法:
#   .\scripts\deploy_and_run.ps1 -SetupOnly
#   .\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -MultiGpu
#   .\scripts\deploy_and_run.ps1 -Dataset fewnerd -Mode fewshot -DataStrategy augmented -MultiGpu

param(
    [switch]$SetupOnly,
    [string]$Dataset = "fewnerd",
    [string]$Mode = "fewshot",
    [string]$DataStrategy = "original",
    [switch]$MultiGpu,
    [string]$CudaVer = "cu124"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

$sshHost = $env:CLOUD_SSH_HOST
$port = $env:CLOUD_SSH_PORT
$user = $env:CLOUD_SSH_USER
$remoteWorkdir = $env:CLOUD_REMOTE_WORKDIR
$keyPath = $env:CLOUD_SSH_KEY_PATH

if (-not $sshHost) {
    Write-Host "[fail] 未配置 CLOUD_SSH_HOST。请在 .env 中设置"
    exit 1
}

$port = if ($port) { $port } else { "22" }
$user = if ($user) { $user } else { "root" }
$remoteWorkdir = if ($remoteWorkdir) { $remoteWorkdir } else { "/root/Graduation-Project" }

$scpArgs = @("-P", $port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL")
$sshArgs = @("-p", $port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL")
if ($keyPath) {
    $scpArgs += @("-i", $keyPath)
    $sshArgs += @("-i", $keyPath)
}

$remote = "${user}@${sshHost}"

Write-Host "[info] 确保远程目录存在"
& ssh @sshArgs $remote "mkdir -p $remoteWorkdir"

Write-Host "[info] 上传项目到 ${remote}:${remoteWorkdir}"
$tempDir = Join-Path $env:TEMP "grad-proj-deploy"
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
New-Item -ItemType Directory -Path $tempDir | Out-Null

# robocopy: /E=subdirs, /XD=exclude dirs, /NFL/NDL/NJH/NJS=quiet
$robocopyResult = robocopy $root $tempDir /E /XD "3" "__pycache__" ".git" "node_modules" /NFL /NDL /NJH /NJS
if ($robocopyResult -ge 8) { Write-Host "[warn] robocopy had errors (code $robocopyResult)" }

& scp @scpArgs -r "$tempDir\*" "${remote}:${remoteWorkdir}/"
Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[ok] 上传完成"

if ($SetupOnly) {
    Write-Host "[info] 仅安装环境 (SetupOnly)"
    $cmd = "cd $remoteWorkdir && CUDA_VER=$CudaVer bash scripts/remote_bootstrap.sh"
} else {
    $runArgs = @("python", "scripts/run_baseline.py", "--dataset", $Dataset, "--mode", $Mode, "--data-strategy", $DataStrategy)
    if ($MultiGpu) { $runArgs += "--multi-gpu" }
    $runCmd = $runArgs -join " "
    $cmd = "cd $remoteWorkdir && CUDA_VER=$CudaVer bash scripts/remote_bootstrap.sh $runCmd"
}

Write-Host "[info] 远程执行: $cmd"
& ssh @sshArgs $remote $cmd
