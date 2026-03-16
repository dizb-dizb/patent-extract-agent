# 快速连接 AutoDL / 云端 GPU
# 用法: .\scripts\connect_autodl.ps1
# 需在 .env 中配置 CLOUD_SSH_HOST, CLOUD_SSH_PORT, CLOUD_SSH_USER

$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
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

if (-not $sshHost) {
    Write-Host "[fail] 未配置 CLOUD_SSH_HOST。请在 .env 中设置，参考 .env.example"
    exit 1
}

$port = if ($port) { $port } else { "22" }
$user = if ($user) { $user } else { "root" }

Write-Host "[info] 连接: $user@${sshHost}:$port"
Write-Host ""
ssh -p $port "${user}@${sshHost}"
