param(
  [string]$ConfigPath = "experiments/configs/deit_base_384.yaml"
)

$ErrorActionPreference = "Stop"

# Resolve project root from this script location
$projectDir = Split-Path $PSScriptRoot -Parent
$trainScript = Join-Path $projectDir "src/train.py"

# Normalize config path: if relative, treat as relative to project root
if (-Not [System.IO.Path]::IsPathRooted($ConfigPath)) {
  $ConfigPath = Join-Path $projectDir $ConfigPath
}

Write-Host "Project Dir: $projectDir" -ForegroundColor DarkGray
Write-Host "Train Script: $trainScript" -ForegroundColor DarkGray
Write-Host "Config: $ConfigPath" -ForegroundColor Cyan

python "$trainScript" --config "$ConfigPath"
