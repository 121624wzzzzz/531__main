param(
  [string]$Device = "auto",
  [Alias("LoraRank")]
  [int[]]$LoraRanks = @(1, 2, 4, 8, 16, 32),
  [Alias("RelaxationScale")]
  [double[]]$RelaxationScales = @(0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LauncherLog = "$Root\runs\scheduled-launcher-$Stamp.log"
New-Item -ItemType Directory -Force -Path "$Root\runs" | Out-Null

Set-Location $Root
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] scheduled launcher start" | Tee-Object -FilePath $LauncherLog -Append

$RunArgs = @{
  Device = $Device
  LoraRanks = $LoraRanks
  RelaxationScales = $RelaxationScales
}
if ($DryRun) {
  $RunArgs.DryRun = $true
}

& "$Root\scripts\run_all_experiments.ps1" @RunArgs 2>&1 | Tee-Object -FilePath $LauncherLog -Append

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] scheduled launcher done" | Tee-Object -FilePath $LauncherLog -Append
