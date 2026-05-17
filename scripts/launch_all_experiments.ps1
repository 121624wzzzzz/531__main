param(
  [string]$Device = "auto",
  [Alias("LoraRank")]
  [int[]]$LoraRanks = @(1, 2, 4, 8, 16, 32),
  [Alias("RelaxationScale")]
  [double[]]$RelaxationScales = @(0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& "$Root\scripts\run_all_experiments.ps1" `
  -Device $Device `
  -LoraRanks $LoraRanks `
  -RelaxationScales $RelaxationScales
