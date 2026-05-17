param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto",
  [switch]$Tf32,
  [switch]$VerboseTraining,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = "$Root\runs\logs-paper-ptb-$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Jobs = @(
  @{ Name = "paper-small-baseline"; Model = "small"; Variant = "baseline"; Extra = @() },
  @{ Name = "paper-small-wt"; Model = "small"; Variant = "wt"; Extra = @() },
  @{ Name = "paper-small-pr"; Model = "small"; Variant = "pr"; Extra = @() },
  @{ Name = "paper-small-wt_pr"; Model = "small"; Variant = "wt_pr"; Extra = @() },

  @{ Name = "paper-large-baseline"; Model = "large"; Variant = "baseline"; Extra = @("--paper_test_eval") },
  @{ Name = "paper-large-wt"; Model = "large"; Variant = "wt"; Extra = @("--paper_test_eval") },

  @{ Name = "paper-bayes1500-baseline-wd1e-7"; Model = "bayes1500"; Variant = "baseline"; Extra = @("--legacy_weight_decay", "1e-7", "--paper_test_eval") },
  @{ Name = "paper-bayes1500-wt-wd0"; Model = "bayes1500"; Variant = "wt"; Extra = @("--legacy_weight_decay", "0", "--paper_test_eval") },
  @{ Name = "paper-bayes1500-wt-wd1e-7"; Model = "bayes1500"; Variant = "wt"; Extra = @("--legacy_weight_decay", "1e-7", "--paper_test_eval") }
)

$Manifest = foreach ($Job in $Jobs) {
  [pscustomobject]@{
    name = $Job.Name
    model = $Job.Model
    variant = $Job.Variant
    extra = ($Job.Extra -join " ")
  }
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "$LogDir\manifest.json"

foreach ($Job in $Jobs) {
  $OutDir = "$Root\runs\$($Job.Name)"
  $DoneJson = Get-ChildItem -Path $OutDir -Filter "*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($DoneJson -and -not $Force) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP $($Job.Name) existing=$($DoneJson.FullName)"
    continue
  }

  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  $LogPath = "$LogDir\$($Job.Name).log"
  $QuietArgs = @()
  if (-not $VerboseTraining) {
    $QuietArgs += "--quiet"
  }
  $PrecisionArgs = @()
  if ($Tf32) {
    $PrecisionArgs += "--tf32"
  }
  "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $($Job.Name)"

  & $Python "$Root\repro_pytorch\train_ptb.py" `
    --data_path "$Root\data\ptb" `
    --model $Job.Model `
    --variant $Job.Variant `
    --device $Device `
    --output_dir $OutDir `
    @($Job.Extra) `
    @PrecisionArgs `
    @QuietArgs *> $LogPath

  if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAIL $($Job.Name) log=$LogPath"
    exit $LASTEXITCODE
  }

  "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $($Job.Name) log=$LogPath"
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] paper PTB suite done; logs=$LogDir"
