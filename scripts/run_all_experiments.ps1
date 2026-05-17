param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto",
  [Alias("LoraRank")]
  [int[]]$LoraRanks = @(1, 2, 4, 8, 16, 32),
  [Alias("RelaxationScale")]
  [double[]]$RelaxationScales = @(0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
  [int]$MaxJobs = 0,
  [Nullable[int]]$MaxEpochs = $null,
  [Nullable[int]]$MaxTrainBatches = $null,
  [Nullable[int]]$MaxEvalBatches = $null,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = "$Root\runs\logs-$RunStamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RankSensitiveVariants = @("s4", "s5", "s6", "s7", "s9", "s10", "s12", "s13")
$ScaleSensitiveVariants = @("s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13")

function Format-ScaleTag([double]$Scale) {
  $ScaleText = $Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
  return "scale$($ScaleText.Replace('-', 'm').Replace('.', 'p'))"
}

function Get-RanksForVariant([string]$Variant) {
  if ($RankSensitiveVariants -contains $Variant) {
    return $LoraRanks
  }
  return @($LoraRanks[0])
}

function Get-ScalesForVariant([string]$Variant) {
  if ($ScaleSensitiveVariants -contains $Variant) {
    return $RelaxationScales
  }
  return @($RelaxationScales[0])
}

$Jobs = @(
  @{ Name = "small-baseline"; Model = "small"; Variant = "baseline"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @() },
  @{ Name = "small-wt"; Model = "small"; Variant = "wt"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @() },
  @{ Name = "small-pr"; Model = "small"; Variant = "pr"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @() },
  @{ Name = "small-wt_pr"; Model = "small"; Variant = "wt_pr"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @() },

  @{ Name = "bayes4090-baseline-wd1e-7"; Model = "bayes4090"; Variant = "baseline"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @("--legacy_weight_decay", "1e-7") },
  @{ Name = "bayes4090-wt-wd0"; Model = "bayes4090"; Variant = "wt"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @("--legacy_weight_decay", "0") },
  @{ Name = "bayes4090-wt-wd1e-7"; Model = "bayes4090"; Variant = "wt"; LoraRank = $LoraRanks[0]; RelaxationScale = $RelaxationScales[0]; Extra = @("--legacy_weight_decay", "1e-7") }
)

foreach ($Variant in @("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13")) {
  foreach ($Rank in (Get-RanksForVariant $Variant)) {
    foreach ($Scale in (Get-ScalesForVariant $Variant)) {
      $ScaleTag = Format-ScaleTag $Scale
      $Jobs += @{
        Name = "large4090-$Variant-r$Rank-$ScaleTag"
        Model = "large4090"
        Variant = $Variant
        LoraRank = $Rank
        RelaxationScale = $Scale
        Extra = @()
      }
    }
  }
}

if ($MaxJobs -gt 0) {
  $Jobs = @($Jobs | Select-Object -First $MaxJobs)
}

$Manifest = foreach ($Job in $Jobs) {
  [pscustomobject]@{
    name = $Job.Name
    model = $Job.Model
    variant = $Job.Variant
    lora_rank = $Job.LoraRank
    relaxation_scale = $Job.RelaxationScale
    extra = ($Job.Extra -join " ")
  }
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "$LogDir\manifest.json"

if ($DryRun) {
  $Manifest | Format-Table -AutoSize
  Write-Output "Dry run only. Manifest written to $LogDir\manifest.json"
  exit 0
}

foreach ($Job in $Jobs) {
  $ScaleText = $Job.RelaxationScale.ToString([Globalization.CultureInfo]::InvariantCulture)
  $OutDir = "$Root\runs\$($Job.Name)"
  $LogPath = "$LogDir\$($Job.Name).log"
  $ExtraArgs = @($Job.Extra)
  if ($null -ne $MaxEpochs) {
    $ExtraArgs += "--max_epochs"
    $ExtraArgs += "$MaxEpochs"
  }
  if ($null -ne $MaxTrainBatches) {
    $ExtraArgs += "--max_train_batches"
    $ExtraArgs += "$MaxTrainBatches"
  }
  if ($null -ne $MaxEvalBatches) {
    $ExtraArgs += "--max_eval_batches"
    $ExtraArgs += "$MaxEvalBatches"
  }
  $Started = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "[$Started] START $($Job.Name)" | Tee-Object -FilePath $LogPath -Append

  & $Python "$Root\repro_pytorch\train_ptb.py" `
    --data_path "$Root\data\ptb" `
    --model $Job.Model `
    --variant $Job.Variant `
    --lora_rank $Job.LoraRank `
    --relaxation_scale $ScaleText `
    --device $Device `
    --output_dir $OutDir `
    @ExtraArgs 2>&1 | Tee-Object -FilePath $LogPath -Append

  if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: $($Job.Name)"
  }

  $Finished = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "[$Finished] DONE $($Job.Name)" | Tee-Object -FilePath $LogPath -Append
}
