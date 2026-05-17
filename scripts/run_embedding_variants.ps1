param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [ValidateSet("small", "medium", "large", "large4090", "bayes1500", "bayes4090")]
  [string]$Model = "large4090",
  [string]$Device = "auto",
  [Alias("LoraRank")]
  [int[]]$LoraRanks = @(1, 2, 4, 8, 16, 32),
  [Alias("RelaxationScale")]
  [double[]]$RelaxationScales = @(0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
  [string[]]$Variants = @("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13"),
  [Nullable[int]]$MaxEpochs = $null,
  [Nullable[int]]$MaxTrainBatches = $null,
  [Nullable[int]]$MaxEvalBatches = $null,
  [int]$LogEvery = 1000,
  [string]$RunTag = "",
  [switch]$MetricsOnly,
  [int]$SkipJobs = 0,
  [int]$MaxJobs = 0,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

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

$Jobs = @()
foreach ($Variant in $Variants) {
  foreach ($Rank in (Get-RanksForVariant $Variant)) {
    foreach ($Scale in (Get-ScalesForVariant $Variant)) {
      $ScaleText = $Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
      $ScaleTag = Format-ScaleTag $Scale
      $Jobs += [pscustomobject]@{
        Model = $Model
        Variant = $Variant
        LoraRank = $Rank
        RelaxationScale = $Scale
        RelaxationScaleText = $ScaleText
        OutDir = "$Root\runs\$Model-$Variant-r$Rank-$ScaleTag$RunTag"
      }
    }
  }
}

if ($SkipJobs -gt 0) {
  $Jobs = @($Jobs | Select-Object -Skip $SkipJobs)
}

if ($MaxJobs -gt 0) {
  $Jobs = @($Jobs | Select-Object -First $MaxJobs)
}

if ($DryRun) {
  $Jobs | Format-Table -AutoSize
  Write-Output "Dry run only. Planned jobs: $($Jobs.Count)"
  exit 0
}

foreach ($Job in $Jobs) {
  $ExtraArgs = @("--log_every", "$LogEvery")
  if ($MetricsOnly) {
    $ExtraArgs += "--metrics_only"
  }
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

  & $Python "$Root\repro_pytorch\train_ptb.py" `
    --data_path "$Root\data\ptb" `
    --model $Job.Model `
    --variant $Job.Variant `
    --lora_rank $Job.LoraRank `
    --relaxation_scale $Job.RelaxationScaleText `
    --device $Device `
    --output_dir $Job.OutDir `
    @ExtraArgs
}
