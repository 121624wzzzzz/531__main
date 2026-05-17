param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto",
  [int]$LogEvery = 1000,
  [string]$RunTag = "-screen35ep",
  [switch]$MetricsOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Format-ScaleTag([double]$Scale) {
  $ScaleText = $Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
  return "scale$($ScaleText.Replace('-', 'm').Replace('.', 'p'))"
}

$Jobs = @(
  @{ Variant = "s1"; Rank = 8; Scale = 0.1 },
  @{ Variant = "s2"; Rank = 8; Scale = 0.1 },
  @{ Variant = "s4"; Rank = 8; Scale = 0.1 },
  @{ Variant = "s5"; Rank = 8; Scale = 0.3 },
  @{ Variant = "s4"; Rank = 8; Scale = 0.3 },
  @{ Variant = "s13"; Rank = 16; Scale = 0.1 },
  @{ Variant = "s12"; Rank = 4; Scale = 0.1 },
  @{ Variant = "s12"; Rank = 16; Scale = 0.1 }
)

foreach ($Job in $Jobs) {
  $ScaleText = $Job.Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
  $ScaleTag = Format-ScaleTag $Job.Scale
  $OutDir = "$Root\runs\large4090-$($Job.Variant)-r$($Job.Rank)-$ScaleTag$RunTag"
  $ExtraArgs = @("--max_epochs", "35", "--log_every", "$LogEvery")
  if ($MetricsOnly) {
    $ExtraArgs += "--metrics_only"
  }

  & $Python "$Root\repro_pytorch\train_ptb.py" `
    --data_path "$Root\data\ptb" `
    --model large4090 `
    --variant $Job.Variant `
    --lora_rank $Job.Rank `
    --relaxation_scale $ScaleText `
    --device $Device `
    --output_dir $OutDir `
    @ExtraArgs

  if ($LASTEXITCODE -ne 0) {
    throw "Final candidate run failed: variant=$($Job.Variant) rank=$($Job.Rank) scale=$ScaleText"
  }
}
