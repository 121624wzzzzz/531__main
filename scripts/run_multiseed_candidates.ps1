param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto",
  [int[]]$Seeds = @(1, 2, 3, 4, 5),
  [int]$LogEvery = 1000,
  [string]$RunTag = "-multiseed35",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Format-ScaleTag([double]$Scale) {
  $ScaleText = $Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
  return "scale$($ScaleText.Replace('-', 'm').Replace('.', 'p'))"
}

$Jobs = @(
  @{ Variant = "s1"; Rank = 8; Scale = 0.1 },
  @{ Variant = "s4"; Rank = 8; Scale = 0.1 },
  @{ Variant = "s5"; Rank = 8; Scale = 0.3 }
)

foreach ($Job in $Jobs) {
  foreach ($Seed in $Seeds) {
    $ScaleText = $Job.Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
    $ScaleTag = Format-ScaleTag $Job.Scale
    $OutDir = "$Root\runs\large4090-$($Job.Variant)-r$($Job.Rank)-$ScaleTag-seed$Seed$RunTag"
    $JsonPath = "$OutDir\ptb_large4090_zaremba_$($Job.Variant).json"

    if ((Test-Path $JsonPath) -and -not $Force) {
      Write-Output "Skipping completed run: variant=$($Job.Variant) rank=$($Job.Rank) scale=$ScaleText seed=$Seed"
      continue
    }

    & $Python "$Root\repro_pytorch\train_ptb.py" `
      --data_path "$Root\data\ptb" `
      --model large4090 `
      --variant $Job.Variant `
      --lora_rank $Job.Rank `
      --relaxation_scale $ScaleText `
      --seed $Seed `
      --device $Device `
      --log_every $LogEvery `
      --output_dir $OutDir

    if ($LASTEXITCODE -ne 0) {
      throw "Multiseed run failed: variant=$($Job.Variant) rank=$($Job.Rank) scale=$ScaleText seed=$Seed"
    }
  }
}
