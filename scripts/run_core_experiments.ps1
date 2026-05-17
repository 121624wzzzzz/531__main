param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Jobs = @(
  @{ Model = "small"; Variant = "baseline"; Extra = @() },
  @{ Model = "small"; Variant = "wt"; Extra = @() },
  @{ Model = "small"; Variant = "pr"; Extra = @() },
  @{ Model = "small"; Variant = "wt_pr"; Extra = @() },
  @{ Model = "large4090"; Variant = "baseline"; Extra = @() },
  @{ Model = "large4090"; Variant = "wt"; Extra = @() },
  @{ Model = "bayes4090"; Variant = "baseline"; Extra = @("--legacy_weight_decay", "1e-7") },
  @{ Model = "bayes4090"; Variant = "wt"; Extra = @("--legacy_weight_decay", "0") },
  @{ Model = "bayes4090"; Variant = "wt"; Extra = @("--legacy_weight_decay", "1e-7") }
)

foreach ($Job in $Jobs) {
  $DecaySuffix = ""
  if ($Job.Extra.Count -eq 2) {
    $DecaySuffix = "-wd$($Job.Extra[1])"
  }
  $OutDir = "$Root\runs\$($Job.Model)-$($Job.Variant)$DecaySuffix"
  $ExtraArgs = $Job.Extra
  & $Python "$Root\repro_pytorch\train_ptb.py" `
    --data_path "$Root\data\ptb" `
    --model $Job.Model `
    --variant $Job.Variant `
    --device $Device `
    --output_dir $OutDir `
    @ExtraArgs
}
