param(
  [string]$Python = "D:\DevTools\anaconda3\envs\wzdt\python.exe",
  [ValidateSet("baseline", "wt", "pr", "wt_pr", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13")]
  [string]$Variant = "wt",
  [ValidateSet("small", "medium", "large", "large4090", "bayes1500", "bayes4090")]
  [string]$Model = "large4090",
  [string]$Device = "auto",
  [Nullable[double]]$LegacyWeightDecay = $null,
  [int]$LoraRank = 8,
  [double]$RelaxationScale = 1.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScaleText = $RelaxationScale.ToString([Globalization.CultureInfo]::InvariantCulture)
$ScaleTag = "scale$($ScaleText.Replace('-', 'm').Replace('.', 'p'))"
$OutDir = "$Root\runs\$Model-$Variant-$ScaleTag"
$ExtraArgs = @()
if ($null -ne $LegacyWeightDecay) {
  $ExtraArgs += "--legacy_weight_decay"
  $ExtraArgs += "$LegacyWeightDecay"
}

& $Python "$Root\repro_pytorch\train_ptb.py" `
  --data_path "$Root\data\ptb" `
  --model $Model `
  --variant $Variant `
  --lora_rank $LoraRank `
  --relaxation_scale $ScaleText `
  --device $Device `
  --output_dir $OutDir `
  @ExtraArgs
