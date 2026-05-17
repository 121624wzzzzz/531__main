@echo off
pushd "%~dp0.."
"C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -File "%~dp0scheduled_all_experiments.ps1" -LoraRanks 1 2 4 8 16 32 -RelaxationScales 0.1 0.3 0.5 0.7 1.0 1.5 2.0 -Device auto
popd
