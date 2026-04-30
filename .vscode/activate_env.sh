#!/bin/bash
# VSCode 终端自动激活脚本
if [ -n "$VSCODE_CONDA_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$VSCODE_CONDA_ENV" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate "$VSCODE_CONDA_ENV" 2>/dev/null
fi
