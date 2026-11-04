# Source this before any nnUNetv2_* command:   source nerveseg/nnunet/env.sh
# Points nnU-Net's three folders at this project (paths work from any current directory).
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
export nnUNet_raw="$_here/nnUNet_raw"
export nnUNet_preprocessed="$_here/nnUNet_preprocessed"
export nnUNet_results="$_here/nnUNet_results"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
echo "nnUNet_raw=$nnUNet_raw"
