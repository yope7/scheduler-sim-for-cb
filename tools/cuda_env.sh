# numba CUDA(rawカーネル/lockstep)用の CUDA_HOME を export する。
#   usage: source tools/cuda_env.sh   (未構築なら tools/setup_nvcc122.sh を先に実行)
# 詳細は tools/setup_nvcc122.sh の先頭コメントを参照。
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$_repo/tools/nvcc122/nvidia/cuda_nvcc"
if [ ! -e "$CUDA_HOME/nvvm/lib64/libnvvm.so.4" ]; then
  echo "[cuda_env] $CUDA_HOME が未構築です → tools/setup_nvcc122.sh を実行してください" >&2
fi
unset _repo
