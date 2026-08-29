#!/usr/bin/env bash
# numba CUDA(rawカーネル/lockstep)用の nvcc 12.2 断片をリポジトリ内に再構築する。
#
# 背景:
#   このマシンのドライバは CUDA 12.2、/usr/local/cuda は 12.5。numba がシステム側
#   libnvvm(12.5) で PTX を生成するとドライバがロードできず
#   CUDA_ERROR_UNSUPPORTED_PTX_VERSION で全 GPU rollout が落ちる。
#   そこで nvidia-cuda-nvcc-cu12==12.2.140 の wheel を展開し、CUDA_HOME をそこへ向ける。
#   (numba は CUDA_HOME/nvvm/lib64/libnvvm.so.4 と CUDA_HOME/nvvm/libdevice を見る。
#    wheel には libnvvm.so しか入らないので .so.4 の symlink をここで張る。)
#
# 生成物: tools/nvcc122/nvidia/cuda_nvcc/{nvvm/lib64/libnvvm.so.4, nvvm/libdevice, bin/ptxas}
#   → CUDA_HOME=<repo>/tools/nvcc122/nvidia/cuda_nvcc  (tools/cuda_env.sh が export する)
#
# usage:
#   tools/setup_nvcc122.sh            # 既に揃っていれば何もしない(冪等)
#   tools/setup_nvcc122.sh --force    # 作り直す
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/tools/nvcc122"
CUDA_HOME_DIR="$DEST/nvidia/cuda_nvcc"
PKG="nvidia-cuda-nvcc-cu12==12.2.140"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

ok() { [ -e "$CUDA_HOME_DIR/nvvm/lib64/libnvvm.so.4" ] \
    && [ -e "$CUDA_HOME_DIR/nvvm/libdevice/libdevice.10.bc" ] \
    && [ -e "$CUDA_HOME_DIR/bin/ptxas" ]; }

if ok && [ "$FORCE" -eq 0 ]; then
  echo "[setup_nvcc122] 既に構築済み: CUDA_HOME=$CUDA_HOME_DIR"
else
  [ "$FORCE" -eq 1 ] && rm -rf "$DEST"
  mkdir -p "$DEST"
  echo "[setup_nvcc122] $PKG を $DEST へ展開中 (要ネットワーク)..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --target "$DEST" "$PKG"
  else
    "$REPO/.venv/bin/python" -m pip install --target "$DEST" "$PKG"
  fi
  # wheel は libnvvm.so だけを置く。numba が探すのは libnvvm.so.4。
  ln -sfn libnvvm.so "$CUDA_HOME_DIR/nvvm/lib64/libnvvm.so.4"
  # wheel 展開直後の ptxas は実行属性が落ちていることがある
  chmod +x "$CUDA_HOME_DIR/bin/ptxas" "$CUDA_HOME_DIR/bin/"* 2>/dev/null || true
  ok || { echo "[setup_nvcc122] 失敗: $CUDA_HOME_DIR の中身が揃っていない" >&2; exit 1; }
  echo "[setup_nvcc122] 構築完了: CUDA_HOME=$CUDA_HOME_DIR"
fi

# --- 動作確認(numba が入っていれば小カーネルをコンパイル・実行) ---
PY="$REPO/.venv/bin/python"
if [ "${SKIP_VERIFY:-0}" = "1" ] || [ ! -x "$PY" ]; then
  echo "[setup_nvcc122] 動作確認はスキップ"
  exit 0
fi
CUDA_HOME="$CUDA_HOME_DIR" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" - <<'PY' || {
import numpy as np
from numba import cuda
from numba.cuda.cuda_paths import get_cuda_paths
print("[verify] nvvm =", get_cuda_paths()["nvvm"].info)

@cuda.jit
def _add1(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1.0

d = cuda.to_device(np.zeros(8, dtype=np.float32))
_add1[1, 32](d)
assert d.copy_to_host()[0] == 1.0
print("[verify] numba CUDA カーネルのコンパイル・実行 OK")
PY
  echo "[setup_nvcc122] 動作確認に失敗(GPUが使用中/未搭載の可能性)。CUDA_HOME 自体は構築済み" >&2
  exit 1
}
