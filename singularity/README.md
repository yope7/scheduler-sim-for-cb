# Singularity イメージ（distributed_pcn.py 用）

`distributed_pcn.py` を Singularity コンテナで別マシンで実行するための手順です。

## ビルド

プロジェクトルートで実行:

```bash
./singularity/build.sh
# または出力ファイル名を指定
./singularity/build.sh my-image.sif
```

生成された `scheduler-sim.sif` を別マシンにコピーして使用できます。

## 実行

### 基本的な実行（distributed_pcn）

```bash
./singularity/run.sh
```

結果は `DISTRIBUTED_PCN_OUTPUT_DIR`（デフォルト: プロジェクトルート）の `execution_YYYYMMDD_HHMMSS/` に保存されます。

### スクリプトを指定して実行

環境変数 `SCHEDULER_SIM_SCRIPT` で実行する Python スクリプトを指定できます。

```bash
# ジョブ数×ノード構成のスイープ（run_distributed_pcn_sweep.py）
SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh --quick

# 結果出力先を指定
SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh -o results.json
```

### 別マシンでの実行

1. `scheduler-sim.sif` をコピー
2. 必要に応じて `config/` と `job_trace/` をマウント

```bash
# 出力先を指定
export DISTRIBUTED_PCN_OUTPUT_DIR=/path/to/output
singularity exec -B /path/to/output:/output scheduler-sim.sif python -m src.distributed.distributed_pcn

# 設定ファイルを差し替える場合
singularity exec \
  -B /path/to/output:/output \
  -B /path/to/myconfig.yml:/app/config/config.yml:ro \
  scheduler-sim.sif python -m src.distributed.distributed_pcn

# job_trace をホストからマウント（job_type=2 の場合）
singularity exec \
  -B /path/to/output:/output \
  -B /path/to/job_trace:/app/job_trace:ro \
  scheduler-sim.sif python -m src.distributed.distributed_pcn
```

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|------------|
| `SCHEDULER_SIM_SCRIPT` | 実行するPythonスクリプト（例: `scripts/run_distributed_pcn_sweep.py`） | （未設定時は distributed_pcn） |
| `DISTRIBUTED_PCN_CONFIG` | 設定ファイルパス | `config/config.yml` |
| `DISTRIBUTED_PCN_OUTPUT_DIR` | 出力ディレクトリ | `.` |
| `DISTRIBUTED_PCN_WORKDIR` | 作業ディレクトリ | （未設定） |
| `DISTRIBUTED_PCN_QUICK` | `1` でクイックモード | `0` |
| `DISTRIBUTED_PCN_FAST` | `1` で高速モード | `0` |

### GPU 使用

CUDA が利用可能な環境では、`--nv` オプションで GPU を有効にします:

```bash
singularity exec --nv -B /path/to/output:/output scheduler-sim.sif python -m src.distributed.distributed_pcn
```

## スパコンでの実行

スパコンに持っていって実行する手順は [SPCOMPUTER.md](SPCOMPUTER.md) を参照してください。

## 注意事項

- **job_type=2**（ジョブトレースファイル使用）の場合、`config/config.yml` の `job_trace_path` が指すファイルがイメージ内に存在するか、実行時に `-B` でマウントする必要があります。
- **job_type=1**（デフォルトジョブ）の場合は `job_trace` は不要です。
- ビルド時に `config/` をイメージに含めています。別の設定を使う場合は `-B` で上書きしてください。
