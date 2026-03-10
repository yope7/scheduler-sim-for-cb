# スパコンでの実行手順

このプロジェクトをスパコンに持っていって実行する手順です。

## 1. 持っていくもの

### 方法A: イメージをローカルでビルドして持っていく（推奨）

スパコンでビルド環境が制限されている場合、ローカルでビルドして転送します。

```
転送するファイル:
  - scheduler-sim.sif          # Singularity イメージ（数GB）
  - プロジェクト全体（または最低限）
    - config/
    - scripts/
    - src/
    - singularity/run.sh
```

### 方法B: ソースだけ持っていってスパコンでビルド

```bash
# 転送するもの（.sif 以外）
rsync -avz --exclude '*.sif' --exclude 'execution_*' --exclude '__pycache__' \
  scheduler-sim-for-cb/ u6c073@squidhpc.hpc.cmc.osaka-u.ac.jp:~/scheduler-sim-for-cb/
```

スパコンで `./singularity/build.sh` を実行してビルド。

---

## 2. 転送

### SQUID HPC（大阪大学 CMC）の場合

**ホーム（~/）は容量制限が厳しいため、ワーク領域に転送することを推奨:**

```bash
# ローカルPCで、プロジェクトの親ディレクトリにいる状態で実行

# ワーク領域へ転送（/sqfs/work/G15612/u6c228）
# .sif を除外して転送し、スパコンでビルドすることを推奨（イメージは数GBあるため）
rsync -avz --progress \
  --exclude '*.sif' \
  --exclude 'execution_*' \
  --exclude '__pycache__' \
  --exclude '*.egg-info' \
  --exclude '.git' \
  scheduler-sim-for-cb/ u6c073@squidhpc.hpc.cmc.osaka-u.ac.jp:/sqfs/work/G15612/u6c228/scheduler-sim-for-cb/

# 転送後、スパコンでビルド:
#   cd /sqfs/work/G15612/u6c228/scheduler-sim-for-cb && ./singularity/build.sh

# または scp（.sif を含めると容量オーバーの可能性あり）
scp -r scheduler-sim-for-cb u6c073@squidhpc.hpc.cmc.osaka-u.ac.jp:/sqfs/work/G15612/u6c228/
```

**ホームに転送する場合**（容量に注意）:
```bash
rsync -avz --progress scheduler-sim-for-cb/ u6c073@squidhpc.hpc.cmc.osaka-u.ac.jp:~/scheduler-sim-for-cb/
```

### 一般的な形式

```bash
rsync -avz --progress scheduler-sim-for-cb/ user@spcomp:/home/user/scheduler-sim-for-cb/
scp -r scheduler-sim-for-cb user@spcomp:/home/user/
```

---

## 3. スパコンでの実行

### ログイン

```bash
ssh u6c073@squidhpc.hpc.cmc.osaka-u.ac.jp
```

### インタラクティブ実行（ログインノード）

```bash
# ワーク領域に転送した場合
cd /sqfs/work/G15612/u6c228/scheduler-sim-for-cb

# sweep 実行（プロジェクトをマウントするのでコード変更が即反映）
SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh --quick

# 本番実行（--quick なし）
SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh -o results.json
```

### バッチジョブ（Slurm の例）

`singularity/job_slurm.sh` を参照するか、以下を参考にジョブスクリプトを作成:

```bash
#!/bin/bash
#SBATCH --job-name=scheduler-sweep
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd $SLURM_SUBMIT_DIR

export SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py
export DISTRIBUTED_PCN_OUTPUT_DIR=$SLURM_SUBMIT_DIR

./singularity/run.sh -o results_$(date +%Y%m%d_%H%M%S).json
```

投入:

```bash
sbatch singularity/job_slurm.sh
```

---

## 4. スパコン固有の確認事項

### Apptainer/Singularity の有無

```bash
which apptainer
# または
which singularity
```

なければ `module load apptainer` などで読み込む場合があります。

### ストレージ

- 出力先: ホーム領域かスクラッチ領域を指定
- イメージ: スクラッチの方が転送・読み込みが速い場合あり

```bash
# スクラッチに出力する例
export DISTRIBUTED_PCN_OUTPUT_DIR=/scratch/$USER/scheduler-results
SCHEDULER_SIM_SCRIPT=scripts/run_distributed_pcn_sweep.py ./singularity/run.sh
```

### GPU ノードを使う場合

ジョブスクリプトに `--gres=gpu:1` などを追加し、run.sh に `--nv` を渡す必要があります。  
その場合は `run.sh` の `$CMD run` に `--nv` を追加するか、環境変数で制御できるようにするのがよいです。

---

## 5. チェックリスト

- [ ] スパコンに Apptainer/Singularity が入っている
- [ ] プロジェクト（または .sif）を転送した
- [ ] `config/config.yml` の内容を確認（job_type, job_trace_path 等）
- [ ] 出力先ディレクトリの容量を確保
- [ ] バッチジョブのリソース（CPU, メモリ, 時間）を適切に設定
