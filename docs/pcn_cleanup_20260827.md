# PCN コードの整理整頓（2026-08-27）

基準コミット cf4125d。**学習・評価の挙動は変えていない**（消したのは実行されないコードだけ）。
判断が要るもの・実験レシピが変わるものは手を付けず、末尾に列挙した。

## 消したもの（すべて参照ゼロを確認、約1440行）

| 対象 | 場所 | 行数 |
|---|---|---|
| `distributed_pcn_multiprocessing.py` 丸ごと | src/distributed/ | 911 |
| `visualize_initial_pareto_front` | distributed_pcn.py | 191 |
| `Learner.evaluate_distributed`（実路は `_distributed_evaluate_episodes`） | distributed_pcn.py | 64 |
| `Learner.update_eval_gap_feedback`（実路は `_driver_eval_gap_feedback`） | distributed_pcn.py | 30 |
| `CNN1D` / `CNNBackedPCN` | pcn_agent.py | 81 |
| `initialize_buffer_with_heuristics` | pcn_agent.py | 84 |
| `check_overfitting` と中身が空の `_update_on_fixed_data` | pcn_agent.py | 29 |
| `plot_rewards` / `get_e_returns` / `get_transitions` / `get_mapmap` ほか小物 | pcn_agent.py | 41 |
| 未使用のモジュール定数 `_EVAL_GAP_FEEDBACK` / `_COMMAND_BALANCE_TARGET` | pcn_agent.py | 3 |
| 未使用 import（heapq, Union, crowding_distance, hypervolume ほか） | 両方 | 4 |
| `PCN_EVAL_PF_GRID` / `PCN_EVAL_STOCHASTIC` の setdefault（読み手が存在しない） | workload_pcn_profile.py, distributed_pcn_cli.py | 4 |

後ろ2件の補足:
- 削除した `execute_selected_policy` は `np.full(np.inf, np.inf, ...)` を呼んでおり、実行すれば必ず
  例外になる。一度も通っていない証拠。
- `PCN_EVAL_PF_GRID` / `PCN_EVAL_STOCHASTIC` は src に読み手が無い（flag_audit の DEAD そのもの）
  のに profile と cli が毎回 setdefault していたため、起動のたびに自分の台帳が自分に警告を出して
  いた。書き込み側だけ消し、台帳の DEAD 記載は残す（scripts 側にはまだ設定が残るため）。

## 「参照ゼロ」の判定に使った照合（grep 1本では足りない）

1. リポジトリ全体の識別子出現数から自己参照を引く（.py だけでなく .sh / .yml / .md も）
2. 基底クラスの契約 — `extract_env_info` は呼び出しゼロに見えるが `MOAgent.__init__` が
   `self.extract_env_info(env)` を呼ぶオーバーライド。**これで消していたら壊れていた**
3. `hasattr` / `getattr` の文字列ゲート 128 個との照合（消しても例外にならず黙って挙動が変わる経路）
4. Ray actor の `.remote()` 呼び出し名一覧との照合
5. 単語境界（`execute_selected_policy` は `evaluate_and_execute_selected_policy` の部分文字列）

## 手つかず（判断が要る）

### 1. `PCN_TRAIN_HEAD_STEP_WEIGHT` が no-op のまま

`_training_flat_step_weights()` の早期 return ガードに HEAD の条件だけが抜けていて、HEAD を
立てても常に None が返る。ガードに1条件足せば直る（実測でテストが通ることまで確認済み）。

**ただし直すと過去 run との比較が壊れる**: `run_j20000_c3.sh` / `run_j50000_gpu.sh` /
`run_jscale_c3.sh` と v10 系の `v9_env_export.sh` が `WEIGHT=20, FRAC=0.15` を設定している。
この設定で有効化すると先頭15%のステップが抽選質量の約8割を占め、一様抽選とは別レシピになる。
v10 smoke20a〜d はいずれも「HEAD 無効」で完走した結果なので、直したまま続けると
v10 の変更由来か HEAD 有効化由来かが切り分けられない。整理整頓とは別コミットで、
smoke を1本引き直してから入れること。台帳（flag_audit の BROKEN）に事情を追記した。

### 2. `EnhancedPCNModel` 経路（355行 + 分岐33箇所）

有効化は distributed_pcn.py の `USE_ENHANCED_MODEL = False` というハードコード定数のみで、
git 履歴上 True になったことがない。`.pt` 32個を調べても `EnhancedPCNModel` を含むものは
0件なので、チェックポイント互換の心配は現物として無い。消すなら、クラス削除・
`use_enhanced_model` 分岐の無条件化（補助損失のゲートを含むので機械置換は不可）・
`PCN.__init__` の kwarg 削除・それを渡す scripts 10本・`src/utils/pf_eval_gap.py:177` を
**1コミットで**やること。半端にやると診断スクリプトだけが後から壊れる。

### 3. 消してはいけないと分かったもの

- MPFT 系: `docs/dissertation/` が実施済みの否定的結果として結果表・abstract・future work に
  記載しており、`PCN_MPFT_GATED` は distributed_pcn.py:753 の生きた Ray 経路から
  `mpft_gate_update.remote()` を呼んでいる。
- 「scripts に grep で無い＝未使用」はこのリポジトリでは成立しない。フラグはアドホックな
  コマンドラインで立てられ、痕跡は成果物のファイル名と docs にしか残らないことがある。
- `src/envs/backup_bitmap/`: 名前に反して高速化主張の基準実装として docs から参照されている。

### 4. 別件（PCN 外）: `src/envs/job_generator.py` 807行が死にファイル

import しているコードがリポジトリ内に1つも無い（実路は全て
`src.utils.job_gen.job_generator`）。先頭60行が実路のファイルと完全一致する複製で、
しかも実路の方には現在未コミットの変更が乗っている＝**間違った方を編集する事故が起きる形**。
今回の PCN 整理の範囲外なので触っていない。

### 5. 構造的な問題: テストが 22 件しかない

src 全体 3万行に対してテスト4ファイル。「安全に消せる」根拠がテストではなく grep しかない。
distributed_pcn を1イテレーションだけ回して PF が出ることを見る smoke test を1本足す方が、
上の 2 を進めるより投資対効果が高い。
