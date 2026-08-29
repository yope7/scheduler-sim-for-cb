# アンカー残差方策 (anchor-residual policy) 実装設計書

対象: `/home/noguchi/scheduler-sim-for-cb`（2026-06-13, Planエージェント調査に基づく）

## 0. 目的と要約

trace256 で方策が深い前線に到達できない問題（クリーン到達 67.7% vs NSGA-II 100%）に対し、行動空間を「ゼロから配置」から「参照アンカー解からの逸脱」へ再定義する。

- 方策出力: `a ∈ {0=follow(アンカーに従う), 1=flip(反転)}`
- env に渡す実行行動: `abs = anchor[job_idx] XOR a`（**env は無改変**、XOR は Actor/eval 側で適用）
- アンカー: 指令 `(cost, wait)` に正規化距離で最近傍の NSGA-II PF 点の遺伝子（`results/eval_pf/nsga2_trace{NJ}_s0.npz` の `chromosomes`）。エピソード開始時に一度だけ決定し固定。
- ゲート: 環境変数 `PCN_ANCHOR_RESIDUAL=<npzパス>`（未設定時は全経路ビット一致）
- 到達はアンカーが構造的に保証し、RL はアンカー間の補間と局所改善（疎な flip）だけを学ぶ。

本設計の核は **「事後リアンカー (post-hoc re-anchoring)」**: アーカイブに格納する全エピソードの action を、エピソード終了後に「達成値から選んだアンカー」基準の残差へ書き換える。γ=1.0（`src/agents/pcn_agent.py:1096`）のため PCN の relabel 指令（t=0 の残り return）＝達成 return が厳密に一致し、「学習時に NN が見る指令→アンカー」の対応が eval 時の「指令→アンカー」と完全に同じ関数になる。

## 1. 現状コードの事実確認（調査結果）

1. **学習バッチは `transition.action` をそのまま教師にする**。`pcn_agent.py:1848`（`get_training_batch`）、`:1979`（`_encode_episode_training_block`）。指令は relabel された残り return（`:1850-1880`, `:1987-1992`）。γ=1.0 なので t=0 の relabel 指令＝エピソード総 return＝`[-total_wait, -cost]`。→ **格納 action を残差にすれば Phase2/3 の学習コードは無変更**。
2. **Actor の rollout は `distributed_pcn.py:1238` `Actor._run_episode`**。`fixed_actions` 再生は `:1280`（行動取得）と `:1317-1318`（`scheduled` 時のみ `_fa_idx += 1`、`nsga2_agent._rollout`(`:59-71`) と同一規約）。指令は `:1261-1263` 設定、`:1331-1335` で残り return/horizon 更新。終了時 `:1342-1343` で `calc_objective_values()` を必ず計算、`:1354` で `transitions[0].objective_values` 付与。
3. **eval 系は全て `pcn_agent.py:3628` `PCN._run_episode` に合流**: `eval_b2_compare.py:78,162,165`、学習中 eval の `Actor.evaluate_episode`(`:1413`)、`eval_uniform_grid_batch`(`:1453`)、`Learner.evaluate`(`:2212`)。env.step は `:3644`。→ **eval 側のゲートは `PCN._run_episode` 1箇所で全経路をカバー**。
4. **PHASE1_NSGA 種まきブロックは `distributed_pcn.py:3729-3783`**（重複除去・cost昇順・K間引き `:3741-3750`、ε摂動 `:3754-3765`、`actor.run(..., fixed_action_seqs=_seqs)` `:3769-3772`）。
5. **アンカー npz**: `pf` (200,2)=(cost, avg_wait) cost昇順・重複行あり、`chromosomes` (200, NJ) int8。座標系は eval と同一（`verify_nsga2_eval_equiv.py` 照合済み）。
6. **訓練インスタンス＝アンカーインスタンス**: `Actor._make_env` は job_seed=0 固定（`:977-982`）、`create_eval_env(config, job_seed=0)` と同一。s0 アンカーは訓練・SEEDS=0 eval の両方で厳密に有効。SEEDS≠0 では無効。
7. 指令↔値の変換: `objectives_to_command(cost, wait, nj) = [-wait*nj, -cost]`（`src/utils/pf_command_eval.py:13-15`）。
8. イベント native env に `n_jobs` 属性は無い。nj はアンカー npz の遺伝子幅から取るのが最も安全。
9. 既存の `PCN_ANCHOR_KL_WEIGHT` は方策スナップショット正則化で**本件と無関係**。名前衝突に注意し `PCN_ANCHOR_RESIDUAL` / `PCN_ANCHOR_OBS` を使う。

## 2. アンカー選択: 指令 → アンカー遺伝子のマッピング

### 2.1 新規モジュール `src/utils/anchor_residual.py`

```python
class AnchorSet:
    # 構築時(npz 1回ロード, プロセス毎 lazy singleton):
    #  - chromosomes を行単位で重複除去(np.unique, 先勝ち) → cost 昇順ソート(PHASE1_NSGA :3741-3746 と同規約)
    #  - 正規化定数を npz 自身から固定: c_lo,c_hi = pf[:,0].min/max, w_lo,w_hi = pf[:,1].min/max
    #  - nj = chromosomes.shape[1]
    def select(self, desired_return):
        cost = -float(desired_return[1]); wait = -float(desired_return[0]) / self.nj
        cn = (cost - c_lo) / max(c_hi - c_lo, eps); wn = (wait - w_lo) / max(w_hi - w_lo, eps)
        d2 = (pf_cn - cn)**2 + (pf_wn - wn)**2
        i = int(np.argmin(d2))   # argmin=最小index で決定的タイブレーク
        return i, self.genes[i]
    def select_by_values(self, cost, avg_wait): ...  # 達成値から直接選ぶ(事後リアンカー用)

_AR = None
def get_anchor_set():  # PCN_ANCHOR_RESIDUAL 未設定なら None
```

設計判断:
- **正規化定数は npz 由来で固定**（実行時データから作らない）→ Actor / Learner-eval / eval スクリプトの3経路で選択が完全一致（ずるなし: eval 時も指令のみから決定論的に選択）。
- **エピソード開始時の desired_return で一度だけ選択し固定**（途中再選択はフラッピングで整合が壊れる）。
- γ=1.0 なので「relabel 指令→select」と「達成値→select_by_values」が厳密に同じ点を返す。

### 2.2 3経路での一貫性

| 経路 | 選択タイミング | 入力 |
|---|---|---|
| Actor(Phase3 生成) | `Actor._run_episode` の指令設定直後(`:1263`付近) | episode の `desired_return`(初期値) |
| 格納(全Phase) | エピソード終了後の事後リアンカー(§4) | 達成値 `(cost, avg_wait)` |
| eval(全経路) | `PCN._run_episode` 冒頭(`:3634`直後) | 引数 `desired_return`(初期値) |

学習時に NN が見る t=0 指令は relabel された達成 return なので、「格納時のアンカー＝達成値最近傍」と「eval 時のアンカー＝指令最近傍」が同一関数で結ばれる。これが一貫性の根拠。

## 3. 行動の意味の切替（XOR）とゲート

### 3.1 env 無改変（採用）
- env は NSGA-II / ヒューリスティック / ランダム基準線と共有。env 内に anchor 状態を持つと検証体系（`verify_nsga2_eval_equiv.py` 等）が崩れる。
- 必要な状態は「scheduled 済みジョブ数」だけ＝呼び出し側ループに既にある規約（`_fa_idx`）。

### 3.2 規約
- `job_idx` は **`scheduled=True` の時のみ +1**（`nsga2_agent._rollout` と同一）。非 scheduled ステップは同ビット再参照。
- `job_idx >= len(gene)` は anchor_bit=0 フォールバック（既存 `:1280` と同じ）。
- 格納する `transition.action` は**残差**（§4 の事後リアンカー後の値）。
- ゲート `PCN_ANCHOR_RESIDUAL`（npz パス）はモジュールレベルで読む（Ray worker は env 継承）。

## 4. 残差ラベルの一貫性: 事後リアンカー（本設計の核）

残差モード時、アーカイブへ格納する**全エピソード**（policy / random / heuristic / fixed_actions）の action を、終了後に達成値基準で書き換える:

```
anchor_ach = select_by_values(cost, avg_wait)      # :1343 calc_objective_values の直後
for t in transitions:
    t.action = abs_action[t] XOR anchor_ach.gene[job_idx[t]]
```

rollout 中に per-transition で `(abs_action, job_idx)` をローカル配列に記録する。

帰結:
1. **Phase1 NSGA 種まきは「全 follow」（残差0）に自動的になる**: 遺伝子 g_k を絶対行動で再生→達成値=その PF 点→`anchor(達成値)=g_k`→残差 = g_k XOR g_k = 0。**PHASE1_NSGA ブロックと `Actor.run` のシグネチャは無変更**。
2. **ε摂動種まきは疎な flip ラベルになる**（「どこで逸脱すると PF 上でどう動くか」の教師がそのまま得られる）。
3. **Phase1 ランダム/wtth も一貫した残差表現になる**＝絶対と残差の混在によるアーカイブ汚染を構造的に防ぐ。
4. Phase3 の policy エピソードは生成時アンカー（指令由来）と格納時アンカー（達成値由来）が違い得るが、PCN は relabel 教師あり学習なので健全（log-prob 再利用は無い）。

## 5. 観測拡張の要否

- **Stage A（拡張なし・先行）**: 残差ラベルの大半は 0。アンカーは episode 初期 dr の決定的関数で NN は部分識別可能。state_dim 不変＝既存 ckpt から warm start 可、OFF 時の構造差分ゼロ。
- **Stage B（`PCN_ANCHOR_OBS=1`）**: アンカー境界帯では同一状態・近接指令でラベル衝突→現ジョブのアンカービット(+1次元)で曖昧性を消す。実装コスト: `_obs_for_policy`(`:3561`)・Actor 方策分岐(`:1310`)・学習側 (`:1717`, `:1838-1847`, `:1974-1978`)・state_dim+1 の伝播 (`:1014-1016`, `eval_b2_compare.py:56,142`)。
- **推奨**: Stage A → §9 Step3 の境界診断で必要性が示されたら Stage B。

## 6. Phase1/2/3 への影響まとめ

| Phase | 変更 |
|---|---|
| Phase1 種まき/ランダム/wtth | コード変更なし（事後リアンカーが自動残差化） |
| Phase2 教師あり | **無変更**（`transition.action` が既に残差） |
| Phase3 生成 | アンカー選択+XOR+事後リアンカー（Actor._run_episode のみ） |
| Phase3 学習・指令選択・報酬 | **無変更**（指令空間は不変） |
| 学習中 eval | `PCN._run_episode` のゲートで自動対応 |

注意: **1 run の途中でゲートを切り替えない**（アーカイブ表現が混在）。ckpt メタに `anchor_residual: <npzパス|''>` を記録し eval 側で不一致警告（`save_model :3034`）。

## 7. eval_b2_compare.py の対応

ゲートが `PCN._run_episode` 内にあるため**ロジック変更ほぼ不要**。
- `:78/:162/:165`（greedy/samp）: 無変更でゲートが効く。
- `:69-71/:150-155` rp 掃引: **意図的に絶対行動のまま**（共通真PF参照の意味を変えない）。
- 追加: ①冒頭で npz ロード+`chromosomes.shape[1]==NJ` assert ②`out["anchor_npz"]` メタ保存 ③SEEDS≠0 警告。
- NPROC>1 fork worker は env 継承で per-process lazy ロード可。`_winit` 変更不要。

## 8. 編集ポイント一覧

新規: **`src/utils/anchor_residual.py`**（AnchorSet・get_anchor_set）、`tests/test_anchor_residual.py`

変更:
1. `src/agents/pcn_agent.py`: `:32-192`付近フラグ追加 / `:3628-3711 _run_episode`（reset直後select、`:3643-3644`間でXOR、scheduled で job_idx+1、Transition には残差格納）
2. `src/distributed/distributed_pcn.py`: `:1238-1357 Actor._run_episode`（`:1244` job_idx一般化※OFF時挙動不変、`:1263`直後select、`:1308-1313` XOR、abs_actions/job_idxs 記録、`:1343`直後に事後リアンカー）
3. `scripts/eval_b2_compare.py`: NJ整合assert・メタ保存・SEEDS警告

## 9. 段階的実装計画

- **Step 0** モジュール+単体検証: 選択の決定性 / `select(dr)==select_by_values` 往復 / **残差0再生の同値性**（全followダミー方策の達成値が `_rollout` の PF 値と一致、128 s0 全200遺伝子）
- **Step 1** 128スクリーン: 短縮学習で follow 診断・アーカイブPFがアンカーPF包含・baseline以上 + **OFF時ビット一致検証**
- **Step 2** 256本番: 比較=クリーン67.7 / 種まき+4.2 / NSGA-II 100。診断=greedy flip率分布（アンカー近傍≈0・アンカー間で疎なflip）。補助baseline=「残差常時0」rollout＝純アンカー再生PFでRLの上積みを分離計測
- **Step 3** 境界・過適合診断: NCMD密(200)でアンカー切替境界の不連続を計測→混乱あればStage B
- **Step 4** unseen-seed検証: SEEDS=1,2（アンカーは s0 のまま）で「丸写し度」定量化。必要なら s1/s2 用 NSGA npz（`JOB_SEED` 変更）で「インスタンス毎アンカー差し替え」上限も測る

## 10. OFF 時ビット一致の担保

- 全変更を `if _AR is not None:` 分岐に閉じ、OFF 時は既存行を一切通らない。
- 検証: パッチ前後で同一 ckpt の `eval_b2_compare.py`（NPROC=1, greedy）npz 全配列 `np.array_equal` 照合 + 短NITER train.log loss系列ハッシュ比較。
- `job_idx` 一般化が OFF 時に `_fa_idx` 挙動を変えない（増分条件に `fixed_actions is not None` を残す）。

## 11. リスクと対策

1. **アンカー境界の不連続**: エピソード内固定選択 / Step3 診断 / Stage B 観測 / 最悪はアンカーK間引きで境界密度低減
2. **アンカー丸写し（汎化喪失）**: flip率・samp分散で検出。ε摂動種まきを残差教師として活用
3. **unseen-seed 不成立**: 「このトレースの解の系統を学ぶ手法」と主張範囲を明確化、Step4 で定量化
4. **表現混在事故**: ckpt メタ記録 + eval 警告
5. **PF範囲外の達成値**: 正規化は npz 固定なので select は端点へ飽和するだけ＝破綻しない
