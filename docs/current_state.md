# 引き継ぎ要約 — PCN スケジューラ PF 改善 / b2 (Fourier command encoding)

> 最終更新: 2026-06-10 / ブランチ `m2_f` / このファイルだけ読めば新セッションで作業を再開できる。
>
> **NSGA-II ベースライン + trace512 PF総括 (2026-06-10)**: [`docs/pf512_nsga2_report.html`](pf512_nsga2_report.html)。
> 既存 NSGA-II を妥当性チェック→修正（①bitmap env は trace で評価不能=600sタイムアウト→event native env 化、
> 一致検証3点 ALL OK=`scripts/verify_nsga2_eval_equiv.py` ②mut 0.1/bit→1/n で HV 98.5% vs 94.6% ③persistent pool+評価キャッシュ+seed固定+端個体）。
> trace512 seed0 で **HV 98.5% / PF 200点 / 402s** — **真PFの左下（平均待ち16,317 = all-cloud比 −59%）を新開拓**、
> 従来の真PF基準（rp∪greedy）は甘かった→**今後の共通真PFは NSGA-II PF を含めること**（`MUT=auto POP=200 GEN=150 NPROC=30 scripts/run_nsga2_trace512.py`）。
> 同時に trace512 全28グループ×5本を新基準で横断集計（`scripts/plot_pf_summary_nsga2.py` → `docs/figures/pf512_*.png`）:
> **最安定= nup200 (b2+N_UPDATES=200) 85.9%±6.2（唯一 min 79% で外れなし）**、spike (PCN_LOSS_SPIKE_SKIP) は全5本崩壊4.7%で棄却。
> 関連: 詳細な物語は [`docs/progress_report.html`](progress_report.html)（発表用）、深掘りは [`docs/trace1024_pcn_report.html`](trace1024_pcn_report.html)。
>
> **高速化 (2026-06-07)**: b2/128 の学習 update を 54.3→28.9 ms/update (1.88×, 結果ビット一致) に高速化。
> 環境変数 `PCN_FAST_UPDATE`（既定 ON）。真因＝CPUのカーネル発行律速(~1400 launch/update)で、条件付けKLの
> ペアごとPythonループのベクトル化＋Archive PFのupdate_many内memo＋診断metrics遅延。`=0`で旧挙動に復帰。
> 詳細: [`docs/pcn_b2_speedup_report_20260607.html`](pcn_b2_speedup_report_20260607.html)。検証: `scripts/bench_update_b2_128.py`。
> さらに **Eval/Actor ロールアウト**も高速化: `PCN_FAST_ENV`（ノード探索早期return＋過去イベント間引き＋job_queue部分更新＋no_grad）
> ＋ `PCN_FAST_ENV_SWEEP`（候補時刻ループ O(R²) を sweep-line 増分 occupancy で O(R log R + R·H) に。pure-Python event env が
> ジョブ数で超線形だった配置探索の指数を下げ、**倍率が規模で上昇: 128J 3.7×→1024J 8.2×**）。両者既定 ON、`=0` で旧挙動。
> 結果不変=配置スケジュール**ビット一致**（env 20セル＋NN eval＋b2/128実学習で崩壊なし・PF同等、学習全体 860.9→529.8s=1.62×）。
> 検証: `scripts/bench_env_alloc.py` `scripts/verify_env_alloc_equiv.sh` `scripts/gate_b2_128_sweep.sh`。実装: `src/envs/scheduling_variants/event_native_env.py`。

---

## 1. 目的

PCN（Pareto Conditioned Networks）スケジューラの **Pareto front (PF) 品質向上**。
クラウドバースティング向けに、ジョブを「オンプレ(0)/クラウド(1)」へ配置する多目的（**待ち時間 × コスト**）スケジューラを、1つの方策で「指令（desired return）通りの PF 上の任意点」に着地させたい。

直近の焦点 = **b2 (Fourier command encoding)**：条件付けベクトル（指令）を NeRF 流のフーリエ特徴に展開し、「指令ダイヤルの解像度」を上げて policy greedy を各インスタンスの真PFへより密着させる試み。**ジョブ数を変えた sweep で b2 のスケール依存性を解明 → 完了（16→1024 全7サイズ, 2026-06-07）**：損益分岐 32〜64、改善は山なり（64〜512ピーク・1024で逓減）、運用ルール「64以上ON・16〜32OFF」確定。

---

## 2. 現在の実装状況（要点）

### b2 は実装・検証済み
- `src/agents/pcn_agent.py` に実装済み。環境変数で ON/OFF：
  - `PCN_FOURIER_CMD=1` … b2 を有効化（**import 時に1度だけ読む module-global**）
  - `PCN_FOURIER_BANDS=4` … 周波数バンド数（既定4 → freqs `[1,2,4,8]`）
- smoke test OK。`cmd_in_dim = (reward_dim+1)*(1+2L)`（reward_dim=2, L=4 → 27）で film_gamma/film_beta/c_emb をサイズ。
- FiLM ゼロ初期化は維持 → b2 ON でも学習開始時は baseline と同一出力（安全スタート）。

### 結果（gap-to-true PF, 低いほど良い。seed0/共通真PF基準）

**全スケール sweep 完了（16→1024 の全7サイズ, seed0 共通真PF gap）：**

| n_jobs | baseline(film) | b2(fourier) | 判定 |
|--------|------|------|------|
| **16** | 0.058 | 0.092 | b2 **悪化** |
| **32** | 0.030 | 0.064 | b2 **悪化** |
| **64** | 0.054 | 0.012 | b2 **改善(−78%)** |
| **128**（seed0/SEEN） | 0.044 | 0.018 | b2 **改善(−59%)** |
| **256** | 0.026 | 0.019 | b2 **改善(−27%)** |
| **512** | 0.024 | 0.007 | b2 **改善(−71%)** |
| **1024** | 0.013 | 0.011 | b2 **改善(−15%)** |
| （参考）128 5インスタンス平均 | 0.030 | 0.018（−40%） | b2 改善 |
| （参考）128 5インスタンス最悪 | 0.044 | 0.024（−45%） | b2 改善 |

- **64〜1024 ジョブ**：b2 が一貫して改善・**害なし**（b2の絶対gapは常に 0.007〜0.019）。改善幅は**山なり（逆U字）**：64 −78% → 128 −59% → 256 −27% → 512 −71% → 1024 −15%。**旨味のピークは 64〜512 の中規模帯**。
- **超大規模(1024)で相対改善が縮む理由**：baseline 自体が全スケール最良の 0.013 に到達＝伸びしろが尽きた（b2は依然 0.011 で勝つが僅差）。＝「超大規模は素のダイヤルでほぼ十分」。
- **16/32 ジョブ**：b2 が**裏目**（分散の壁）。→ **損益分岐は 32〜64 の間（64 で一気に転換）で確定**。128 は5インスタンス（1 seen + 4 unseen）でも系統的。探索も無傷（discovered PF 173点, loss 0.34）。
- **運用ルール（確定）：b2 は 64以上で ON、16〜32 では OFF（生指令で十分）。**
- 図：`pf_b2_compare_{16,32,64,256,512,1024}_s0.png` / `pf_b2_compare.png`(unseen 128) / `pf_b2_compare_seen.png`(seen 128) / `pf_b2_summary.png`(5インスタンス集約) / `pf_b2_scale.png`(gap vs n_jobs 全7点横断) / `pf_b2_edges.png`(端ギャップ横断) / `pf_b2_cmdfollow.png`(指令追従の飽和)。すべて `docs/figures/` にもコピー済み。

### 端点喪失の原因究明（低ジョブ数で b2 が裏目になる根本）
ユーザ観察「効果ゼロどころか PF の端（全オンプレの安い角）が消える」を root-cause した。
- **核心の訂正：sin/cos は「ならす」ではなく「鋭くする」**（ユーザの当初直観は逆）。指令＝ダイヤル。生指令＝粗いが端を越えて回せる（線形外挿可）。sin/cos＝細かい目盛りだが塗ってある範囲の外に伸びない（周期的・有界＝外挿不能）。
- **2軸**：(A) 解像度＝b2勝ち（n=16 でも greedy 達成点 35 vs film 24）。(B) 端への外挿＝低ジョブ数で b2 負け（指令 cost=0 に対し film は 2445 まで下げるが b2 は 15066 で飽和。n=32: 25688 vs 57859。**64 で逆転**: 188910 vs 129531）。
- **なぜ低ジョブ数だけ**：小規模は §6 分散の壁で PF がスカスカ＆ギザギザ → 端到達に大きな外挿が必要 → 外挿に弱い Fourier が端で飽和し中央に縮こまる。大規模は端も密 → 補間で届く → b2 の鋭さが純粋に効く。
- **証拠スクリプト（npz のみ・GPU不要・OOM安全）**：`scripts/analyze_b2_edges.py`（端ギャップ横断）、`scripts/analyze_b2_command_follow.py`（指令列を rp linspace から再構成、greedy[i]↔cg[i]、指令 vs 達成コスト）。HTML §8 に「原因究明」節として記載済み。
- **直し方の示唆**：低ジョブ数では逆に低周波化／`PCN_FOURIER_BANDS` を下げる／生指令の線形成分を強く残す／規模が出る所だけ b2 ON。次の A/B 候補＝バンド数掃引。

### スケール依存の解釈（§6「分散の壁」の裏取り）
b2 = 指令ダイヤルを**細かく**する道具。
- **大規模(128)**：真PFが滑らか・安定（決定が多く平均化＝ブレ小）→ 細かいダイヤルが「狙った場所にピタッと止める」のに効く。
- **小規模(16/32)**：真PF自体が高分散でギザギザ（決定が少なく平均化されない）→ 細かいダイヤルが**ノイズまで追って行き過ぎる**（spectral bias の裏返し）。

→ 運用ルール：**b2 は規模が大きい時 ON、小さい時 OFF（生の指令で十分）**。同じ分散の壁が、PFの点数（§6）でも追従精度（b2）でも顔を出す＝一貫した物理像。

---

## 3. 変更済み / 新規ファイル

### コア実装
- `src/agents/pcn_agent.py` … b2 本体。`_FOURIER_CMD` / `_FOURIER_BANDS`（module-global）、`BasePCNModel.__init__` で `cmd_in_dim` 算出＋`fourier_freqs` buffer、`_encode_cmd(c)` メソッド、`forward`/`predict_archive_value` で `_encode_cmd` 適用、Discrete/Continuous の c_emb/film_gamma/film_beta を `nn.Linear(self.cmd_in_dim, ...)` に変更。

### 評価・作図スクリプト（新規）
- `scripts/eval_b2_compare.py` … **1方策**を SEEDS で走らせ npz 保存（`greedy_{sd}` / `samp_{sd}`=best-of-k / `rp_{sd}`=random-p 掃引）。`ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=...)`。`_FOURIER_CMD` が global なので **film と fourier は別プロセスで実行必須**。
- `scripts/plot_b2_compare.py` … film/fourier npz を読み、**共通真PF = 非支配(random-p ∪ 両方策の best-of-k)** を引いて gap 比較図。**タイトルに判定語(no improvement 等)を入れない**（ユーザ要望）。`TAG` env で suptitle に注記。
- `scripts/plot_b2_summary.py` … 5インスタンス集約（左=baseline順の棒、右=baseline gap vs b2 gap 散布）。AVG/WORST を print。
- `scripts/plot_b2_scale.py` … **gap vs n_jobs 横断サマリ**（新規）。各サイズの seed0 共通真PF gap を baseline/b2 で折れ線、b2優劣を緑/赤帯で塗る。**128 だけ npz 無印**（`truepf_film_s0.npz`）、他は `truepf_film_{NJ}_s0.npz`。存在するサイズだけ使う → 64+ が揃ったら再実行で自動更新。
- `scripts/analyze_b2_edges.py` … **端点喪失の定量化**（新規）。greedy が真PFの両端（低コスト角／高wait角）にどれだけ届かないかを正規化し横断図に。npz のみ・GPU不要。
- `scripts/analyze_b2_command_follow.py` … **指令追従の飽和**（新規・メカニズム決定打）。npz から指令コスト列を再構成（`cg=linspace(rp.min,rp.max,NCMD)`、`greedy[i]↔cg[i]`）し、指令 vs 達成コストを描く。低ジョブ数で b2 が端で飽和（外挿不能）を可視化。GPU不要。
- `scripts/run_b2_jobsize.sh` … **オーケストレータ**。`usage: run_b2_jobsize.sh JOBS [NITER]`。baseline(film)→b2(fourier) を同条件で新規学習し、seed0 で eval+plot まで一気通貫。学習は Ray+GPU（`CUDA_VISIBLE_DEVICES` 未設定）、eval は単体プロセスで GPU1（`CUDA_VISIBLE_DEVICES=1`）。

### ドキュメント
- `docs/progress_report.html` … 発表用の8章ナラティブ。§8 に b2（128の効果＋**スケール依存の新節**＋関連研究5本）。本ファイル更新で `pf_b2_scale.png` / `pf_b2_compare_16/32_s0.png` を追加済み。HTML 健全性 OK（div 85/85, figure 12/12, details 5/5）。
- `docs/index.html` … progress_report.html へのナビ＋カード追加済み。
- `index.html`（リポジトリ直下, gitignore対象）… progress_report.html への誘導リンク（ランディング）。

### メモリ（`~/.claude/.../memory/`）
- `b2-fourier-command-result.md`（5インスタンス結果）, `explain-plainly.md`, `no-overfitting-unseen-truepf.md`, `pcn-pf-point-count-jobcount.md`（分散の壁）, ほか。`MEMORY.md` がインデックス。

---

## 4. 重要な設計判断

1. **`_FOURIER_CMD` は module-global（import時読み込み）** → 同一プロセス内で film と fourier を切り替え不可。eval は必ず **別プロセス**で（`run_b2_jobsize.sh` はそうしている）。
2. **低周波 `[1,2,4,8]` を選ぶ理由**：探索ステップ（archive点の少し外側へ ~0.1）が最高周波数の周期（~0.78）より十分小さい → 指令空間で局所的に滑らか → PCN の bootstrap 探索（近傍指令を汎化で当てて新PF点にする仕組み）が壊れない。高周波すぎるとノイズを拾う（spectral bias）。
3. **FiLM ゼロ初期化を維持** → b2 ON の学習開始時 = baseline と同一（安全な漸進導入）。
4. **gap-to-true 指標**：共通真PF = 非支配(random-p 掃引 ∪ 両方策の best-of-k)。gap = mean( clip(greedy_wait − interp(greedy_cost, truePF_wait), 0) ) / truePF_wait_range。どちらの方策にも公平な緑線。
5. **図に判定語を書かない**（"no improvement" 等。ユーザ明示要望）。結果はポジティブに、かつ正直に記述。
6. **平易な説明を常に**（ユーザ要望 `explain-plainly`）：ダイヤル/ノブ比喩 → 2軸分解 → blockquote 結論。専門用語先行は NG（"むずすぎ"）。
7. **sweep は直列実行（OOM回避）**：学習中に別の学習を**起動しない**。1サイズ完了 → 次サイズ。

---

## 5. 未解決の問題

1. ~~64 / 256 / 512 / 1024 の sweep~~ → **完了（16→1024 全7サイズ）**。損益分岐 32〜64 確定、改善は山なり（64〜512がピーク、1024で逓減）。残課題は下記2〜5。
2. **256 の端点喪失の再発**：256 では overall 改善でも安い角が再び未達（指令0に film 40278 / b2 92367）＝固定周波数 [1,2,4,8] が大きな指令スケールと相互作用。512/1024 では解消。→ **`PCN_FOURIER_BANDS` を下げる／指令スケールで周波数を正規化する A/B** が次の検証候補（小規模の裏目と 256 端再発を同時に潰せるか）。
3. **③低コスト帯（左上, 全オンプレ寄り）の cost/wait 両立**は依然一部未達（progress_report §3 の正直な壁）。大規模ほど b2 が端に届く（1024: film 4.46M / b2 3.83M）ことは確認。
4. **実トレース・別サイズへの汎化**は未検証（合成 synthetic_urgency のみ）。
5. **plot_b2_scale.py は seed0 のみ**。複数 seed の平均/最悪も見るなら拡張が要る。

---

## 6. 次にやるべきこと（sweep は完了。次の検証候補）

> **b2 スケール sweep は 16→1024 全7サイズ完了**（2026-06-07 02:02、1024 eval 完走）。損益分岐 32〜64、改善は山なり（64〜512ピーク、1024 −15%に逓減）、運用ルール「64以上ON・16〜32OFF」確定。図・HTML §8・メモリ `b2-fourier-command-result.md` 反映済み。

### 次の A/B 候補（優先度順）
1. **`PCN_FOURIER_BANDS` 掃引（4→2,3 など）**：256 の端点再発（指令0に film 40278 / b2 92367）と小規模(16/32)の裏目を同時に潰せるか。低周波化＝外挿能力↑＝端到達↑の仮説検証。`PCN_FOURIER_BANDS=2 bash scripts/run_b2_jobsize.sh 256 100` の要領（命名衝突に注意：別 OUT/npz 名にするか別ディレクトリで）。
2. **指令スケールで周波数を正規化**：固定 [1,2,4,8] が大きな指令スケール（512/1024 は数百万オーダ）と相互作用するのを、指令レンジで割って吸収。
3. **複数 seed で平均/最悪 gap**（特に 256/512/1024）。`plot_b2_scale.py` は seed0 のみなので拡張が要る。
4. 実トレース / 別分布への汎化検証。

### sweep を再実行・追加する場合のルーチン（参考）
```bash
cd /home/noguchi/scheduler-sim-for-cb
bash scripts/run_b2_jobsize.sh ${NJ} 100   # film+fourier 学習→eval→plot を一括（直列厳守・OOM回避）
cp pf_b2_compare_${NJ}_s0.png docs/figures/
OUT=pf_b2_scale.png PYTHONPATH=. .venv/bin/python scripts/plot_b2_scale.py     # 存在する npz を自動で拾う
OUT=pf_b2_cmdfollow.png PYTHONPATH=. .venv/bin/python scripts/analyze_b2_command_follow.py
OUT=pf_b2_edges.png PYTHONPATH=. .venv/bin/python scripts/analyze_b2_edges.py
cp pf_b2_scale.png pf_b2_cmdfollow.png pf_b2_edges.png docs/figures/
```
- 各サイズ 学習 film+fourier @ iter100 ＋ eval。**OOM は杞憂**：1024 でも GPU ~30GB（env がCPU律速＝n_jobs に対しGPUメモリほぼ平坦）。ただし eval は重く 1024 で ~3.5h（film は安い角まで届く＝1エピソードのステップ数が増え fourier より遅い）。
- 完了ごとに図を SendUserFile でチャット掲示 → HTML §8 に追記。

---

## 7. よく使うパス / コマンド早見

- 学習レシピ：`scripts/run_synthetic_urgency.sh NAME JOBS [NITER]`。`PCN_FILM` / `PCN_FOURIER_CMD` は親 env から継承。`OUT` を `rm -rf` してから書くので再実行で上書き。`train.log` は `$OUT/train.log`（タイムスタンプ subdir の中ではない）。
- 学習出力：`experiments/distributed_pcn/run_synth${NJ}_${NAME}/<YYYYMMDD_HHMMSS>/iteration_${N}/model_iter_${N}.pth`。
- eval npz 命名：128 のみ無印 `truepf_{film,fourier}_s0.npz`、他は `truepf_{film,fourier}_${NJ}_s0.npz`。
- background タスク出力：`/tmp/claude-1002/-home-noguchi-scheduler-sim-for-cb/<session-uuid>/tasks/<task-id>.output`（session-uuid は現セッションのもの。前セッションは `cfc7d1d8-884d-4a3c-bbca-a7be6f929e47`）。
- 進捗確認：`nvidia-smi` で GPU、`ps -eo pid,etime,cmd | grep -E "python|ray::"` で学習プロセス、`ls experiments/distributed_pcn/run_synth${NJ}_fourier${NJ}/*/iteration_100/model_iter_100.pth` で学習完了。

---

## 8. 関連研究（b2 の裏付け、後で読む用）
- **Tancik et al. 2020**（Fourier Features, NeurIPS, arXiv:2006.10739）★本命 — 低次元入力を sin/cos に写すと MLP が高周波を学べる（NTK で理論化）。
- Vaswani 2017（Transformer, sinusoidal PE の発祥, arXiv:1706.03762）。
- Mildenhall 2020（NeRF, positional encoding γ(p), arXiv:2003.08934）— b2 の実装 `[c, sin(2ᵏc), cos(2ᵏc)]` と同型。
- Rahimi-Recht 2007（Random Fourier Features, 古典）。
- Rahaman 2019（spectral bias, arXiv:1806.08734）— 「生のダイヤルは粗い」問題の正体。
