# v8設計: GPU2枚フル・全工程GPU前提の5万ジョブオンラインPCN学習 (2026-08-24)

前提(確定事項): refit廃止(REFIT_EVERY=0、オンライン=毎iter ingest+update)。v7スモークで経路健全性は確認済み(Phase2復活・iter損失非ゼロ・checkpoint保存・正常終了)。

## 0. 今日の調査で確定した配管の事実

| 事実 | 根拠 |
|---|---|
| 非同期オーバーラップは**オンラインでは既定ON**: `learner.learn(N_UPDATES)`と次iterのrollout波が並走する設計が既にある | distributed_pcn.py:5205- (`_ASYNC_OVERLAP and REFIT_EVERY==0`)。つまり**GPU0(更新)∥GPU1(生産)の2枚並走は構造として既製** |
| Phase3を工場に回すと、iterの全指令が**1回のrun_commands=1チャンク**に束なる | _phase3_actor_wave (5155-5183) |
| lockstepはobsを**GPU上で構築・記録済み=CPUリプレイ不要** | gpu_factory.py:269-273 コメント(実装済) |
| lockstepの1チャンクは**T=50000の逐次が律速でB非依存**(枚数でもBでも割れない) | 08-22〜24実測+コード構造 |
| ★**11.4分は「B=64・軽い指令混合」のベンチ値**。実運用の重い指令では**全iter定額19.8分**(08-23記録: 「CPU側の重い指令が毎iter約19分の律速。ベンチ11.4分は軽い指令混合の数字。装備は正しいが配員計画が旧時代のまま」) | docs/diary/2026-08-23.html |
| 床の余地: 19.8分/iter = **23.8ms/step**。rawカーネルは同じ5万で**7.8ms/step**(B=64, 390s)。差の16ms/stepがobsカーネル+NNの取り分=**lockstep高速化の伸び代3倍** | 実測から算出 |
| Phase1(rawカーネル)のobsだけがCPU決定論リプレイ(1026s=Phase1の48%)。lockstepに行動列固定モードは**未実装**(greedy/sampleのみ) | lockstep_nn.py:95-136 |
| 工場ワーカは1体設計(`ray.remote(num_gpus=N)`)。2体化はドライバ側の小改修 | gpu_factory.py:198, distributed_pcn.py:4206-4218 |
| CHEAP_TO_ACTOR はP3波・評価の両方で cheap端をCPUへ逃がす(そしてPhase1では56本を黙って捨てるバグ) | 5165-5183 ほか。rawカーネルは厳密+溢れ自動拡大なので**この保険自体が不要** |
| warmupの724秒はrawでは使わないXLA空回し | 監査確定(独立ベンチ2.5本/分×32本=768sと一致) |

## 1. v8アーキテクチャ(工程別デバイス割当)

| 工程 | GPU0 (Learner) | GPU1 (工場) | CPU |
|---|---|---|---|
| Phase1 生産 | (v8bでは半分を分担) | rawカーネル or lockstep固定行動 | v8aのみ: obsリプレイ(48並列)→v8bで廃止 |
| Phase2 教師あり | 更新2,000回(≈37分)※epochs=20時 | — | v8aのみ: 初期cache構築→v8bでGPU化 |
| 学習ループ | update×100/iter + cache増分 | **lockstep rollout(1チャンク/iter)** — 更新と**並走**(既製のasync overlap) | 投入のみ |
| 評価 | (v8bで分担) | lockstep greedy一括(全量工場・cheap委譲なし) | — |

## 2. v8a — 今日できる構成(env+2行の微小変更)

```bash
REFIT_EVERY=0                          # オンライン(確定事項)
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=20   # Phase2復活(必須。プロファイルが0を焼き込む)
PCN_GPU_FACTORY_P3=1 PCN_GPU_RAW_P3=1  # ★Phase3をGPU1のlockstepへ(全GPU化の本丸)
PCN_GPU_FACTORY_CHEAP_TO_ACTOR=0       # ★データ29%捨てバグ根治+CPU委譲全廃
PCN_GPU_RAW_E_MAX=65536                # cheap端(混雑側)を工場で受けるためのバッファ
PCN_RAW_REPLAY_NPROC=48                # Phase1 obsリプレイ 1026s→約333s(暫定、v8bで廃止)
PCN_EVAL_GAP_FEEDBACK=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=0  # 学習中の格子評価OFF
EVALINT=NITER                          # 最終のみ(checkpoint保存の制約上、倍数必須)
# +微小変更: raw時のwarmup XLA空回しスキップ(gpu_factory.py:402-410をifで囲む) → −12分
```

### v8a 見込み時間(NITER=14, 16本/iter, 192本初期)

| 工程 | 時間 | 備考 |
|---|---:|---|
| Phase1 | ≈24分 | raw 18分+obsリプレイ5.5分(データ29%増えた上で) |
| Phase2 | ≈60分 | **cache初期構築≈25分(CPU)+2,000更新37分** ← v8aの新ボトルネック。epochs=10なら≈43分 |
| 学習ループ | **≈277分(4.6h)** | **max(lockstep 19.8分, learn≈4分)×14** ← 重い指令の実測値で再計算。11.4分想定は誤り |
| 最終評価 | ≈60分 | 全量工場lockstep(格子~160指令÷64チャンク×19.8分) |
| **合計** | **≈7.4時間** | ★**v6(5.5h)より遅い**。16本/iterでlockstepを使うのは配員ミス(下記) |

### ★v8a単純適用は失敗する — 「定額の高さ」を本数で割るしかない

lockstepは**1チャンク定額**(B非依存)なので、16本/iterで使うと1本あたり74秒相当。CPU Actor(16本を4.1分=1本15秒)に**5倍負ける**。定額を正当化するには本数を積むしかない:

| 構成 | 本/iter | iter数 | ループ時間 | 総エピソード | 方策更新の刻み |
|---|---:|---:|---:|---:|---|
| v6実測(CPU P3) | 16 | 14 | 57分 | 224 | 14回 |
| v8a素直(GPU P3) | 16 | 14 | 277分 | 224 | 14回 ← **論外** |
| **v8a配員修正** | **64** | **14** | **277分** | **896(4倍)** | 14回 |
| v8a時短型 | 64 | 7 | 139分 | 448(2倍) | 7回 |

**つまり「全GPU化はデータ量を4倍にして初めて元が取れる」**。同じ224本を作るだけなら、GPU化は損。逆に「1iterに64本を投げても時間が変わらない」性質は、**データ量を増やす方向にはタダで効く**(探索の厚みが4倍=品質側の伸び)。

## 3. ★正直なトレードオフ(全GPU化の壁)

**lockstepの11.4分/チャンクはB非依存(T逐次律速)なので、GPU枚数でもバッチでも割れない。**16本/iterの少量生産では、CPU Actor16体(overlap込み実測7.7分/iter)に負ける。対処は3軸:

1. **iterの本数を増やしiter数を減らす**(例: NITER=7×32本/iter): lockstepはB非依存なので**タダで倍積める**→ループ≈80分、総データ量同一。ただし方策改善の刻みが半分になる=**refit廃止で取り戻したブートストラップ頻度と正面衝突**。ここはユーザーの研究判断事項
2. **lockstepカーネル自体の高速化**(13.7ms/stepの内訳改善、5発行のCUDA Graph化): 見込み1.5〜2倍、要カーネル手術
3. **learnを重くしてオーバーラップを活かす**(N_UPDATESを100→200+): rollout11.4分の裏でGPU0を遊ばせない。更新回数は品質パラメータなので副作用に注意

## 4. v8b — 手術込みの本命(見込み計≈2〜2.5時間)

| 手術 | 内容 | 効果 | 規模 |
|---|---|---|---|
| ① lockstepに`mode="fixed"`追加 | 事前生成行動列でNNなしlockstep→**Phase1のobsをGPU構築**、CPUリプレイ全廃 | Phase1≈10分・CPU脚ゼロ | 小(lockstep_nn.pyに1モード) |
| ② 教師cacheのGPUエンコード(SoA化) | _encode_episode_training_blockのGPU化。Phase2初期25分+毎iter2分が数分に | Phase2≈20分・iterのlearn軽量化 | 大(既知の恒久課題) |
| ③ 工場ワーカ2体化 | Phase1と評価のチャンクをGPU0/GPU1へ分割(Phase1中・評価中はLearner暇) | Phase1・評価をさらに≈2分の1 | 中(ドライバ+シャーディング) |
| ④ lockstep高速化 | obsカーネル増分の続き+CUDA Graph | ループ160→≈100分 | 中〜大 |

## 5. 検証プラン(正確性は譲らない)

1. v8aスモーク(NITER=2): P3のlockstep sample行動がCPU Actorと**同分布**(検証器verify_lockstep_nn.py、ビット別・分布一致が既検証)であることをcmd_track/損失の健全性で確認
2. CHEAP_TO_ACTOR=0+E_MAX=65536で**ovf=0**を確認(溢れたら自動拡大→失敗時は例外=黙って壊れない設計)
3. mode="fixed"(v8b①)はverify_replay_obs.py相当で**CPUリプレイと全ステップ一致**を移植時に確認

## 5.5 床の検証(「それで限界か」を常に確かめる)

| 数字 | 1点測定か | 床の検証状況 |
|---|---|---|
| lockstep 19.8分/iter | 重い指令での実測(08-23) | **要B掃引**: B=16/64/256で定額が本当に不変か未検証。v8a_smokeで実測中 |
| lockstep 23.8ms/step | 上から算出 | **床でない**: raw 7.8ms/stepとの差16msはobs+NNの取り分。obs増分化は4.16→1.76倍まで来ており、残りの余地は未測定。CUDA Graph化(5発行→1)も未着手 |
| raw 7.8ms/step (B=64) | 実測 | **床でない**: B=1024で13.1ms/stepだが本数16倍=**スループット9.5倍**。レジスタ148個の席上限≈1000ブロックまで伸びる |
| Phase1 obsリプレイ nproc=48 | 未実測(16の実測から外挿) | 32/48/64を1チャンクで実測してから固定(fork+JAXデッドロック注意) |
| 教師cache初期25分 | refit時代47分/18M遷移からの比例推定 | 未実測 |

## 6. 未確定(見積りの仮定)

- lockstepチャンクが混雑エピソード(cheap端復活)でどこまで伸びるか(11.4分は従来分布での実測)
- Phase2のcache初期構築25分はrefit時代の47分/18M遷移からの比例推定
- 評価160指令の構成(格子12²+低尾)は設定依存
