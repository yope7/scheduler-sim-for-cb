# フラグ管理・バグ監査 (2026-08-25)

5系統の並列監査 + 私による裏取り。**裏取りできたものだけ**を「確認済み」に置き、エージェントの主張でも反証できたものは明示的に却下した。

## ❌ まず却下: 「20フラグがLearnerに届かない」は誤り

エージェントの最重要主張は「`pcn_agent`のimport(distributed_pcn.py:341)で定数が凍結され、その後(4059-4076)にプロファイルが環境変数を設定しても手遅れ。Learnerは旧値で動く」だった。**これは誤り**:

- `Learner`は素のクラスではなく **`LearnerActor = ray.remote(num_gpus=1)(Learner)`(4162行)** としてRay actor化される
- `ray.init()`は4143行 = **プロファイル適用(4059-4076)より後**
- Rayワーカーは起動時に更新後の`os.environ`を継承し、`pcn_agent`を新規importする
- 実行ログでも `(Learner pid=2375300)` = 別プロセス

→ **LearnerもActorも正しいフラグ値で動いている**。ただしドライバプロセス自身は旧値のまま(オーケストレーションにしか使わないので実害は限定的)。

---

## ✅ 確認済みの実害(優先度順)

### P0-1. 学習中evalでdropoutが効いている(best model選抜がノイズで揺れる)

| 連鎖 | 根拠 |
|---|---|
| プロファイルが`PCN_S_EMB_DROPOUT=0.08`をsetdefault | workload_pcn_profile.py:43 / distributed_pcn_cli.py:48。run_j50000_gpu.shは未設定なので**適用される** |
| Actorは重みロード時に`.eval()`を呼ばない | `_load_policy_weights`は`load_state_dict`のみ。`.eval()`は`load_checkpoint`(1911行)にしか無い |
| モデルは`training=True`のまま | nn.Module既定 |
| forwardでdropout発火 | pcn_agent.py:794 `if self.training and _S_EMB_DROPOUT > 0.0` |

**影響**: ①学習中evalが非決定的 → HV/PF/EARLYSTOPのbest選抜が乱数で揺れる ②外部評価器は`tg.eval()`でdropout OFF → **学習中evalと最終報告値が系統的に別物** ③`_act`の`elif _JIT_ACT and _S_EMB_DROPOUT <= 0.0`により、**TorchScript JIT高速化(forward 1.5倍)も黙って無効化されている**

**対策**: Actorの`_load_policy_weights`直後に`model.eval()`、rollout前に`train()`へ戻す。または100iter実験は`PCN_S_EMB_DROPOUT=0`で固定。

### P0-2. Phase3の工場失敗が一切チェックされない(1iter分のデータが黙って消える)

- `gpu_factory.run_commands`の`except Exception`(1014-1019)が`episodes_generated=0, _factory_failed=...`を返す
- 受け側`_phase3_actor_wave`(5185-5194)と`_episodes_generated_sum`/`_collect_command_outcomes`(5038-5051)は**`_factory_failed`を読まない**
- **Phase1は対策済み**(4337で`raise RuntimeError`)。Phase3だけ非対称

**影響**: OOM/コンパイル失敗で新規エピソード0本の周が発生しても、`ray.get`は例外を出さず完走する。100iterのうち後半が全滅していても気づけない。

### P0-3. `healthy()`が一度も呼ばれない(フォールバック機構が死んでいる)

`gpu_factory.py:399`に定義、docstring(22-23行)は「連続2回失敗でhealthy()=Falseになり従来Actor経路に戻る」と書くが、**`src/`全体で呼び出し0件**(定義とコメントのみ)。`_fail_streak`はインクリメントされるだけの死んだ変数。P0-2と合わせて「工場が死んだまま100iter完走」が成立する。

### P0-4. AMP既定ONで凍結検知器が盲目

- `use_amp = os.environ.get("PCN_USE_AMP", "1") == "1"` = **既定ON**(pcn_agent.py:1530)
- `_nan_skip_total`/`_opt_step_total`の更新は**非AMP経路(`else:`ブロック)にしかない**(3779以降)
- AMPの`scaler.step()`はinf/nan勾配で**無言でstepをスキップ**するがカウンタを更新しない
- driver側は`if _tot > 0:`でのみ`[STEP_SKIP]`を出す(5325-5331) → **既定設定では一度も出ない**

**影響**: GradScalerが全stepをスキップして重みが1mmも動かなくても、ログは完全に無音。しかもこの検知器自身が`except Exception: pass`(5316-5333)で囲われている。

### P0-5. `CHEAP_TO_ACTOR=1`がエピソードを捨てる(既報・スクリプト修正済)

p<0.2を「exact種に委譲」と印字するが、委譲先3経路(`PHASE1_HEURISTIC_THRESHOLDS`/`PHASE1_GIANT_DEFER`/`SEED_CHROMOSOMES`)が**全て未設定**。v6では計画192本→136本(29%消失)。**p=0.0=全オンプレ=cost最小端**が消えるため、PFの左端が生えない。

### P1-1. `update_many`失敗 → loss=0.0で「完璧に収束」に見える

`distributed_pcn.py:2624-2629`が例外時に`total_loss.extend([0.0]*n_updates)`し、`global_step`も進め、`update_weights_ref()`まで通る。**未更新の重みが「更新済み」として配布される**。表示は`平均損失: 0.0000`。NaN損失も0.0に潰される経路が3箇所(pcn_agent.py:3712/3869、JAX経路)。

### P1-2. 溢れ検知が3系統で機能しない

| 箇所 | 症状 |
|---|---|
| `_run_count_ev`のエスカレーション(486-500) | j50000設定(NAMB=2561/KPICK=128)では上限に張り付き`esc`が空 → **溢れたまま無言でreturn** |
| Phase3の`overflow` | `run_commands`は返すが受け側が読まない → ログに1文字も出ない |
| eval経路の`overflow` | `_distributed_evaluate_episodes`/`_driver_eval_gap_feedback`が読まない → **溢れたwaitでbest選抜される** |
| `_produce_random_raw`/`_produce_cmd_raw` | top-levelに`overflow`キーを置かない/Falseハードコード → `res.get('overflow')`は常にFalse |

### P1-3. 評価規約の非対称

| 項目 | 学習 | 評価 |
|---|---|---|
| `desired_horizon`初期値 | `len(ep)-2`(pcn_agent.py:4173/4246/4711) | `len(ep)`(全eval経路) |
| `PCN_DESIRED_RETURN_UB`クリップ | 適用(distributed_pcn.py:1625) | **非適用**(pcn_agent.py:4990の`not eval_mode`) |
| dropout | ON | ON(学習中eval) / OFF(外部評価器) |

`-2`は原論文の著者実装由来だが論文本文に記述がなく、著者実装でも学習中evalと最終evalで非対称(既知)。

### P1-4. checkpointがフラグ構成を記録していない

`save_model`(3619-3626)が保存するのは`model_state_dict/global_step/config/model_type/device`のみ。**`PCN_*`/`SCHEDULER_*`は一切保存されない**。読み込みは全経路`strict=False`で、`load_checkpoint`は欠損キー数をprintするだけでraiseしない。→ 旧規約ckptをwarm-startすると欠けたキー(fourier_freqs/command_balance/film_*)が**初期値のまま学習が続く**。

### P2. 死にフラグ・ガード漏れ

| フラグ | 状態 |
|---|---|
| `PCN_EVAL_PF_GRID` / `PCN_EVAL_STOCHASTIC` | **読み手ゼロ**。14スクリプトが設定しているが全て無効 |
| `PCN_COMMAND_BALANCE_TARGET` | 代入されるが参照0件(HV山登り方式への移行で削除漏れ) |
| `PCN_TRAIN_HEAD_STEP_WEIGHT` (+`_FRAC`) | ガード漏れでno-op。しかも初回`None`後は増分パスが恒久的に到達不能 |
| `PCN_TEACH_FRONT_ONLY` / `PCN_FROZEN_PF_MAX` | `PCN_FROZEN_PF_CLONE=1`が前提(既定OFF)。**ただし論文§4.4は非支配のみに絞ることを警告しているので、有効化しないのが正しい** |
| 条件付き死に 18件 | 親フラグ既定OFFで到達不能(COMMAND_BALANCE_*, COND_ADD_SCALE, FOURIER_BANDS, MPFT_*, DEDUP_TRAIN_DECIMALS 等) |

**総計: 338個の環境変数のうち約43個(13%)が要対応。**

---

## 提案: 「フラグ台帳」を1箇所に作る

個別修正より、**起動時に有効なフラグ構成を印字し、前提フラグの整合を検査する**のが根治策。

```python
# 起動時(ray.init後、Learner生成前)に1回
def audit_flags() -> None:
    REQUIRES = {  # 子フラグ: 前提フラグ
        "PCN_TEACH_FRONT_ONLY": "PCN_FROZEN_PF_CLONE",
        "PCN_FROZEN_PF_MAX": "PCN_FROZEN_PF_CLONE",
        "PCN_COND_ADD_SCALE": "PCN_FILM",
        "PCN_FOURIER_BANDS": "PCN_FOURIER_CMD",
        "PCN_DEDUP_TRAIN_DECIMALS": "PCN_DEDUP_TRAIN_WEIGHT",
        # ... 18件
    }
    DEAD = ["PCN_EVAL_PF_GRID", "PCN_EVAL_STOCHASTIC", "PCN_COMMAND_BALANCE_TARGET"]
    for child, parent in REQUIRES.items():
        if _truthy(child) and not _truthy(parent):
            raise ValueError(f"{child} は {parent}=1 が前提です(現在OFF=無効)")
    for f in DEAD:
        if f in os.environ:
            print(f"[FLAG_AUDIT] ⚠️ {f} は読み手が存在しません(無効)")
    print("[FLAG_AUDIT] 有効フラグ:", {k: v for k, v in sorted(os.environ.items())
          if k.startswith(("PCN_", "SCHEDULER_", "DISTRIBUTED_PCN_"))})
```

加えて、**iterationサマリーJSONにhealthセクション**を足すと「エラーも出ずに結果だけおかしい」が「終了時に必ず気づく」に変わる:

```python
_summary["health"] = {
  "factory_failed_iters": [...],      # P0-2
  "factory_overflow_iters": [...],    # P1-2
  "episodes_expected_vs_got": [...],  # P0-2/P0-5
  "update_failed_iters": [...],       # P1-1
  "step_skip": (skip, ok),            # P0-4
  "teacher_cache_steps": [...],
  "backend": {"jax":…, "amp":…, "factory":…, "raw_kernel":…},
}
```

---

## 100iter実験の前に必須の修正(最小セット)

1. **`PCN_S_EMB_DROPOUT=0`を明示**(またはActorで`model.eval()`) — best選抜の再現性+JIT復活
2. **Phase3の`_factory_failed`/`overflow`検査** — データ消失の検知
3. **AMP経路のカウンタ実装** — 凍結検知器を機能させる
4. **`CHEAP_TO_ACTOR=0`** — cost端の復活(E_MAX=8192で溢れないことは実測済み)
5. **起動時フラグ台帳** — 上記の再発防止

## 未検証(残タスク)

- スクリプト整合監査(実行中)
- 長時間実行バグ監査(実行中)
- `_replay_max=300`床の妥当性、ReplayBuffer max_size=5000のFIFO破棄
