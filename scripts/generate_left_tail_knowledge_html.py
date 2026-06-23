#!/usr/bin/env python3
"""Generate docs/distributed_pcn_left_tail_knowledge.html from experiment artifacts."""
from __future__ import annotations

import json
import html
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "distributed_pcn_left_tail_knowledge.html"
EXP = ROOT / "experiments" / "distributed_pcn"


def rel_img(path: Path) -> str:
    """HTML は docs/ 配下のため、実験図は常に ../experiments/... になる。"""
    rel = os.path.relpath(path.resolve(), DOCS.resolve())
    return html.escape(rel.replace("\\", "/"))


def img_block(path: Path, caption: str, css: str = "") -> str:
    if not path.is_file():
        return f'<figure class="missing"><figcaption>{html.escape(caption)} — ファイルなし: {path}</figcaption></figure>'
    src = rel_img(path)
    style = f' style="{css}"' if css else ""
    return (
        f'<figure class="fig"{style}>'
        f'<img src="{src}" alt="{html.escape(caption)}" loading="lazy">'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
    )


def load_goal(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text())
    return None


def goal_row(name: str, g: dict, note: str = "") -> str:
    ok = "達成" if g.get("goal") else "未達"
    rng = "本番規模" if g.get("ok_range") else "小規模試験"
    return (
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{g.get('score', 0):.1f}</td>"
        f"<td>{g.get('knee_drop', 0):.0f}</td>"
        f"<td>{g.get('low_slope_gap', 0):.0f}</td>"
        f"<td>{g.get('cost_max', 0):.0f}</td>"
        f"<td>{g.get('wait_min', 0):.0f}</td>"
        f"<td>{rng}</td><td>{ok}</td>"
        f"<td>{html.escape(note)}</td></tr>"
    )


def gallery(paths: list[tuple[Path, str]], cols: int = 2) -> str:
    items = "".join(img_block(p, c, "max-width:100%") for p, c in paths if p)
    return f'<div class="gallery cols-{cols}">{items}</div>'


def main() -> None:
    # --- metrics ---
    runs: list[tuple[str, Path, str]] = [
        ("dual12 ベスト (1024)", EXP / "pf_best_current/pf_bulge_left_tail_best.json", "SCALE1024_left_tail_best.pth"),
        ("knee05 基準", EXP / "pf_best_current/pf_bulge_midcore_knee05.json", "中域向け・低域ギャップ最良"),
        ("eval_gap 試行", EXP / "left_tail_eval_gap_20260601_213046/goal.json", "12 iter・悪化"),
        ("scratch100", EXP / "dual12_scratch100_20260601_093703/goal.json", "0→100 iter"),
        ("24j Phase2=0", EXP / "20260601_233851/bulge.json", "高速スモーク"),
        ("24j Phase2=100", EXP / "left_tail_p2e100_j24_20260602_023001/goal.json", "教師あり100ep"),
        ("lowfix", EXP / "left_tail_lowfix_20260601_194827/goal.json", ""),
    ]
    table_rows = ""
    for name, p, note in runs:
        if p.suffix == ".json" and "bulge" in p.name:
            import subprocess
            r = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/pf_left_tail_goal.py"), str(p)],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            g = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
        else:
            g = load_goal(p) or {}
        if g:
            table_rows += goal_row(name, g, note)

    # LIVE_PF series
    p2_dir = EXP / "left_tail_p2e100_j24_20260602_023001/20260602_023004"
    live_pf = [
        (p2_dir / f"uniform_cmd_pf_iter_{i:03d}.png", f"LIVE_PF iter {i}")
        for i in range(10, 101, 10)
    ]

    pf_best_imgs = sorted((EXP / "pf_best_current").glob("uniform_cmd_pf_*.png"))
    pf_best_pairs = [(p, p.name) for p in pf_best_imgs]

    key_pf = [
        (EXP / "pf_best_current/uniform_cmd_pf_midcore_knee05_iter205_20260531_042845.png", "knee05 @1024"),
        (EXP / "pf_best_current/uniform_cmd_pf_dual12_scratch100_20260601_160644.png", "scratch100"),
        (EXP / "pf_best_current/uniform_cmd_pf_left_tail_lowfix_20260601_223217.png", "lowfix"),
        (EXP / "20260601_233851/uniform_cmd_pf_latest.png", "24j Phase2=0"),
        (EXP / "left_tail_p2e100_j24_20260602_023001/uniform_cmd_pf_p2e100_j24_final_20260602_034430.png", "24j Phase2=100 (grid16)"),
        (EXP / "pf_best_current/SCALE1024_midcore_knee05.png", "knee05 別形式"),
        (EXP / "pf_best_current/SCALE1024_midcore.png", "midcore"),
        (EXP / "pf_best_current/SCALE1024_iter100.png", "scale1024 iter100"),
        (EXP / "pf_best_current/SCALE1024_left_tail_best.png", "left_tail best (if exists)"),
    ]

    phase2_key = [
        (p2_dir / "phase2_feature_importance/phase2_feature_importance.png", "Phase2 重要度 (p2e100)"),
        (p2_dir / "phase2_feature_importance/phase2_feature_importance_all.png", "Phase2 重要度 全特徴"),
        (EXP / "left_tail_dual12_20260531_221137/20260531_221139/phase2_feature_importance/phase2_feature_importance.png", "dual12"),
        (EXP / "20260601_233851/phase2_feature_importance/phase2_feature_importance.png", "24j p2=0"),
    ]

    all_uniform = sorted(EXP.rglob("uniform_cmd_pf*.png"))
    all_uniform_pairs = [(p, str(p.relative_to(EXP))) for p in all_uniform]

    quest_phase2 = sorted(EXP.glob("quest_*/**/phase2_feature_importance.png"))
    quest_p2_pairs = [(p, str(p.parent.parent.parent.name)) for p in quest_phase2]

    quest_rows = ""
    qlog = EXP / "quest_left_tail_log.jsonl"
    if qlog.is_file():
        for line in qlog.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            quest_rows += (
                f"<tr><td>{html.escape(o.get('trial','?'))}</td>"
                f"<td>{o.get('score',0):.0f}</td>"
                f"<td>{o.get('knee_drop',0):.0f}</td>"
                f"<td>{o.get('low_slope_gap',0):.0f}</td>"
                f"<td>{'✓' if o.get('beats_knee05_both') else '—'}</td></tr>"
            )

    rootfix_pf = sorted(EXP.rglob("rootfix*/**/uniform_cmd_pf*.png"))
    rootfix_pairs = [(p, str(p.relative_to(EXP))) for p in rootfix_pf]

    live_pf_rows = ""
    p2_exec = EXP / "left_tail_p2e100_j24_20260602_023001/20260602_023004"
    for sf in sorted(p2_exec.glob("uniform_cmd_stats_iter_*.json")):
        d = json.loads(sf.read_text())
        live_pf_rows += (
            f"<tr><td>{d.get('iteration','?')}</td>"
            f"<td>{d.get('n_pf','?')}</td>"
            f"<td>{d.get('n_achieved','?')}</td></tr>"
        )

    review_log = ""

    doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>分散PCN 左上PF改善 — 知見全集</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#0c1017; --surface:#151c28; --surface2:#1e2a3d; --text:#e8edf5;
  --muted:#93a4bc; --accent:#4da3ff; --accent2:#5ee0a0; --warn:#f0a050; --bad:#ff6b7a;
  --border:#2d3f56; --mono:'JetBrains Mono',monospace; --sans:'Noto Sans JP',system-ui,sans-serif;
}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.7}}
.hero{{padding:2.5rem 2rem;background:linear-gradient(135deg,#0a1628,#1a3050 60%,#0c1017);border-bottom:1px solid var(--border)}}
.hero h1{{margin:0 0 .5rem;font-size:clamp(1.4rem,3vw,2.2rem)}}
.hero .meta{{color:var(--muted);font-size:.9rem}}
.tab-bar{{display:flex;flex-wrap:wrap;gap:.35rem;padding:.75rem 1rem;background:rgba(12,16,23,.95);
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:50}}
.tab-btn{{background:var(--surface2);border:1px solid var(--border);color:var(--muted);
  padding:.45rem .85rem;border-radius:6px;cursor:pointer;font-size:.82rem;font-family:inherit}}
.tab-btn:hover{{color:var(--text);border-color:var(--accent)}}
.tab-btn.active{{background:var(--accent);color:#051018;border-color:var(--accent);font-weight:600}}
.tab-panel{{display:none;max-width:72rem;margin:0 auto;padding:1.5rem 1.25rem 4rem}}
.tab-panel.active{{display:block}}
h2{{color:var(--accent2);border-bottom:2px solid var(--accent);padding-bottom:.35rem;margin-top:2rem}}
h3{{color:var(--accent);margin-top:1.5rem}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.1rem 1.25rem;margin:1rem 0}}
.callout{{border-left:4px solid var(--warn);padding:.75rem 1rem;background:rgba(240,160,80,.08);margin:1rem 0}}
.callout.good{{border-color:var(--accent2);background:rgba(94,224,160,.08)}}
.callout.bad{{border-color:var(--bad);background:rgba(255,107,122,.08)}}
table{{width:100%;border-collapse:collapse;font-size:.88rem;margin:1rem 0}}
th,td{{border:1px solid var(--border);padding:.45rem .6rem;text-align:left}}
th{{background:var(--surface2)}}
tr:nth-child(even){{background:rgba(30,42,61,.4)}}
code,pre{{font-family:var(--mono);font-size:.85rem}}
pre{{background:#0a0e14;padding:1rem;border-radius:8px;overflow-x:auto;border:1px solid var(--border)}}
.gallery{{display:grid;gap:1rem;margin:1rem 0}}
.gallery.cols-2{{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}
.gallery.cols-3{{grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}}
.fig{{margin:0;background:var(--surface);border-radius:8px;overflow:hidden;border:1px solid var(--border)}}
.fig img{{width:100%;display:block}}
.fig figcaption{{padding:.5rem .75rem;font-size:.78rem;color:var(--muted)}}
.mermaid-wrap{{background:var(--surface);padding:1rem;border-radius:8px;overflow-x:auto}}
ul.compact li{{margin:.25rem 0}}
details{{margin:.75rem 0;border:1px solid var(--border);border-radius:8px;padding:.5rem 1rem;background:var(--surface)}}
summary{{cursor:pointer;font-weight:600;color:var(--accent)}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:700px){{.compare{{grid-template-columns:1fr}}}}
.story{{border:1px solid var(--border);border-radius:10px;margin:1.25rem 0;overflow:hidden}}
.story h4{{margin:0;padding:.6rem 1rem;font-size:.95rem}}
.story .obs{{background:rgba(255,107,122,.12)}} .story .obs h4{{color:var(--bad)}}
.story .hyp{{background:rgba(240,160,80,.10)}} .story .hyp h4{{color:var(--warn)}}
.story .act{{background:rgba(94,224,160,.08)}} .story .act h4{{color:var(--accent2)}}
.story .body{{padding:.75rem 1rem 1rem;font-size:.92rem}}
.story .verdict{{padding:.5rem 1rem;background:var(--surface2);font-size:.85rem;color:var(--muted);border-top:1px solid var(--border)}}
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
<header class="hero">
  <h1>ジョブ割り当ての学習 — 実験メモ（わかりやすい版）</h1>
  <p class="meta">更新: {datetime.now().strftime("%Y-%m-%d %H:%M")} · 研究に詳しくない人向けに書き直しました</p>
  <p><strong>やりたいこと</strong>：コストを抑えつつ待ち時間も短くしたい、という二つの目標の<strong>良い組み合わせの線</strong>に、
    学習後の方策（赤い線）が、過去の試行で見つかった良い結果（菱形）に近づくこと。</p>
</header>

<nav class="tab-bar" role="tablist">
  <button class="tab-btn active" data-tab="narrative">なぜ何をしたか</button>
  <button class="tab-btn" data-tab="glossary">用語と図</button>
  <button class="tab-btn" data-tab="live-pf">学習中の図</button>
  <button class="tab-btn" data-tab="overview">まとめ</button>
  <button class="tab-btn" data-tab="problem">うまくいかない例</button>
  <button class="tab-btn" data-tab="pipeline">学習の3段階</button>
  <button class="tab-btn" data-tab="eval-types">評価の種類</button>
  <button class="tab-btn" data-tab="experiments">実験結果</button>
  <button class="tab-btn" data-tab="phase2">第2段階の学習</button>
  <button class="tab-btn" data-tab="timeline">いつ何をしたか</button>
  <button class="tab-btn" data-tab="figures-pf">比較用の図</button>
  <button class="tab-btn" data-tab="figures-live">学習中の図一覧</button>
  <button class="tab-btn" data-tab="figures-all">保存した図すべて</button>
  <button class="tab-btn" data-tab="pitfalls">注意点</button>
  <button class="tab-btn" data-tab="commands">実行コマンド</button>
</nav>

<!-- TAB: narrative -->
<section id="narrative" class="tab-panel active">
<h2>なぜ、いろいろ試したのか</h2>
<p>各取り組みを「困ったこと → こうすればよいのでは → やったこと → どうなったか」に分けています。</p>

<div class="story">
  <div class="obs"><h4>① 困っていたこと</h4><div class="body">
    図の<strong>左下（コストが小さい側）</strong>で、<strong>赤い線</strong>（今の方策の結果）が<strong>菱形</strong>（過去の良い試行）より
    <strong>待ち時間が長いまま横に伸びる</strong>ことがある。コストを少し上げると待ち時間が急に下がる、という不自然な形もある。
  </div></div>
  <div class="hyp"><h4>② 考えたこと</h4><div class="body">
    目標の指定の仕方を学習していない、または安いコスト帯の例が少なすぎるのではないか。
    数値の扱いを直し、安いコスト帯だけ学習を強くすれば、赤い線が菱形に近づくのでは。
  </div></div>
  <div class="act"><h4>③ やったこと</h4><div class="body">
    学習の数値まわりを修正し、補助的な損失（value の再現）を切った。
    そのうえ設定「dual12」で、安いコスト帯向けの学習を二通り組み合わせた。
  </div></div>
  <div class="verdict"><strong>結果</strong>：急に下がる形は少しマシになったが、安いコスト帯のずれはまだ大きい。
    「急な下がりを直す」と「安いコストで菱形に寄せる」を同時にはまだ満たしていない。</div>
</div>

<div class="story">
  <div class="obs"><h4>① 困っていたこと</h4><div class="body">
    ある補助損失を入れると、赤い点が図の一角にまとまり、目標の指定に反応しなくなる。
  </div></div>
  <div class="hyp"><h4>② 考えたこと</h4><div class="body">
    補助の予測用ネットワークが、本筋の方策の学習を邪魔しているのではないか。
  </div></div>
  <div class="act"><h4>③ やったこと</h4><div class="body">
    その補助損失を通常はオフにした。
  </div></div>
  <div class="verdict"><strong>結果</strong>：これは悪化の原因としてほぼ確定。
    第2段階の学習回数をゼロにしていることとは別の話（第2段階をゼロにしたから悪化した、という記録はない）。</div>
</div>

<div class="story">
  <div class="obs"><h4>① 困っていたこと</h4><div class="body">
    どこが悪いか分かっても、過去データへの重みが均等で、弱いところが学習に反映されにくい。
  </div></div>
  <div class="hyp"><h4>② 考えたこと</h4><div class="body">
    悪い帯域だけ過去データの重みを上げれば、そこが改善するのでは。
  </div></div>
  <div class="act"><h4>③ やったこと</h4><div class="body">
    一定の目標を並べて試し、菱形とのずれが大きい帯域だけ重みを上げる仕組みを入れた（安いコスト帯だけ）。
  </div></div>
  <div class="verdict"><strong>結果</strong>：短い追加学習では、むしろ総合点が悪化した例もある。
    24件ジョブの試運転ではずれの数値は下がったが、本番1024件では未達のまま。</div>
</div>

<div class="story">
  <div class="obs"><h4>① 困っていたこと</h4><div class="body">
    学習が終わるまで良い図が見られず、気づくのが遅い。
  </div></div>
  <div class="hyp"><h4>② 考えたこと</h4><div class="body">
    10回ごとに簡易版の図を保存すれば、おかしさに早く気づけるのでは。
  </div></div>
  <div class="act"><h4>③ やったこと</h4><div class="body">
    学習中に自動で PNG を出す機能（LIVE_PF）を入れた。ただし格子は粗く、本番の図とは別物。
  </div></div>
  <div class="verdict"><strong>結果</strong>：ざっくりの監視には使えるが、図の質は本番用評価より劣る（「学習中の図」タブ参照）。
    ある run では学習中の図は出ず、終わったあとだけ図がある。</div>
</div>

<div class="story">
  <div class="obs"><h4>① 困っていたこと</h4><div class="body">
    本番（ジョブ1024件）は何時間もかかる。設定ミスに気づくのが遅い。
  </div></div>
  <div class="hyp"><h4>② 考えたこと</h4><div class="body">
    ジョブ24件で同じ手順を回せば、パイプラインだけ短時間で確認できるのでは。
  </div></div>
  <div class="act"><h4>③ やったこと</h4><div class="body">
    24件×100回の学習を、第2段階あり・なしで実施。
  </div></div>
  <div class="verdict"><strong>結果</strong>：手順の確認には使える。ただしコストの最大値が本番の目標（500万）に届かないので、
    そこでの合否判定は使えない。第2段階100回は24件ではずれ指標がよくなったが、1024件では未検証。</div>
</div>
</section>

<!-- TAB: glossary -->
<section id="glossary" class="tab-panel">
<h2>用語と図の見方（このページだけで読むための説明）</h2>
<div class="card">
<table>
<tr><th>言葉</th><th>意味</th></tr>
<tr><td>横軸・コスト</td><td>かかった費用。左ほど安い。</td></tr>
<tr><td>縦軸・待ち時間</td><td>平均待ち時間。下ほど短い（よい）。</td></tr>
<tr><td>菱形の点</td><td>これまでの試行で見つかった「コストと待ち時間の良い組み合わせ」。</td></tr>
<tr><td>赤い線・赤い点</td><td>いろいろな<strong>目標</strong>を指定して、<strong>今の方策</strong>が実際に出した結果をつないだもの。</td></tr>
<tr><td>青い点</td><td>指定した目標ごとの到達点（重なっていることも多い）。</td></tr>
<tr><td>目標の指定</td><td>「このくらいのコスト・待ち時間を目指せ」とネットワークに渡す数値（論文の command に相当）。</td></tr>
<tr><td>過去データ（replay）</td><td>学習に使い回す、過去の試行の記録。</td></tr>
<tr><td>第1・2・3段階</td><td>①ランダムにデータ収集 ②そのデータで模倣 ③本番の反復学習。</td></tr>
<tr><td>ずれ（gap）</td><td>同じくらいのコストで、赤が菱形より待ち時間がどれだけ長いか。</td></tr>
<tr><td>急な下がり（knee_drop）</td><td>コストを少し上げたとき、待ち時間がどれだけ急に減るか。</td></tr>
</table>
</div>
<h3>図で「近い」をどう測っているか</h3>
<ul>
<li><strong>本番用の図</strong>：たくさんの目標を格子状に並べ、各点で「菱形の同じコスト帯より待ち時間がどれだけ悪いか」を平均する。</li>
<li><strong>学習中の別評価（50本）</strong>：過去データから50個の目標を選び、目標ベクトルと実際の報酬ベクトルの距離を測る（図の赤線とは別の評価）。</li>
</ul>
<p>「点を指定したとき、近くの値にどれだけ近いか」は<strong>評価の種類ごとに違う</strong>。詳しくは「評価の種類」タブ。</p>
</section>

<!-- TAB: live-pf -->
<section id="live-pf" class="tab-panel">
<h2>学習中に自動保存される図（LIVE_PF）の読み方</h2>
<div class="callout bad">
<strong>要点</strong>：この図は<strong>おおよその監視用</strong>です。論文や最終判断には、学習終了後に別スクリプトで出す図を使ってください。
図が汚いからといって、必ずしも学習が失敗したとは限りません。ただし赤い線の点が極端に減ったときは要注意です。
</div>

<h3>同じ保存モデルでも、図の作り方で数字が違う（24件ジョブ・100回目）</h3>
<table>
<tr><th>どう作ったか</th><th>目標の並べ方</th><th>試した数</th><th>赤線の点数（タイトル）</th><th>補足</th></tr>
<tr><td>学習中の図</td><td>粗い格子</td><td>324</td><td><strong>12</strong></td><td>青点は324あるが赤線だけ少ない</td></tr>
<tr><td>終了後の本番用図</td><td>細かい格子</td><td>560</td><td>324</td><td>別の数え方もあり</td></tr>
</table>

<h3>学習の進みと赤線の点数（24件・第2段階100回の run）</h3>
<table>
<tr><th>何回目</th><th>赤線の点数</th><th>試した目標の数</th></tr>
{live_pf_rows}
</table>
<p>10回目は赤線73点、90回目は<strong>1点</strong>まで減る。試す目標の数324は変わらない。
→ 出てくる結果が似た点に寄り、赤い「良い線」だけが消えて見える。</p>

<h3>本番用の図と違う理由</h3>
<ul>
<li>格子が粗い（12分割 vs 16分割＋追加点）</li>
<li>菱形の点の集め方が簡略版</li>
<li>タイトルは赤線の点数だけ。青い点はたくさんあっても表示されない</li>
<li>24件ジョブではコストの桁が本番と違う（数万 vs 数百万）</li>
</ul>

<h3>run ごとの注意</h3>
<ul>
<li>20260601_233851：学習中の iter 図は<strong>無い</strong>。latest.png は<strong>終了後</strong>に別途作った図</li>
<li>left_tail_p2e100：学習中の図は出るが、上の表のとおり赤線点数が不安定</li>
</ul>

<div class="compare">
{img_block(p2_exec / "uniform_cmd_pf_iter_010.png", "10回目：赤線がまだ多い")}
{img_block(p2_exec / "uniform_cmd_pf_iter_090.png", "90回目：赤線がほぼ1点")}
</div>
<div class="compare">
{img_block(p2_exec / "uniform_cmd_pf_iter_100.png", "100回目・学習中の図")}
{img_block(EXP / "left_tail_p2e100_j24_20260602_023001/uniform_cmd_pf_p2e100_j24_final_20260602_034430.png", "同じモデル・終了後の本番用図")}
</div>
</section>

<!-- TAB: overview -->
<section id="overview" class="tab-panel">
<h2>いま分かっていること（短く）</h2>
<div class="card">
<ul class="compact">
<li><strong>いちばんマシな保存モデル（ジョブ1024件）</strong>：設定 dual12。急な下がりは改善したが、安いコスト帯のずれはまだ大きい。総合の合否は未達。</li>
<li><strong>安いコスト帯だけ見ると別設定が良かった例</strong>：knee05。ずれは小さいが、急な下がりが悪い。</li>
<li><strong>両方同時に良いモデル</strong>：まだ見つかっていない。</li>
<li><strong>24件の試運転</strong>：手順確認用。本番の合否数字は使えない（コストの桁が違う）。</li>
<li><strong>第2段階を100回やる（24件）</strong>：ずれ指標はよくなったが、時間が約2倍。第2段階ゼロが悪いという証拠はない。</li>
<li><strong>確実に悪かったもの</strong>：補助的な value 再現損失（入れると目標指定に反応しなくなる）。</li>
</ul>
</div>

<h3>本番の合否で見ている数字（ジョブ1024件想定）</h3>
<table>
<tr><th>名前</th><th>よい方向</th><th>何を見ているか</th></tr>
<tr><td>急な下がり</td><td>3000以下</td><td>コストを少し上げたとき、待ち時間が急にどれだけ減るか</td></tr>
<tr><td>安いコスト帯のずれ</td><td>1800以下</td><td>安いコストで、赤が菱形より待ち時間がどれだけ長いか（平均）</td></tr>
<tr><td>コストの最大</td><td>500万以上</td><td>高コスト側まで試せているか</td></tr>
<tr><td>待ち時間の最小</td><td>5000以下</td><td>短い待ち時間側も維持できているか</td></tr>
</table>
</section>

<!-- TAB: problem -->
<section id="problem" class="tab-panel">
<h2>うまくいかないときのパターン</h2>
<p>図は横軸がコスト、縦軸が待ち時間。菱形が「過去の良い試行」、赤が「今の方策が目標を変えながら試した結果」です。</p>

<div class="compare">
{img_block(EXP / "pf_best_current/uniform_cmd_pf_midcore_knee05_iter205_20260531_042845.png", "安いコスト帯のずれは比較的小さい例")}
{img_block(EXP / "pf_best_current/uniform_cmd_pf_dual12_scratch100_20260601_160644.png", "左下が横ばい・急に下がる例")}
</div>

<div class="card">
<ol>
<li><strong>左下が横ばい</strong> — コストを変えても待ち時間が下がらない</li>
<li><strong>途中で急に下がる</strong> — 不自然に待ち時間だけ良くなる</li>
<li><strong>点が一角に固まる</strong> — 目標指定が効いていない</li>
<li><strong>安いコストだけ強く学びすぎ</strong> — 全体の形が歪む設定もあった</li>
</ol>
</div>
</section>

<!-- TAB: pipeline -->
<section id="pipeline" class="tab-panel">
<h2>学習の3段階</h2>
<ol>
<li><strong>第1段階</strong>：ランダムに近い割り当てでデータをたくさん集める（例：3200本）。</li>
<li><strong>第2段階</strong>：そのデータで「真似る」学習（回数0でも、準備処理は走る）。</li>
<li><strong>第3段階</strong>：100回くり返し、新しい試行を足しながら方策を更新。10回ごとに評価と図。</li>
</ol>
</section>

<!-- TAB: eval-types -->
<section id="eval-types" class="tab-panel">
<h2>評価は3種類ある（ここが混乱の元）</h2>
<p>「10回ごとの評価」と「学習中の図」「終了後の図」は<strong>別物</strong>です。目標の並べ方も本数も違います。</p>

<table>
<tr><th>種類</th><th>いつ</th><th>目標の数・作り方</th><th>何を測るか</th></tr>
<tr><td>A. 学習用の定期評価</td><td>10,20,…100回目</td><td>過去データから<strong>最大50個</strong>を選ぶ（格子ではない）</td><td>目標ベクトルと実際の報酬の距離</td></tr>
<tr><td>B. 学習中の図（LIVE_PF）</td><td>同上の直後</td><td>格子状に<strong>約324個</strong>（粗い）</td><td>菱形とのずれ＋図。赤線点数は少なく出ることがある</td></tr>
<tr><td>C. 終了後の本番図</td><td>学習後に手動</td><td>格子状に<strong>約560個</strong>（細かい）</td><td>菱形とのずれ（合否判定用）</td></tr>
</table>

<div class="callout">
<strong>最後の評価</strong>：設定により、100回目の A をそのまま使い直すこともある（もう一度たくさん試さない）。
終了後の C とは別。
</div>

<h3>「点を指定して近いか」はどれか</h3>
<ul>
<li><strong>C（本番図）</strong>：コストが近い帯で「待ち時間が菱形よりどれだけ長いか」— 合否はここ。</li>
<li><strong>A</strong>：過去の目標ベクトルに、今の方策がどれだけ近い報酬を出したか — 図の赤線とは別尺度。</li>
<li><strong>B</strong>：C に近いが、格子が粗く、菱形の作り方も簡略。</li>
</ul>
</section>

<!-- TAB: experiments -->
<section id="experiments" class="tab-panel">
<h2>実験結果の表（数字は専門用語のまま残しています）</h2>
<p>score は小さいほど良い。本番規模＝ジョブ1024件で意味がある行。</p>
<table>
<tr><th>実験名</th><th>総合点↓</th><th>急な下がり</th><th>安いコストのずれ</th><th>コスト最大</th><th>待ち最小</th><th>規模</th><th>合否</th><th>メモ</th></tr>
{table_rows}
</table>

<h3>24件ジョブでの比較（試運転）</h3>
<div class="compare">
{img_block(EXP / "20260601_233851/uniform_cmd_pf_latest.png", "24件・第2段階なし")}
{img_block(EXP / "left_tail_p2e100_j24_20260602_023001/uniform_cmd_pf_p2e100_j24_final_20260602_034430.png", "24件・第2段階100回")}
</div>
<table>
<tr><th></th><th>第2段階なし</th><th>第2段階100回</th></tr>
<tr><td>かかった時間</td><td>約39分</td><td>約72分</td></tr>
<tr><td>安いコストのずれ（24件）</td><td>147</td><td>93</td></tr>
</table>
</section>

<!-- TAB: timeline -->
<section id="timeline" class="tab-panel">
<h2>いつ何をしたか</h2>
<div class="card">
<table>
<tr><th>時期</th><th>内容</th></tr>
<tr><td>5/30</td><td>数値の扱いを直し、補助損失を切る。ジョブ1024件で全体の形を安定化。</td></tr>
<tr><td>5/31</td><td>コストの中間帯・急な下がり向けの設定を試す。dual12 で急な下がりは改善。</td></tr>
<tr><td>6/1</td><td>設定を総当たり。弱点帯域の重み付けを入れるが、短い追加学習では悪化も。</td></tr>
<tr><td>6/1–2</td><td>ジョブ24件で試運転。第2段階100回は24件ではずれが改善。</td></tr>
</table>
</div>
</section>

<!-- TAB: phase2 -->
<section id="phase2" class="tab-panel">
<h2>第2段階の学習（真似る学習）について</h2>
<ul>
<li>第2段階を<strong>0回</strong>にしているのは、時間短縮の慣習。悪化したから、という記録はない。</li>
<li>悪化の原因として分かっているのは、<strong>別の補助損失</strong>（第3段階側）。</li>
<li>24件で第2段階を100回やると、損失は下がり、ずれ指標はよくなったが、本番1024件では未検証。</li>
</ul>
</section>

<!-- TAB: figures-pf -->
<section id="figures-pf" class="tab-panel">
<h2>比較用の図</h2>
{gallery(key_pf, 2)}
</section>

<!-- TAB: figures-live -->
<section id="figures-live" class="tab-panel">
<h2>学習中に保存された図（10回ごと）</h2>
{gallery(live_pf, 3)}
</section>

<!-- TAB: figures-all -->
<section id="figures-all" class="tab-panel">
<h2>保存してある図すべて（{len(all_uniform_pairs)} 枚）</h2>
{gallery(all_uniform_pairs, 3)}
</section>

<!-- TAB: pitfalls -->
<section id="pitfalls" class="tab-panel">
<h2>注意点</h2>
<ul>
<li>学習中の図と、終了後の図は<strong>目標の数・並べ方が違う</strong>。同じ評価だと思わない。</li>
<li>24件の run の図で、本番（1024件）の合否を決めない（コストの桁が違う）。</li>
<li>ある run では学習中の図が無く、終了後だけ図がある。</li>
<li>ジョブ数が違う保存モデルを混ぜない。</li>
</ul>
</section>

<!-- TAB: commands -->
<section id="commands" class="tab-panel">
<h2>実行コマンド（開発者向け）</h2>
<h3>本番（ジョブ1024件）</h3>
<pre>
PYTHONPATH=. .venv/bin/python -u -m src.distributed.distributed_pcn_event \\
  --left-tail --no-viz --jobs 1024 \\
  --n-iterations 100 --initial-episodes 100 --eval-interval 10 --eval-samples 50
</pre>
<h3>試運転（ジョブ24件・第2段階100回）</h3>
<pre>
PYTHONPATH=. .venv/bin/python -u -m src.distributed.distributed_pcn_event \\
  --left-tail --no-viz --jobs 24 \\
  --n-iterations 100 --initial-episodes 100 --eval-interval 10 --eval-samples 50 \\
  --supervised-epochs 100
</pre>
<h3>終了後の本番用の図</h3>
<pre>
PYTHONPATH=. .venv/bin/python scripts/eval_uniform_command_pf.py \\
  --checkpoint &lt;model_iter_100.pth&gt; --replay-snapshot &lt;learner_replay_snapshot.pkl.gz&gt; \\
  --output experiments/distributed_pcn/pf_best_current --label myrun \\
  --grid 16 --n-jobs 24 --low-tail-frac 0.18 --low-tail-extra 20
</pre>
<h3>HTML の見方</h3>
<p>画像は <code>docs/</code> からの相対パス（<code>../experiments/...</code>）です。
リポジトリ <strong>root</strong> で HTTP サーバを起動してください（<code>docs/</code> だけを cwd にしないこと）:</p>
<pre>cd /path/to/scheduler-sim-for-cb && python -m http.server 8765</pre>
<p><a href="http://localhost:8765/docs/distributed_pcn_left_tail_knowledge.html">http://localhost:8765/docs/distributed_pcn_left_tail_knowledge.html</a></p>
</section>

<script>
mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});
</script>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
