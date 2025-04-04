import itertools
import wandb
import yaml
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os
from datetime import datetime
from src.utils.job_gen.job_generator import JobGenerator
from multiEnv2 import SchedulingEnv

#printで改行しない
np.set_printoptions(linewidth=np.inf)

class SchedulingVisualizer:
    def __init__(self, results_data=None):
        self.results_data = results_data or []
        self.output_dir = "visualization_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_exhaustive_evaluation(self, nb_jobs):
        """
        指定したジョブ数に対して、ジョブごとに取るべき割り当て（0または1）の全組み合わせを試行し、
        結果をHTML形式でインタラクティブに可視化する
        """
        # wandb の初期化（ログ出力用）
        wandb.init(project="ExhaustiveAssignmentEvaluation", name="interactive_visualization")

        # コンフィグファイルの読み込み
        with open("config.yml", "r") as yml:
            config = yaml.safe_load(yml)

        n_window = config["param_env"]["n_window"]
        n_on_premise_node = config["param_env"]["n_on_premise_node"]
        n_cloud_node = config["param_env"]["n_cloud_node"]
        n_job_queue_obs = config["param_env"]["n_job_queue_obs"]
        n_job_queue_bck = config["param_env"]["n_job_queue_bck"]
        weight_wt = config["param_agent"]["weight_wt"]
        weight_cost = config["param_agent"]["weight_cost"]
        penalty_not_allocate = config["param_env"]["penalty_not_allocate"]
        penalty_invalid_action = config["param_env"]["penalty_invalid_action"]
        nb_steps = config["param_simulation"]["nb_steps"]
        nb_episodes = 0

        # 固定ジョブセットの生成
        # lam = 0.2
        job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, n_cloud_node, config, nb_jobs, 0.3, nb_episodes)
        jobs_set = job_generator.generate_jobs_set()
        print("jobs_set:", jobs_set)

        max_step = nb_jobs

        # 各ジョブに対するアクション（0 または 1）の全組み合わせを生成
        all_action_sets = list(itertools.product([1, 0], repeat=nb_jobs))
        print(f"Total action sets: {len(all_action_sets)}")

        results = []
        scheduling_details = []
        
        # 各組み合わせごとにエピソードをシミュレーション
        for i, action_set in enumerate(all_action_sets):
            print(f"Processing action set {i+1}/{len(all_action_sets)}: {action_set}")
            
            # 環境の初期化
            env = SchedulingEnv(
                max_step,
                n_window,
                n_on_premise_node,
                n_cloud_node,
                n_job_queue_obs,
                n_job_queue_bck,
                weight_wt,
                weight_cost,
                penalty_not_allocate,
                penalty_invalid_action,
                jobs_set=jobs_set,
                job_type=1,
                flag=1,
            )
            
            obs = env.reset()
            done = False
            step = 0
            wt_step_sum = 0
            
            # 各ステップでのマップの状態を記録
            map_history = {
                'on_premise': [],
                'cloud': []
            }

            # 無限ループ防止用
            iteration_count = 0
            max_iterations = nb_jobs * 10

            while not done and step < nb_jobs and iteration_count < max_iterations:
                iteration_count += 1
                action = action_set[step]
                
                # マップの現在の状態をコピーして保存
                map_history['on_premise'].append(env.on_premise_window_history_full.copy())
                map_history['cloud'].append(env.cloud_window_history_full.copy())
                
                # 環境ステップの実行
                obs, rewards, scheduled, wt_step, done = env.step(action)
                
                if scheduled:
                    step += 1
                    wt_step_sum += wt_step
                if done:
                    break

            # 最終状態を保存
            map_history['on_premise'].append(env.on_premise_window_history_full.copy())
            map_history['cloud'].append(env.cloud_window_history_full.copy())

            

            # エピソード終了後に objective values を算出
            total_cost, _ = env.calc_objective_values()
            
            # 結果の保存
            results.append([total_cost, wt_step_sum])
            
            # スケジューリングの詳細情報を保存
            detail = {
                'action_set': list(action_set),
                'total_cost': float(total_cost),
                'total_waiting_time': float(wt_step_sum),
                'final_on_premise_map': env.on_premise_window_history_full.tolist(),
                'final_cloud_map': env.cloud_window_history_full.tolist(),
                'map_history': {
                    'on_premise': [m.tolist() for m in map_history['on_premise']],
                    'cloud': [m.tolist() for m in map_history['cloud']]
                }
            }
            scheduling_details.append(detail)
            
            wandb.log({
                "total_cost": total_cost,
                "total_waiting_time": wt_step_sum
            })
            
            print(f"Action set {action_set} -> Cost: {total_cost}, Total Waiting Time: {wt_step_sum}")
        print("map_history['on_premise'][-1]: \n",map_history['on_premise'][-1])
        print("map_history['cloud'][-1]: \n",map_history['cloud'][-1])

        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{self.output_dir}/scheduling_results_{nb_jobs}jobs_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(scheduling_details, f)
        
        # 結果の可視化
        self.results_data = scheduling_details
        self.create_interactive_plot(
            f"scheduling_results_{nb_jobs}jobs_{timestamp}",
            f"全ジョブ数: {nb_jobs}、全組み合わせ数: {len(all_action_sets)}"
        )
        
        return results

    def create_interactive_plot(self, filename, title_text):
        """
        スケジューリング結果をインタラクティブなHTMLプロットとして出力する
        """
        if not self.results_data:
            print("結果データが見つかりません。先に実行評価を行ってください。")
            return
        
        # 結果データからプロット用のデータを抽出
        total_costs = [result['total_cost'] for result in self.results_data]
        total_waiting_times = [result['total_waiting_time'] for result in self.results_data]
        action_sets = [tuple(result['action_set']) for result in self.results_data]
        
        # プロットの作成
        fig = go.Figure()
        
        # カスタムデータとして各点のインデックスを追加
        fig.add_trace(go.Scatter(
            x=total_costs, 
            y=total_waiting_times,
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.7
            ),
            customdata=list(range(len(self.results_data))),
            hovertemplate='<b>コスト:</b> %{x}<br><b>待ち時間:</b> %{y}<br><b>アクションセット:</b> %{text}<extra></extra>',
            text=action_sets
        ))
        
        # プロットの設定
        fig.update_layout(
            title=title_text,
            xaxis_title="総コスト (Total Cost)",
            yaxis_title="総待ち時間 (Total Waiting Time)",
            template="plotly_white",
            height=700,
            width=1000,
            hovermode='closest'
        )
        
        # クリックイベントを処理するJavaScriptコード
        fig.update_layout(
            clickmode='event+select',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # HTML出力の準備
        html_content = self._generate_interactive_html(fig)
        
        # HTMLファイルとして保存
        html_filename = f"{self.output_dir}/{filename}.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"インタラクティブな可視化結果が {html_filename} に保存されました。")
        
        return html_filename
    
    def _generate_interactive_html(self, fig):
        """
        Plotlyグラフをインタラクティブな機能を持つHTMLに変換する
        """
        # PlotlyグラフをHTMLに変換
        plot_div = pio.to_html(fig, include_plotlyjs=True, full_html=False)
        
        # HTMLの詳細データ部分を生成
        detail_data_json = json.dumps(self.results_data)
        
        # HTMLテンプレート
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>スケジューリング結果の可視化</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f7fa;
                }}
                .container {{
                    display: flex;
                    flex-direction: row;
                    width: 100%;
                    min-height: 100vh;
                }}
                .plot-container {{
                    flex: 6;
                    padding: 20px;
                    background-color: white;
                    border-right: 1px solid #e0e0e0;
                }}
                .details-container {{
                    flex: 4;
                    padding: 20px;
                    overflow-y: auto;
                    background-color: #f9f9f9;
                }}
                h2 {{
                    color: #333;
                    border-bottom: 2px solid #4c8bf5;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .detail-box {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    margin-bottom: 20px;
                }}
                .map-container {{
                    display: flex;
                    flex-direction: column;
                    margin-top: 20px;
                }}
                .map-row {{
                    display: flex;
                    flex-direction: row;
                }}
                .map-cell {{
                    width: 25px;
                    height: 25px;
                    border: 1px solid #ddd;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 11px;
                }}
                .map-cell-empty {{
                    background-color: #f0f0f0;
                }}
                .tabs {{
                    display: flex;
                    margin-bottom: 15px;
                }}
                .tab {{
                    padding: 8px 15px;
                    background-color: #e0e0e0;
                    margin-right: 5px;
                    border-radius: 5px 5px 0 0;
                    cursor: pointer;
                }}
                .tab.active {{
                    background-color: #4c8bf5;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .actions-list {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 5px;
                    margin-bottom: 15px;
                }}
                .action-item {{
                    padding: 5px 10px;
                    background-color: #e8f0fe;
                    border-radius: 15px;
                    font-size: 12px;
                }}
                .step-controls {{
                    display: flex;
                    align-items: center;
                    margin-top: 10px;
                    gap: 10px;
                }}
                button {{
                    padding: 8px 12px;
                    background-color: #4c8bf5;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #3a78d4;
                }}
                button:disabled {{
                    background-color: #c0c0c0;
                    cursor: not-allowed;
                }}
                .step-info {{
                    font-size: 14px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="plot-container">
                    {plot_div}
                </div>
                <div class="details-container" id="details">
                    <h2>スケジューリング詳細</h2>
                    <div class="detail-box">
                        <p>左のプロット上の点をクリックすると、そのスケジューリングの詳細が表示されます。</p>
                    </div>
                </div>
            </div>
            
            <script>
                // 結果データをJavaScriptで利用できるようにする
                const schedulingData = {detail_data_json};
                
                // プロットの点がクリックされたときの処理
                document.querySelector('.plotly').on('plotly_click', function(data) {{
                    const point = data.points[0];
                    const pointIndex = point.customdata;
                    const selectedData = schedulingData[pointIndex];
                    
                    // 詳細表示エリアを更新
                    updateDetailsView(selectedData, pointIndex);
                }});
                
                // 詳細表示を更新する関数
                function updateDetailsView(data, index) {{
                    const detailsContainer = document.getElementById('details');
                    
                    // アクションセットをフォーマット
                    const actionSetFormatted = data.action_set.map((a, i) => 
                        `<span class="action-item">ジョブ${{i+1}}: ${{a === 1 ? 'クラウド' : 'オンプレ'}}</span>`
                    ).join('');
                    
                    let content = `
                        <h2>スケジューリング詳細 #${{index + 1}}</h2>
                        <div class="detail-box">
                            <h3>パフォーマンス指標</h3>
                            <p><strong>総コスト:</strong> ${{data.total_cost}}</p>
                            <p><strong>総待ち時間:</strong> ${{data.total_waiting_time}}</p>
                            
                            <h3>アクションセット</h3>
                            <div class="actions-list">
                                ${{actionSetFormatted}}
                            </div>
                            
                            <div class="tabs">
                                <div class="tab active" onclick="showTab('final-map')">最終マップ</div>
                                <div class="tab" onclick="showTab('map-transition')">マップ変遷</div>
                            </div>
                            
                            <div id="final-map" class="tab-content active">
                                <h4>最終スケジューリングマップ</h4>
                                <div class="map-container">
                                    <h5>オンプレミス</h5>
                                    ${{renderMap(data.final_on_premise_map)}}
                                    
                                    <h5>クラウド</h5>
                                    ${{renderMap(data.final_cloud_map)}}
                                </div>
                            </div>
                            
                            <div id="map-transition" class="tab-content">
                                <h4>スケジューリングマップの変遷</h4>
                                <div class="step-controls">
                                    <button id="prevStep" onclick="changeStep(-1)" disabled>前のステップ</button>
                                    <span class="step-info" id="stepInfo">ステップ 1 / ${{data.map_history.on_premise.length}}</span>
                                    <button id="nextStep" onclick="changeStep(1)">次のステップ</button>
                                </div>
                                
                                <div id="transitionMapContainer" class="map-container">
                                    <h5>オンプレミス</h5>
                                    <div id="transitionOnPremMap"></div>
                                    
                                    <h5>クラウド</h5>
                                    <div id="transitionCloudMap"></div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    detailsContainer.innerHTML = content;
                    
                    // 変遷マップの初期表示
                    currentStepIndex = 0;
                    updateTransitionMaps(data);
                }}
                
                // マップをHTMLとして描画する関数
                function renderMap(mapData) {{
                    let html = '<div class="map-container">';
                    
                    for (let row = 0; row < mapData.length; row++) {{
                        html += '<div class="map-row">';
                        
                        for (let col = 0; col < mapData[row].length; col++) {{
                            const value = mapData[row][col];
                            const cellClass = value === -1 ? 'map-cell map-cell-empty' : 'map-cell';
                            const cellText = value === -1 ? '' : value;
                            
                            html += `<div class="${{cellClass}}" title="Row: ${{row}}, Col: ${{col}}, Value: ${{value}}">${{cellText}}</div>`;
                        }}
                        
                        html += '</div>';
                    }}
                    
                    html += '</div>';
                    return html;
                }}
                
                // タブ切り替え関数
                function showTab(tabId) {{
                    // タブのアクティブ状態を更新
                    document.querySelectorAll('.tab').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    document.querySelector(`.tab[onclick="showTab('${{tabId}}')"]`).classList.add('active');
                    
                    // タブコンテンツの表示を更新
                    document.querySelectorAll('.tab-content').forEach(content => {{
                        content.classList.remove('active');
                    }});
                    document.getElementById(tabId).classList.add('active');
                }}
                
                // マップ変遷のためのグローバル変数
                let currentStepIndex = 0;
                let currentDetailData = null;
                
                // ステップを変更する関数
                function changeStep(delta) {{
                    const detailsContainer = document.getElementById('details');
                    const selectedData = JSON.parse(detailsContainer.getAttribute('data-selected'));
                    
                    currentStepIndex += delta;
                    currentStepIndex = Math.max(0, Math.min(selectedData.map_history.on_premise.length - 1, currentStepIndex));
                    
                    updateTransitionMaps(selectedData);
                    
                    // ボタンの有効/無効状態を更新
                    document.getElementById('prevStep').disabled = currentStepIndex === 0;
                    document.getElementById('nextStep').disabled = currentStepIndex === selectedData.map_history.on_premise.length - 1;
                    
                    // ステップ情報を更新
                    document.getElementById('stepInfo').textContent = `ステップ ${{currentStepIndex + 1}} / ${{selectedData.map_history.on_premise.length}}`;
                }}
                
                // 変遷マップを更新する関数
                function updateTransitionMaps(data) {{
                    document.getElementById('details').setAttribute('data-selected', JSON.stringify(data));
                    
                    const onPremMap = data.map_history.on_premise[currentStepIndex];
                    const cloudMap = data.map_history.cloud[currentStepIndex];
                    
                    if (document.getElementById('transitionOnPremMap')) {{
                        document.getElementById('transitionOnPremMap').innerHTML = renderMap(onPremMap);
                        document.getElementById('transitionCloudMap').innerHTML = renderMap(cloudMap);
                    }}
                    
                    // ステップ情報とボタン状態の更新
                    if (document.getElementById('stepInfo')) {{
                        document.getElementById('stepInfo').textContent = `ステップ ${{currentStepIndex + 1}} / ${{data.map_history.on_premise.length}}`;
                        document.getElementById('prevStep').disabled = currentStepIndex === 0;
                        document.getElementById('nextStep').disabled = currentStepIndex === data.map_history.on_premise.length - 1;
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return html_template

if __name__ == "__main__":
    # ジョブ数を設定（注意: 大きな値にすると組合せ数が爆発的に増加します）
    nb_jobs = 15  # 小さい値から始めることをお勧めします (2^5 = 32の組み合わせ)
    
    visualizer = SchedulingVisualizer()
    results = visualizer.run_exhaustive_evaluation(nb_jobs)
    
    print("解析完了。HTMLファイルをブラウザで開いて結果を確認してください。") 