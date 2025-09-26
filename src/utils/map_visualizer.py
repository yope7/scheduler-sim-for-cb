import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import glob

# Linuxで使用可能なフォントを自動設定
def setup_linux_fonts():
    """Linuxで使用可能なフォントを自動設定"""
    import matplotlib.font_manager as fm
    import platform
    
    # Linuxシステムでのみ実行
    if platform.system() == 'Linux':
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 優先順位付きでフォントを選択
        preferred_fonts = [
            'DejaVu Sans',
            'Liberation Sans',
            'Ubuntu',
            'Noto Sans CJK JP',
            'Noto Sans',
            'Arial',
            'Helvetica'
        ]
        
        # 利用可能なフォントから選択
        selected_font = None
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # フォントが見つからない場合は、利用可能なフォントから最初のものを使用
        if selected_font is None and available_fonts:
            # sans-serif系のフォントを優先
            sans_serif_fonts = [f for f in available_fonts if 'sans' in f.lower() or 'sans' in f.lower()]
            if sans_serif_fonts:
                selected_font = sans_serif_fonts[0]
            else:
                selected_font = available_fonts[0]
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['font.sans-serif'] = [selected_font]
            print(f"フォントを設定しました: {selected_font}")
        else:
            print("警告: 利用可能なフォントが見つかりませんでした")

# Linuxフォントを自動設定
# setup_linux_fonts()
#行列をprintしたときに省略しないためのオプション
np.set_printoptions(threshold=np.inf)

# def clear_directory(directory):
#     files = glob.glob(os.path.join(directory, '*'))
#     for file in files:
#         os.remove(file)

def get_unique_filename(base_name):
    """重複しないファイル名を生成する"""
    counter = 1
    new_name = base_name
    while os.path.exists("./maps/" + new_name + ".png"):
        new_name = f"{base_name}_{counter}"
        counter += 1
    return new_name

def visualize_map(on_premise_map, cloud_map, job_list, name):
    """スケジューリング結果を色付けして可視化する"""
    name = get_unique_filename(name)
    # 各ジョブに異なる色を割り当てるためのカラーマップを作成
    base_cmap = plt.get_cmap('tab20')  # 基本のカラーマップ
    # 0から9までの色を使用し、-1は黒に設定
    colors = ['black'] + [base_cmap(i % base_cmap.N) for i in range(10)]
    cmap = ListedColormap(colors)
    # -1を特別に扱い、0から9の範囲で色分け
    norm = BoundaryNorm(np.arange(-1.5, 10.5, 1), cmap.N)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # オンプレミスのマップを表示
    im0 = ax[0].imshow(np.where(on_premise_map == -1, -1, on_premise_map % 10), cmap=cmap, norm=norm, interpolation='nearest')  # 1の位で色分け
    ax[0].set_title('On-Premise Map')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Node')

    # ク���ウドのマップを表示
    im1 = ax[1].imshow(np.where(cloud_map == -1, -1, cloud_map % 10), cmap=cmap, norm=norm, interpolation='nearest')  # 1の位で色分け
    ax[1].set_title('Cloud Map')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Node')

    # 各セルに値を表示
    for i in range(on_premise_map.shape[0]):
        for j in range(on_premise_map.shape[1]):
            ax[0].text(j, i, str(on_premise_map[i, j]), ha='center', va='center', color='white')
            ax[1].text(j, i, str(cloud_map[i, j]), ha='center', va='center', color='white')

    # 凡例を表示
    ax_legend = fig.add_subplot(111, frameon=False)
    ax_legend.axis('off')
    # ジョブリストを凡例として追加
    handles = [plt.Line2D([0], [0], color=base_cmap(i % base_cmap.N), lw=4) for i in range(len(job_list))]
    labels = [f"Job Size: {job['size']}" for job in job_list]
    ax_legend.legend(handles, labels, loc='center', ncol=4, fontsize='small', frameon=False)

    plt.tight_layout()
    plt.savefig("./maps/" + name + ".png")
    # plt.close()
if __name__ == "__main__":
    # テスト用のダミーデータ
    on_premise_map = np.random.randint(0, 5, (10, 20))
    cloud_map = np.random.randint(0, 5, (10, 20))
    job_list = [{'size': (2, 3)}, {'size': (3, 2)}, {'size': (1, 4)}]  # ダミーのジョブリスト
    visualize_map(on_premise_map, cloud_map, job_list) 