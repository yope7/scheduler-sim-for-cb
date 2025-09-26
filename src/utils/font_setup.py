import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_linux_fonts():
    """Linuxで使用可能なフォントを自動設定"""
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
    else:
        print(f"現在のシステム: {platform.system()}. Linux以外のシステムではフォント設定をスキップします。")

def get_available_fonts():
    """利用可能なフォントのリストを取得"""
    return [f.name for f in fm.fontManager.ttflist]

def list_preferred_fonts():
    """優先フォントのリストを表示"""
    preferred_fonts = [
        'DejaVu Sans',
        'Liberation Sans',
        'Ubuntu',
        'Noto Sans CJK JP',
        'Noto Sans',
        'Arial',
        'Helvetica'
    ]
    
    available_fonts = get_available_fonts()
    print("優先フォントの利用可能性:")
    for font in preferred_fonts:
        status = "✓" if font in available_fonts else "✗"
        print(f"  {status} {font}")
    
    return preferred_fonts 