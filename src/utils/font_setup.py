import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from typing import Optional
import os

def _font_supports_text(font_path: str, text: str) -> bool:
    """指定フォントがtext内の文字を概ね描画可能かをチェック（Matplotlib内部のFT2Fontを使用）"""
    try:
        # TTCはFT2Fontで扱えない環境があるため、ここでは保守的に「OK」とする
        # （実際の描画はMatplotlib側で問題なく行えることが多い）
        if font_path.lower().endswith(".ttc"):
            return True
        from matplotlib.ft2font import FT2Font  # pylint: disable=import-error
        ft = FT2Font(font_path)
        cmap = ft.get_charmap()
        for ch in text:
            if ch == "\n" or ch == " ":
                continue
            if ord(ch) not in cmap:
                return False
        return True
    except Exception:
        return False

def setup_japanese_font(required: bool = False) -> Optional[str]:
    """
    日本語ラベルを描けるフォントを優先的に設定する。
    - required=True の場合、見つからなければ例外を投げる（日本語のみのグラフを保証したい時用）
    Returns:
        selected_font_family (見つからなければNone)
    """
    # Linux以外でも動くようにする（フォント探索はOS依存だが、無いならNone）
    preferred_families = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "VL Gothic",
        "Yu Gothic",
        "Hiragino Sans",
    ]

    # この文字列が描ければ今回のプロットはまず大丈夫、という最小セット
    must_support = "待ち時間 コスト 利用率 しきい値"

    # family名→font pathの候補リストを作る（同一familyが複数pathを持つことがある）
    family_to_paths: dict[str, list[str]] = {}
    for f in fm.fontManager.ttflist:
        family_to_paths.setdefault(f.name, []).append(f.fname)

    # Matplotlibが.ttcを拾わない環境があるので、代表的な日本語フォントを明示的に追加してから再探索する
    known_font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    added_any = False
    for p in known_font_paths:
        try:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                added_any = True
        except Exception:
            pass

    if added_any:
        family_to_paths = {}
        for f in fm.fontManager.ttflist:
            family_to_paths.setdefault(f.name, []).append(f.fname)

    for fam in preferred_families:
        paths = family_to_paths.get(fam, [])
        for p in paths:
            if _font_supports_text(p, must_support):
                plt.rcParams["font.family"] = fam
                plt.rcParams["font.sans-serif"] = [fam]
                return fam

    if required:
        raise RuntimeError(
            "日本語フォントが見つからないため、日本語ラベルのみのグラフを生成できません。"
            "Linuxなら例: `sudo apt-get install -y fonts-noto-cjk` などで日本語フォントを入れてから再実行してください。"
        )
    return None

def setup_linux_fonts():
    """Linuxで使用可能なフォントを自動設定"""
    # Linuxシステムでのみ実行
    if platform.system() == 'Linux':
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 優先順位付きでフォントを選択
        preferred_fonts = [
            'Noto Sans CJK JP',
            'Noto Sans JP',
            'IPAexGothic',
            'IPAGothic',
            'TakaoGothic',
            'VL Gothic',
            'DejaVu Sans',
            'Liberation Sans',
            'Ubuntu',
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
            sans_serif_fonts = [f for f in available_fonts if 'sans' in f.lower()]
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