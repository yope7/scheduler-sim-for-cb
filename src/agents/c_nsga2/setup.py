from setuptools import setup, Extension
import os
import sys

# pybind11のインポートを試行
# PEP 517ビルドでは、pyproject.tomlでビルド依存関係が指定されているため、
# ここでインポートできるはずです
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_include
    import pybind11
except ImportError as e:
    # ビルド時にはpyproject.tomlで依存関係が解決されるはずですが、
    # 念のためエラーメッセージを表示
    print(f"エラー: pybind11のインポートに失敗しました: {e}")
    print("以下のコマンドでインストールしてください:")
    print("  pip install pybind11")
    print("または、pyproject.tomlでビルド依存関係が正しく設定されているか確認してください。")
    # PEP 517ビルドでは、setup.pyが実行される前に依存関係がインストールされるため、
    # ここでsys.exit(1)を呼ぶとビルドが失敗します
    # 代わりに、エラーを再発生させてビルドシステムに処理させます
    raise

# コンパイラフラグ（C++としてコンパイル）
extra_compile_args = ['-O3', '-march=native', '-std=c++17', '-fopenmp']
extra_link_args = ['-fopenmp']

# デバッグモード（環境変数で制御）
if os.environ.get('DEBUG', '').lower() == 'true':
    extra_compile_args = ['-g', '-O0', '-std=c++17']
    extra_link_args = ['-g']

# 現在のディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Pybind11Extension(
        "nsga2_core",
        [
            os.path.join(current_dir, "nsga2_bindings.cpp"),
            os.path.join(current_dir, "nsga2_core.c"),
        ],
        include_dirs=[
            current_dir,
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="nsga2_core",
    version="0.1.0",
    description="NSGA-II C言語実装のPythonバインディング",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    packages=[],
    py_modules=[],
    # install_requiresはpyproject.tomlで指定されているため、ここでは指定しない
)

