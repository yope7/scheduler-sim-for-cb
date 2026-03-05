from setuptools import setup, Extension
import os
import sys

# pybind11のインポートを試行
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_include
    import pybind11
    PYBIND11_AVAILABLE = True
except ImportError:
    PYBIND11_AVAILABLE = False
    print("エラー: pybind11がインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("  uv sync")
    sys.exit(1)

# コンパイラフラグ（C++としてコンパイル）
extra_compile_args = ['-O3', '-march=native', '-std=c++17']
extra_link_args = []

# デバッグモード（環境変数で制御）
if os.environ.get('DEBUG', '').lower() == 'true':
    extra_compile_args = ['-g', '-O0', '-std=c++17']
    extra_link_args = ['-g']

# 現在のディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Pybind11Extension(
        "scheduling_env_core",
        [
            os.path.join(current_dir, "scheduling_env_bindings.cpp"),
            os.path.join(current_dir, "scheduling_env_core.c"),
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
    name="scheduling_env_core",
    version="0.1.0",
    author="Your Name",
    description="SchedulingEnv C言語実装のPythonバインディング",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    packages=[],  # パッケージは含めない（C拡張のみ）
    py_modules=[],  # Pythonモジュールも含めない
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0",
    ],
)

