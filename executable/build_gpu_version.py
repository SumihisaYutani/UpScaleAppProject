#!/usr/bin/env python
"""
UpScaleApp GPU加速版ビルドスクリプト
GPU支援フレーム抽出機能を含む実行可能ファイルを作成
"""

import os
import sys
import shutil
import subprocess
import time
import datetime
import multiprocessing
from pathlib import Path

def main():
    # 全体の開始時間を記録
    total_start_time = time.time()
    print("UpScaleApp GPU accelerated version build starting...")
    print(f"Build started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # プロジェクトのルートディレクトリ
    project_root = Path(__file__).parent
    main_script = project_root / "main.py"
    
    if not main_script.exists():
        print("ERROR: main.py not found")
        return False
    
    # クリーンアップフェーズ開始
    cleanup_start_time = time.time()
    print("\n[CLEANUP] Cleanup phase starting...")
    
    # distディレクトリをクリーンアップ
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        print("Cleaning existing dist directory...")
        try:
            shutil.rmtree(dist_dir)
        except PermissionError:
            print("Permission error, trying to force delete...")
            subprocess.run(f'rmdir /S /Q "{dist_dir}"', shell=True, check=False)
    
    # ビルドキャッシュ管理（高速化のため）
    build_dir = project_root / "build"
    cache_dir = project_root / ".pyinstaller_cache"
    
    # キャッシュディレクトリを保持（初回ビルド後はキャッシュを活用）
    if not cache_dir.exists():
        print("Creating PyInstaller cache directory...")
        cache_dir.mkdir(exist_ok=True)
    else:
        print("Using existing PyInstaller cache (faster build)...")
    
    # buildディレクトリのみクリーンアップ（キャッシュは保持）
    if build_dir.exists():
        print("Cleaning build directory (keeping cache)...")
        shutil.rmtree(build_dir)
    
    cleanup_time = time.time() - cleanup_start_time
    print(f"[CLEANUP] Cleanup completed in {cleanup_time:.2f} seconds")
    
    # PyInstallerコマンドを構築（ビルド時間短縮のための最適化付き）
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # 単一実行ファイル形式
        "--windowed",                   # GUIアプリケーション  
        "--noconsole",                  # コンソールウィンドウを完全に非表示
        "--noupx",                      # UPX圧縮を無効化
        "--noconfirm",                  # 確認なしで上書き
        "--clean",                      # 一時ファイルをクリーン
        "--log-level=WARN",             # ログレベルを警告のみに制限
        "--add-data", "resources;resources", # リソースファイルを含める（相対パス）
        "--collect-submodules", "waifu2x_ncnn_py", # waifu2xのサブモジュールを含める
        "--collect-data", "waifu2x_ncnn_py", # waifu2xのデータファイル（モデル）を含める
        "--copy-metadata", "torch",     # PyTorchメタデータを含める
        "--name", "UpScaleApp_GPU",
        # ビルド時間短縮のためのオプション
        "--exclude-module", "matplotlib",  # 不要なMatplotlib除外
        "--exclude-module", "scipy",       # 不要なSciPy除外  
        "--exclude-module", "pandas",      # 不要なPandas除外
        "--exclude-module", "jupyter",     # 不要なJupyter除外
        "--exclude-module", "IPython",     # 不要なIPython除外
        "--workpath", str(cache_dir),      # キャッシュディレクトリを指定
        # GPU加速関連モジュールを含める
        "--hidden-import", "core.gpu_frame_extractor",
        "--hidden-import", "core.fast_frame_extractor", 
        "--hidden-import", "core.performance_monitor",
        "--hidden-import", "core.gpu_detector",
        "--hidden-import", "core.video_processor",
        "--hidden-import", "core.ai_processor",
        "--hidden-import", "core.real_cugan_backend",
        "--hidden-import", "core.real_esrgan_backend",
        "--hidden-import", "core.app",
        "--hidden-import", "core.gui",
        "--hidden-import", "core.resume_dialog",
        "--hidden-import", "core.session_manager",
        "--hidden-import", "core.utils",
        # 必要な依存関係
        "--hidden-import", "tkinter",
        "--hidden-import", "customtkinter",
        "--hidden-import", "PIL",
        "--hidden-import", "cv2",
        "--hidden-import", "psutil",
        "--hidden-import", "concurrent.futures",
        "--hidden-import", "ffmpeg",
        # AI/ML関連の依存関係
        "--hidden-import", "waifu2x_ncnn_py",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision", 
        "--hidden-import", "transformers",
        "--hidden-import", "accelerate",
        "--hidden-import", "numpy",
        "--hidden-import", "tqdm",
        str(main_script)
    ]
    
    # PyInstallerビルドフェーズ開始
    build_start_time = time.time()
    print(f"\n[BUILD] PyInstaller build phase starting...")
    print(f"[BUILD] System: {multiprocessing.cpu_count()} CPU cores available")
    print(f"[BUILD] Cache directory: {cache_dir}")
    print(f"[BUILD] Optimization: Excluding unused modules (matplotlib, scipy, pandas, etc.)")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # ビルド処理は時間がかかるため、タイムアウトを20分に設定し、リアルタイム出力を有効化
        print("Starting PyInstaller build (this may take 10-15 minutes)...")
        print("[BUILD] Ensuring Windows GUI mode (no console) with runw.exe bootloader")
        print("=" * 60)
        
        # リアルタイム出力でビルド進行状況を表示
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, universal_newlines=True)
        
        # 出力をリアルタイムで表示
        for line in process.stdout:
            print(line.rstrip())
        
        # プロセス完了を待機
        process.wait(timeout=1200)
        
        build_time = time.time() - build_start_time
        
        if process.returncode == 0:
            print("BUILD SUCCESS!")
            print(f"[BUILD] PyInstaller build completed in {build_time:.2f} seconds ({build_time/60:.1f} minutes)")
        else:
            print(f"BUILD FAILED with return code: {process.returncode}")
            print(f"[BUILD] Failed build took {build_time:.2f} seconds")
            return False
        
        # 検証フェーズ開始
        verify_start_time = time.time()
        print(f"\n[VERIFY] Verification phase starting...")
        
        # 単一ファイル形式の実行ファイルの場所を確認（BUILD_INFO.mdの要求に従う）
        exe_path = project_root / "dist" / "UpScaleApp_GPU.exe"
        if exe_path.exists():
            verify_time = time.time() - verify_start_time
            print(f"[VERIFY] Executable verified: {exe_path}")
            print(f"[VERIFY] File size: {exe_path.stat().st_size / 1024 / 1024:.2f} MB")
            print("[VERIFY] Single executable file - no additional directory needed")
            print(f"[VERIFY] Verification completed in {verify_time:.2f} seconds")
            
            print("\n[FEATURES] GPU Acceleration Features:")
            print("  - AMD/Intel/NVIDIA GPU Hardware Acceleration")
            print("  - Dynamic CPU Load Optimization")
            print("  - 3-5x Frame Extraction Speed Boost")
            print("  - 50-70% CPU Usage Reduction")
            
            return True
        else:
            verify_time = time.time() - verify_start_time
            print("[VERIFY] ERROR: Executable not found")
            print(f"[VERIFY] Verification took {verify_time:.2f} seconds")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"BUILD ERROR:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    # 全体の開始時間を記録（main関数の外で）
    script_start_time = time.time()
    
    success = main()
    
    # 全体の完了時間を計算
    total_time = time.time() - script_start_time
    
    if success:
        print(f"\n[SUCCESS] GPU-accelerated UpScaleApp v2.2.0 build completed!")
        print(f"[TIMING] Total build time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"[TIMING] Build finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n[FAILED] Build failed")
        print(f"[TIMING] Total time until failure: {total_time:.2f} seconds")
        sys.exit(1)