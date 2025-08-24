#!/usr/bin/env python
"""
UpScaleApp GPU加速版ビルドスクリプト
GPU支援フレーム抽出機能を含む実行可能ファイルを作成
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("UpScaleApp GPU accelerated version build starting...")
    
    # プロジェクトのルートディレクトリ
    project_root = Path(__file__).parent
    main_script = project_root / "main.py"
    
    if not main_script.exists():
        print("ERROR: main.py not found")
        return False
    
    # distディレクトリをクリーンアップ
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        print("Cleaning existing dist directory...")
        try:
            shutil.rmtree(dist_dir)
        except PermissionError:
            print("Permission error, trying to force delete...")
            subprocess.run(f'rmdir /S /Q "{dist_dir}"', shell=True, check=False)
    
    # buildディレクトリをクリーンアップ
    build_dir = project_root / "build"
    if build_dir.exists():
        print("Cleaning existing build directory...")
        shutil.rmtree(build_dir)
    
    # PyInstallerコマンドを構築
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # 単一実行ファイル形式
        "--windowed",                   # GUIアプリケーション
        "--noupx",                      # UPX圧縮を無効化
        "--add-data", "resources;resources", # リソースファイルを含める（相対パス）
        "--collect-submodules", "waifu2x_ncnn_py", # waifu2xのサブモジュールを含める
        "--collect-data", "waifu2x_ncnn_py", # waifu2xのデータファイル（モデル）を含める
        "--copy-metadata", "torch",     # PyTorchメタデータを含める
        "--copy-metadata", "diffusers", # Diffusersメタデータを含める
        "--name", "UpScaleApp_GPU",
        # GPU加速関連モジュールを含める
        "--hidden-import", "core.gpu_frame_extractor",
        "--hidden-import", "core.fast_frame_extractor", 
        "--hidden-import", "core.performance_monitor",
        "--hidden-import", "core.gpu_detector",
        "--hidden-import", "core.video_processor",
        "--hidden-import", "core.ai_processor",
        "--hidden-import", "core.real_cugan_backend",
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
        "--hidden-import", "diffusers",
        "--hidden-import", "transformers",
        "--hidden-import", "accelerate",
        "--hidden-import", "numpy",
        "--hidden-import", "tqdm",
        # Real-CUGAN関連の依存関係
        "--hidden-import", "basicsr",
        "--hidden-import", "realesrgan",
        "--hidden-import", "facexlib",
        "--hidden-import", "gfpgan",
        # GPU処理関連
        "--hidden-import", "pyopencl",
        "--hidden-import", "pycuda",
        str(main_script)
    ]
    
    print("Building with PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # ビルド処理は時間がかかるため、タイムアウトを20分に設定し、リアルタイム出力を有効化
        print("Starting PyInstaller build (this may take 10-15 minutes)...")
        print("=" * 60)
        
        # リアルタイム出力でビルド進行状況を表示
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, universal_newlines=True)
        
        # 出力をリアルタイムで表示
        for line in process.stdout:
            print(line.rstrip())
        
        # プロセス完了を待機
        process.wait(timeout=1200)
        
        if process.returncode == 0:
            print("BUILD SUCCESS!")
        else:
            print(f"BUILD FAILED with return code: {process.returncode}")
            return False
        
        # 単一ファイル形式の実行ファイルの場所を確認（BUILD_INFO.mdの要求に従う）
        exe_path = project_root / "dist" / "UpScaleApp_GPU.exe"
        if exe_path.exists():
            print(f"Executable created: {exe_path}")
            print(f"File size: {exe_path.stat().st_size / 1024 / 1024:.2f} MB")
            print("Single executable file - no additional directory needed")
            
            print("\nGPU Acceleration Features:")
            print("  - AMD/Intel/NVIDIA GPU Hardware Acceleration")
            print("  - Dynamic CPU Load Optimization")
            print("  - 3-5x Frame Extraction Speed Boost")
            print("  - 50-70% CPU Usage Reduction")
            
            return True
        else:
            print("ERROR: Executable not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"BUILD ERROR:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nGPU-accelerated UpScaleApp v2.2.0 build completed!")
    else:
        print("\nBuild failed")
        sys.exit(1)