# Claude Code Settings

## 言語設定
- **使用言語**: 日本語で対話してください

## プロジェクト情報
- **プロジェクト名**: UpScale App Project
- **言語**: Python (GUI: CustomTkinter, 動画処理: FFmpeg)
- **目的**: 動画のアップスケーリングアプリケーション

## ビルドコマンド
```bash
cd executable
python build_gpu_version.py
```

## テストコマンド
```bash
cd executable
python main.py
```

## 実行ファイル
- **場所**: `executable/dist/UpScaleApp_GPU.exe`
- **ビルドツール**: PyInstaller
- **サイズ**: 703.53MB (最適化版)

## ログファイル場所
- **パス**: `C:\Users\Yutani Sumihisa\UpScaleApp_Logs\`
- **形式**: `upscale_app_YYYYMMDD_HHMMSS.log`

## セッション管理
- **一時フォルダ**: `C:\tmp\upscale_app_sessions\`
- **セッションファイル**: `progress.json`
- **フレーム保存**: `frames/` サブフォルダ

## 重要な修正履歴
- 再開機能のフリーズ問題を修正 (gui.py:2810-2816)
- ビデオ検証タイムアウト追加 (video_processor.py)
- Real-CUGAN・Real-ESRGAN TTA無効化で50%高速化 (v2.3.0)
- スレッド設定最適化とGUI選択機能追加 (real_cugan_backend.py, real_esrgan_backend.py)
- ビルドプロセス最適化とキャッシュ機能追加 (build_gpu_version.py)