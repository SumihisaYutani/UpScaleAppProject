# UpScaleAppProject

🎬 **AI Video Upscaling Tool** - 低解像度のMP4動画をAI技術（Stable Diffusion）を使用して高解像度・高画質に変換するツール

## 🌟 特徴

- **AI駆動のアップスケール**: Stable Diffusionを使用した高品質な画質向上
- **対応フォーマット**: MP4ファイル（H.264, H.265/HEVC, AVC対応）
- **1.5倍アップスケール**: 標準的な1.5倍の解像度向上
- **バッチ処理**: フレーム単位での効率的な処理
- **CLI & GUI**: コマンドライン版（将来的にGUI対応予定）
- **GPU加速**: CUDA対応でより高速な処理

## 🚀 インストール

### 前提条件
- Python 3.8以降
- FFmpeg（システムにインストール済みである必要があります）
- 推奨: NVIDIA GPU（CUDA対応）

### 🎯 クイックスタート
```bash
# リポジトリのクローン
git clone https://github.com/SumihisaYutani/UpScaleAppProject.git
cd UpScaleAppProject

# 自動環境テスト
python quick_test.py

# 自動セットアップ（推奨）
python setup_environment.py
```

### 🔧 手動セットアップ
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 段階的依存関係インストール
python install_dependencies.py

# 環境テスト
python test_environment.py
```

### ⚠️ トラブルシューティング
Pythonの実行でエラーが出る場合：

**Windows:**
```cmd
# 異なるPythonコマンドを試行
python --version
py --version
python3 --version

# バッチファイルを使用
run_test.bat
```

**Linux/macOS:**
```bash
# シェルスクリプトを使用
./run_test.sh
```

📚 **詳細な環境設定ガイド**: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

## 💻 使用方法

### 基本的な使用方法
```bash
# 動画をアップスケール（AI使用）
python main.py upscale input_video.mp4

# カスタムスケールファクター指定
python main.py upscale input_video.mp4 --scale 2.0

# 出力ファイル名を指定
python main.py upscale input_video.mp4 --output upscaled_video.mp4

# AIを使わずに単純なアップスケール
python main.py upscale input_video.mp4 --no-ai
```

### 🚀 拡張機能（Phase 2）
```bash
# 拡張CLIでシステム監視付き処理
python main_enhanced.py upscale input_video.mp4 --show-system-stats

# 品質プリセット指定
python main_enhanced.py upscale input_video.mp4 --quality-preset quality

# エラー回復機能付き処理
python main_enhanced.py upscale input_video.mp4 --max-retries 5

# 処理ログの分析
python main_enhanced.py logs --last-n 10

# 詳細システム情報とレポート保存
python main_enhanced.py system --save-report
```

### その他のコマンド
```bash
# 動画ファイル情報を表示
python main.py info input_video.mp4

# プレビュー作成（短い動画で確認）
python main.py preview input_video.mp4

# システム情報を表示
python main.py system

# 設定を表示
python main.py config

# ヘルプ
python main.py --help
```

## 🏗️ プロジェクト構造

```
UpScaleAppProject/
├── src/                    # ソースコード
│   ├── modules/           
│   │   ├── video_processor.py  # 動画処理
│   │   ├── video_builder.py    # 動画再構築
│   │   └── ai_processor.py     # AI処理
│   └── upscale_app.py          # メインアプリケーション
├── config/
│   └── settings.py             # 設定ファイル
├── tests/                      # テストファイル
├── temp/                       # 一時ファイル
├── output/                     # 出力ファイル
├── main.py                     # CLI エントリーポイント
├── requirements.txt            # Python依存関係
└── PROJECT_DESIGN.md           # 設計書
```

## ⚙️ 設定

主要な設定は `config/settings.py` で管理されています：

- **最大ファイルサイズ**: デフォルト2GB
- **最大動画長**: デフォルト30分
- **アップスケール倍率**: デフォルト1.5倍
- **AI処理設定**: モデル選択、バッチサイズ等

## 🔧 トラブルシューティング

### よくある問題

**1. FFmpegが見つからない**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

**2. CUDA関連エラー**
```bash
# CUDAツールキットがインストールされているか確認
python -c "import torch; print(torch.cuda.is_available())"
```

**3. メモリ不足**
- より小さなバッチサイズを使用
- 一時ファイルをクリーンアップ
- システムメモリを増やす

## 🧪 テスト

```bash
# 基本テストの実行
python -m pytest tests/

# 詳細なテスト結果
python -m pytest tests/ -v
```

## 📋 制限事項

- **ファイルサイズ**: 最大2GB
- **動画長**: 最大30分
- **解像度**: 入力最大1920x1080
- **フォーマット**: MP4のみ対応
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## 🛣️ ロードマップ

### Phase 1 ✅
- [x] 基本機能実装
- [x] MP4ファイル処理
- [x] CLI インターフェース

### Phase 2 ✅  
- [x] AI統合とテスト
- [x] パフォーマンス最適化
- [x] エラーハンドリング強化
- [x] 拡張AI処理モジュール
- [x] リアルタイム性能監視
- [x] 包括的エラー回復機能

### Phase 3 📋
- [ ] GUI実装
- [ ] バッチ処理対応
- [ ] プラグインシステム

## 🤝 コントリビュート

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 謝辞

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - AI画像生成
- [FFmpeg](https://ffmpeg.org/) - 動画処理
- [OpenCV](https://opencv.org/) - コンピュータビジョン
- [Hugging Face](https://huggingface.co/) - ML モデルとライブラリ

---

**作成者**: SumihisaYutani  
**バージョン**: 0.1.0  
**最終更新**: 2025-08-13
