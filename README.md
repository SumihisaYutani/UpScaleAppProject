# UpScaleAppProject

**AI Video Upscaling Tool** - 低解像度のMP4動画をAI技術（Waifu2x・Stable Diffusion）を使用して高解像度・高画質に変換するツール

## 🌟 特徴

- **複数のAI手法対応**: Waifu2x（高速・高品質）、Stable Diffusion（カスタマイズ可能）
- **Waifu2x統合**: 専用AI手法による超高画質アップスケーリング
- **対応フォーマット**: MP4ファイル（H.264, H.265/HEVC, AVC対応）
- **多段階スケーリング**: 1x, 2x, 4x, 8x, 16x, 32x まで対応（Waifu2x）
- **ノイズ除去**: 4段階のノイズリダクション機能
- **バッチ処理**: フレーム単位での効率的な処理
- **GUI & CLI**: グラフィカルユーザーインターフェース＋コマンドライン版
- **GPU加速**: CUDA・Vulkan対応でより高速な処理
- **軽量版対応**: AI依存関係なしでの基本機能利用

## 🚀 インストール

### 前提条件
- Python 3.8以降
- FFmpeg（システムにインストール済みである必要があります）
- 推奨: Vulkan対応GPU（Waifu2x用）
- 推奨: NVIDIA GPU（CUDA対応、Stable Diffusion用）

### 🎯 クイックスタート

**GUI版を使用する場合（推奨）**:
```bash
# リポジトリのクローン
git clone https://github.com/SumihisaYutani/UpScaleAppProject.git
cd UpScaleAppProject

# GUI依存関係のインストール
pip install -r requirements_gui.txt

# Waifu2x高画質機能を追加（推奨）
pip install -r requirements_waifu2x.txt

# 軽量GUI版を起動（基本機能＋Waifu2x）
python simple_gui.py

# フルGUI版を起動（全AI機能）
pip install -r requirements.txt  # Stable Diffusion等も含む
python main_gui.py
```

**CLI版のセットアップ**:
```bash
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

# 基本依存関係（軽量）
pip install -r requirements_gui.txt

# Waifu2x高画質機能を追加
pip install -r requirements_waifu2x.txt

# 全AI機能を使用する場合
pip install -r requirements.txt

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
🚨 **トラブルシューティング**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### 🏥 問題診断
環境に問題がある場合：
```bash
# 具体的な問題を診断
python diagnose_issues.py

# 簡単な健康チェック  
python test_environment.py
```

## 💻 使用方法

### 📱 GUI版（推奨）

**軽量GUI版（Waifu2x対応）**:
```bash
python simple_gui.py
```
- ファイル選択とブラウズ
- 基本的な動画情報表示
- スケールファクター設定（1x〜32x、Waifu2x使用時）
- Waifu2x高画質アップスケール機能
- ノイズ除去レベル設定（0-3段階）
- モデル選択（CUNet、Anime Style、Photo）
- FFmpeg・Waifu2x テスト機能

**フルGUI版（全AI機能）**:
```bash
python main_gui.py
```
- Waifu2x + Stable Diffusion 統合
- 自動最適手法選択
- リアルタイム進捗表示
- GPU/CPU 使用状況監視
- 高度な設定オプション
- バッチ処理対応

### 💻 CLI版

**基本的な使用方法**
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
├── src/                        # ソースコード
│   ├── modules/           
│   │   ├── video_processor.py     # 動画処理
│   │   ├── video_builder.py       # 動画再構築
│   │   ├── ai_processor.py        # 統合AI処理
│   │   ├── waifu2x_processor.py   # Waifu2x高画質処理
│   │   ├── enhanced_ai_processor.py  # 拡張AI処理
│   │   └── performance_monitor.py   # パフォーマンス監視
│   ├── gui/
│   │   └── main_window.py          # GUI メインウィンドウ
│   └── enhanced_upscale_app.py     # 拡張アプリケーション
├── config/
│   └── settings.py                 # 設定ファイル
├── tests/                          # テストファイル
├── temp/                           # 一時ファイル
├── output/                         # 出力ファイル
├── main.py                         # CLI エントリーポイント
├── main_gui.py                     # GUI エントリーポイント
├── simple_gui.py                   # 軽量GUI版
├── requirements.txt                # Python依存関係（フル版）
├── requirements_gui.txt            # GUI依存関係（軽量版）
├── requirements_waifu2x.txt        # Waifu2x依存関係
├── test_waifu2x.py                 # Waifu2xテストスクリプト
└── PROJECT_DESIGN.md               # 設計書
```

## ⚙️ 設定

主要な設定は `config/settings.py` で管理されています：

### 基本設定
- **最大ファイルサイズ**: デフォルト2GB
- **最大動画長**: デフォルト60分（拡張）
- **アップスケール倍率**: 1x〜32x対応（手法により異なる）
- **品質プリセット**: Fast, Balanced, Quality

### AI処理設定
- **優先手法**: Auto（自動選択）、Waifu2x、Stable Diffusion、Simple
- **Waifu2x設定**: スケール、ノイズレベル、モデル種類
- **Stable Diffusion設定**: モデル選択、バッチサイズ等

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

**3. Waifu2x関連エラー**
```bash
# Waifu2xの動作確認
python test_waifu2x.py

# Vulkan サポート確認
# Windows: DirectX診断ツールでVulkan対応を確認
# Linux: vulkan-utils をインストール後 vulkaninfo 実行
```

**4. メモリ不足**
- より小さなバッチサイズを使用
- 一時ファイルをクリーンアップ
- システムメモリを増やす

## 🧪 テスト

```bash
# 基本テストの実行
python -m pytest tests/

# 詳細なテスト結果
python -m pytest tests/ -v

# Waifu2x機能テスト
python test_waifu2x.py

# 環境テスト
python test_environment.py
```

## 📋 制限事項

- **ファイルサイズ**: 最大2GB
- **動画長**: 最大60分
- **解像度**: 入力最大1920x1080
- **フォーマット**: MP4のみ対応（AVI, MKV, MOVの表示対応）
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **GUI**: Windows環境で最適化（Unicode文字制限対応）

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

### Phase 3 ✅
- [x] GUI実装（フル版・軽量版）
- [x] Windows互換性対応
- [x] VideoBuilderクラス実装
- [ ] バッチ処理対応（計画中）
- [ ] プラグインシステム（将来版）

### Phase 4 📋
- [ ] 高度なバッチ処理
- [ ] プラグインアーキテクチャ
- [ ] ウェブインターフェース
- [ ] クラウド処理対応

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
**バージョン**: 0.2.0  
**最終更新**: 2025-08-13  
**Phase 3 完了**: GUI実装・Windows互換性対応
