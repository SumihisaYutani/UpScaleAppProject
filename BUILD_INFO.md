# UpScale App - ビルド情報

## 現在のビルド先

**最新ビルド**: `D:\ClaudeCode\project\UpScaleAppProject\executable\dist\UpScaleApp_GPU.exe`

## ディストリビューション構成

```
UpScaleApp_GPU.exe                       # 単一実行ファイル (481.70 MB)
                                        # 全依存関係を内包：
                                        # ├── ffmpeg.exe (動画処理)
                                        # ├── ffprobe.exe (動画情報取得)
                                        # ├── waifu2x-ncnn-vulkan.exe (AI画像処理)
                                        # ├── waifu2x_ncnn_py/ (AIモデル)
                                        # ├── torch/ (PyTorchライブラリ)
                                        # ├── numpy/ (数値計算)
                                        # ├── cv2/ (OpenCV)
                                        # ├── customtkinter/ (GUI)
                                        # └── その他の依存関係
```

## サイズ情報

- **単一実行ファイル**: 481.70 MB
- **追加ファイル**: 不要
- **依存関係含む完全版**: ✅

## ビルドコマンド

```bash
cd "D:\ClaudeCode\project\UpScaleAppProject\executable"
python build_gpu_version.py
```

## 必要なバイナリ

以下のバイナリが`resources/binaries/`に含まれている必要があります：

1. **FFmpeg** (`C:\ffmpeg\bin\ffmpeg.exe`)
2. **FFprobe** (`C:\ffmpeg\bin\ffprobe.exe`)  
3. **Waifu2x** (`tools\waifu2x-ncnn-vulkan\waifu2x-ncnn-vulkan-20220728-windows\waifu2x-ncnn-vulkan.exe`)

## GPU加速機能

### ✅ GPU検出・対応状況
- **AMD Radeon RX Vega 56**: ✅ Vulkan対応・Real-CUGAN GPU加速動作確認済み
- **NVIDIA GPU**: ✅ CUDA・Vulkan対応 (自動検出)
- **Intel GPU**: ✅ 統合GPU検出対応
- **Vulkan API**: ✅ Real-CUGAN NCNN-Vulkan バックエンド対応

### ✅ AI処理GPU加速
- **Real-CUGAN**: ✅ Vulkan GPU加速 (anime/illustration特化)
- **Waifu2x**: ✅ NCNN-Vulkan GPU加速 (汎用超解像)
- **GPU自動選択**: ✅ 最適バックエンド自動検出

### ⚠️ フレーム抽出最適化状況
- **CPU フレーム抽出**: ✅ 安定動作・確実性重視
- **AMD D3D11VA**: ⚠️ 互換性問題 (ドライバー/FFmpeg組み合わせ依存)
- **最適構成**: CPU抽出 + GPU AI処理 (ハイブリッド最適化)

## 最終更新

- **日時**: 2025-08-24 21:30 (GPU検出・フレーム抽出最適化・Real-CUGAN完全統合)
- **バージョン**: v2.2.1 - GPU加速AI完全版 + Vulkan対応
- **ビルド形式**: PyInstaller --onefile (単一実行ファイル)
- **ステータス**: ✅ **GPU検出・AI処理完全動作・配布準備完了**

## 完成機能

### ✅ Real-CUGAN統合 (新機能)
- **Real-CUGAN NCNN-Vulkan**: anime/illustration特化AI超解像
- **AMD Radeon RX Vega対応**: Vulkan GPU加速確認済み
- **モデル選択**: conservative/denoise1x/denoise2x/denoise3x対応
- **デフォルト設定**: 2x scale, Real-CUGAN, denoise2x (バランス最適)

### ✅ GUI改良・最適化
- **スケール選択**: 2.0x/4.0x/8.0x (実用的な倍率に変更)
- **動的システム情報**: AIプロセッサー選択時リアルタイム更新
- **GPU検出表示**: Vulkan/AMD/NVIDIA対応状況を詳細表示
- **日本語完全対応**: 全メニュー・メッセージの日本語化

### ✅ 高速並列処理・安定性
- **3ワーカー並列実行**: ThreadPoolExecutor最適化
- **GPU同期処理**: threading.Lock()によるリソース管理
- **レジューム機能**: 中断処理の完全再開対応
- **エラーハンドリング**: GPU検出・処理失敗時の適切な処理
- **詳細ログ**: GPU状態監視・性能計測・デバッグ情報

## ログファイルの保存場所

**実行時ログ**: タイムスタンプ付きログファイルが自動作成されます

- **保存先**: `{実行ディレクトリ}/logs/upscale_app_{YYYYMMDD_HHMMSS}.log`
- **例**: `logs/upscale_app_20250824_213000.log`
- **内容**: 
  - GPU検出詳細 (Vulkan/AMD/NVIDIA対応状況)
  - Real-CUGAN初期化・GPU加速確認
  - AI処理の実行状況・GPU使用率
  - フレーム抽出方式 (CPU/GPU選択理由)
  - エラー詳細・GPU互換性問題
  - 性能計測・処理時間分析

**ログファイルのアクセス**:
- GUIの「ログファイルを開く」ボタンから直接表示可能
- 実行後もファイルは保持され、後から分析可能

## GPU加速に関する技術情報

### ✅ 動作確認済み構成
- **OS**: Windows 10/11
- **GPU**: AMD Radeon RX Vega 56 (Vulkan対応)
- **ドライバー**: AMD Adrenalin最新版推奨
- **GPU加速**: Real-CUGAN NCNN-Vulkan完全対応

### ⚠️ 既知の制限事項
- **AMD D3D11VA フレーム抽出**: FFmpegとの互換性問題あり
  - 症状: `scale_nv12 filter not found`、フォーマット変換エラー
  - 対処: CPU抽出使用で安定性確保 (AI処理はGPU加速継続)
- **Vulkan検出**: 一部環境でvulkaninfoが利用不可
  - 対処: GPU検出によるVulkan対応推定で代替

### 🔧 トラブルシューティング
- **GPU検出されない**: AMD/NVIDIAドライバー更新
- **Vulkan未対応表示**: 最新グラフィックドライバーインストール
- **AI処理が遅い**: GPU温度・メモリ使用率確認

## 注意事項

- ビルド先が変更されると依存関係が取りこぼされる可能性があります
- 必ず`resources/binaries/`に必要なバイナリが含まれていることを確認してください
- 総サイズが800MB+であることを確認して、依存関係の不足を防いでください
- GPU加速は環境依存のため、CPU処理への自動フォールバック機能を搭載