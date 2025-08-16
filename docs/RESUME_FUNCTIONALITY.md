# 🔄 途中再開機能ガイド

UpScale Appの途中再開機能により、長時間の動画処理が中断されても安全に再開できます。

## 🌟 概要

途中再開機能は、AIによる動画アップスケーリング処理（数時間〜数十時間）を中断から安全に復旧する機能です。

### 主な利点
- ⏱️ **時間節約**: 最大19時間の処理時間を節約
- 💾 **進捗保存**: フレーム単位での正確な進行状況管理
- 🔄 **自動検出**: 同じ動画・設定での再開セッション自動検出
- 🛡️ **安全性**: エラー耐性とデータ損失防止

## 📊 対応処理ステップ

| ステップ | 状態管理 | 再開方法 |
|---------|---------|---------|
| 🔍 **動画検証** | 完了/失敗 | スキップ/再実行 |
| 📸 **フレーム抽出** | 完了/失敗 | スキップ/再実行 |
| 🤖 **AI処理** | フレーム単位 | 未処理フレームのみ継続 |
| 🎬 **動画結合** | 完了/失敗 | スキップ/再実行 |

## 🚀 使用方法

### 1. 自動検出
```
1. アプリケーションを起動
2. 動画ファイルを選択（Browseボタン）
3. 前回の未完了セッションが検出された場合、再開ダイアログが表示
```

### 2. 再開ダイアログ

<img src="../assets/resume_dialog_example.png" alt="再開ダイアログ例" width="600"/>

#### 表示情報:
- **📹 動画ファイル名**
- **📅 最終更新日時**
- **⚙️ 処理設定**（拡大率、品質）
- **📊 各ステップの進行状況**

#### 選択肢:
- **🚀 途中から再開**: 前回の続きから処理開始
- **🔄 最初から開始**: 新規セッションで全工程実行
- **❌ キャンセル**: 処理を中止

### 3. フレーム単位再開

AIアップスケーリング段階では、フレーム単位での正確な再開が可能です：

```
処理済み: 1,200/46,764 フレーム (2.6%)
残り: 45,564 フレーム
推定時間短縮: 約18.2時間
```

## 🔧 技術仕様

### セッション管理

#### セッションID生成
```python
# 動画ファイル + 設定のハッシュで一意ID生成
session_data = {
    'video_path': '/path/to/video.mp4',
    'video_size': 1234567890,
    'video_mtime': 1642345678.9,
    'scale_factor': 2.0,
    'quality': 'Quality'
}
session_id = hashlib.md5(json.dumps(session_data).encode()).hexdigest()[:12]
```

#### ディレクトリ構造
```
%TEMP%/upscale_app_sessions/[session_id]/
├── progress.json          # 進行状況データ
├── frames/               # 抽出フレーム
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── upscaled/            # アップスケール済み
│   ├── frame_000001_upscaled.png
│   ├── frame_000002_upscaled.png
│   └── ...
└── logs/                # 処理ログ
    └── session.log
```

### 進行状況データ構造

```json
{
  "session_id": "abc123def456",
  "video_file": "C:/Videos/sample.mp4",
  "video_info": {
    "width": 960,
    "height": 720,
    "frame_count": 46764,
    "duration": 1950.1,
    "frame_rate": 23.98
  },
  "settings": {
    "scale_factor": 2.0,
    "quality": "Quality",
    "noise_reduction": 3
  },
  "created_at": "2025-01-16T20:40:18",
  "last_updated": "2025-01-16T21:13:32",
  "steps": {
    "validate": {
      "status": "completed",
      "progress": 100,
      "start_time": "2025-01-16T20:40:18",
      "end_time": "2025-01-16T20:40:19"
    },
    "extract": {
      "status": "completed", 
      "progress": 100,
      "extracted_frames": 46764,
      "start_time": "2025-01-16T20:40:19",
      "end_time": "2025-01-16T21:13:30"
    },
    "upscale": {
      "status": "in_progress",
      "progress": 2.6,
      "total_frames": 46764,
      "completed_frames": [
        "/temp/session/upscaled/frame_000001_upscaled.png",
        "/temp/session/upscaled/frame_000002_upscaled.png",
        "..."
      ],
      "failed_frames": [],
      "start_time": "2025-01-16T21:13:32"
    },
    "combine": {
      "status": "pending",
      "progress": 0
    }
  }
}
```

## 🛠️ 高度な設定

### セッション自動クリーンアップ

```python
# 7日以上古いセッションを自動削除
session_manager.cleanup_old_sessions(max_age_days=7)
```

### 手動セッション管理

```python
# 特定セッションの削除
session_manager.cleanup_session(session_id)

# 全セッション確認
sessions = session_manager.get_all_resumable_sessions()
```

## 📝 ベストプラクティス

### 1. 処理前の確認事項
- ✅ 十分なディスク容量（動画サイズの3-5倍推奨）
- ✅ 安定した電源環境
- ✅ システムの自動スリープ無効化

### 2. 中断時の対応
- 🔄 アプリケーションを再起動
- 📁 同じ動画ファイルを再選択
- ⚙️ 同じ処理設定を確認
- 🚀 再開ダイアログで「途中から再開」を選択

### 3. トラブルシューティング

#### セッションが検出されない場合
```
原因: 
- 動画ファイルパスの変更
- 処理設定の変更
- セッションファイルの破損

対処法:
- 動画ファイルを元の場所に戻す
- 同じ処理設定を使用
- 最初から開始を選択
```

#### ディスク容量不足
```
症状: 処理が途中で停止
対処法:
- 不要ファイルの削除
- より大容量のドライブへ移動
- バッチサイズの削減
```

## 🔍 ログとデバッグ

### セッションログの確認
```bash
# セッションディレクトリの確認
ls %TEMP%/upscale_app_sessions/

# 進行状況の確認
cat %TEMP%/upscale_app_sessions/[session_id]/progress.json

# ログの確認
tail -f %TEMP%/upscale_app_sessions/[session_id]/logs/session.log
```

### デバッグ情報
アプリケーションの「詳細ログウィンドウ」で以下の情報を確認できます：
- セッション作成・再開ログ
- フレーム処理進捗
- エラー詳細情報
- GPU使用状況

## 📈 パフォーマンス効果

### 時間節約例

| 動画長さ | 総フレーム数 | 中断地点 | 節約時間 |
|---------|-------------|---------|---------|
| 32分 | 46,764 | 2.6% | 18.2時間 |
| 60分 | 90,000 | 50% | 9.5時間 |
| 120分 | 180,000 | 75% | 4.8時間 |

### リソース効率
- 💾 **ストレージ**: 処理済みフレームの再利用
- 🔋 **電力**: 重複処理の回避
- 🖥️ **GPU**: 最適化された負荷分散

## 🆕 アップデート履歴

### v2.1.0 (2025-01-16)
- ✨ 途中再開機能の実装
- 📊 セッション管理システム
- 🎯 フレーム単位精密再開
- 🎨 再開ダイアログUI

### 予定機能
- 📱 モバイル通知連携
- ☁️ クラウドセッション同期
- 🤖 AI予測による最適再開点提案