# 🔧 途中再開機能 - トラブルシューティング

## 🚨 一般的な問題と解決法

### 問題1: セッションが検出されない

#### 症状
- 以前処理を中断したのに再開ダイアログが表示されない
- 同じ動画ファイルを選択してもセッションが見つからない

#### 原因と対処法

**📁 ファイルパスの変更**
```
原因: 動画ファイルを移動した
対処法: 
1. 動画ファイルを元の場所に戻す
2. または、新しい場所で「最初から開始」を選択
```

**⚙️ 設定の変更**
```
原因: 拡大率や品質設定が前回と異なる
対処法:
1. 前回と同じ設定（拡大率・品質）を選択
2. 設定が不明の場合は「最初から開始」を選択
```

**🗂️ セッションファイルの削除**
```
原因: 一時ファイルが削除された
確認方法:
1. %TEMP%/upscale_app_sessions/ フォルダを確認
2. セッションフォルダが存在するかチェック

対処法: 新規セッションで開始
```

### 問題2: 処理が途中で停止する

#### 症状
- AIアップスケーリング中にアプリケーションが応答しなくなる
- プログレスバーが進まない

#### 原因と対処法

**💾 ディスク容量不足**
```
確認方法:
- セッションディレクトリのサイズをチェック
- C:ドライブの空き容量を確認

対処法:
1. 不要ファイルを削除して容量を確保
2. より大容量のドライブにセッションを移動
3. 一時的に他のプログラムを終了
```

**🔥 GPU過熱・メモリ不足**
```
症状: 
- GPU使用率が急激に下がる
- システムが重くなる

対処法:
1. GPU温度を確認（80℃以下推奨）
2. GPU使用率をタスクマネージャーで監視
3. 他のGPU使用プログラムを終了
4. システムを再起動してから再開
```

**⚡ 電源・システム設定**
```
確認項目:
□ 自動スリープが無効になっているか
□ 電源プランが「高パフォーマンス」になっているか
□ Windows Updateが実行中でないか

設定方法:
1. 電源オプション → 「高パフォーマンス」選択
2. スリープ設定 → 「なし」
3. Windows Update → 「更新の一時停止」
```

### 問題3: フレーム数の不一致

#### 症状
- 「期待フレーム数: 46764, 実際: 46750」のようなエラー
- 一部フレームが処理されない

#### 原因と対処法

**🎬 動画フォーマットの問題**
```
対処法:
1. 動画を再エンコードしてフレーム数を統一
   ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4

2. 「最初から開始」を選択して新規処理
```

**📸 フレーム抽出エラー**
```
確認方法:
- セッション/frames/ フォルダ内のファイル数をチェック
- ログウィンドウでFFmpegエラーを確認

対処法:
1. frames/ フォルダを削除
2. 「最初から開始」で再処理
```

### 問題4: 再開ダイアログが応答しない

#### 症状
- 再開ダイアログのボタンが効かない
- ダイアログが閉じられない

#### 対処法

**🔄 アプリケーション再起動**
```
手順:
1. タスクマネージャーでUpScaleAppプロセスを確認
2. 必要に応じて強制終了
3. アプリケーションを再起動
4. セッションが自動検出されることを確認
```

**🖱️ UI応答性の改善**
```
確認事項:
- システムリソース使用率
- 他の重いアプリケーションの終了
- メモリ使用量の確認

対処法:
1. 不要なプログラムを終了
2. 「詳細ログウィンドウ」を閉じる
3. シンプルな操作で再開を試行
```

## 🔍 詳細診断方法

### セッション状態の確認

#### 進行状況ファイルの確認
```bash
# セッションディレクトリの表示
dir %TEMP%\upscale_app_sessions

# 進行状況ファイルの内容確認
type %TEMP%\upscale_app_sessions\[session_id]\progress.json
```

#### ファイル数の確認
```bash
# 抽出フレーム数の確認
dir /b %TEMP%\upscale_app_sessions\[session_id]\frames | find /c "frame_"

# アップスケール済みフレーム数の確認  
dir /b %TEMP%\upscale_app_sessions\[session_id]\upscaled | find /c "_upscaled.png"
```

### ログ分析

#### アプリケーションログの確認
```
重要なログパターン:

[成功]
- "Created new session: abc123def456"
- "User chose to resume session abc123def456"  
- "Session completed successfully and cleaned up"

[警告]
- "Found resumable session for sample.mp4"
- "Frame count discrepancy: extracted 1000, expected 1001"
- "High memory usage detected (2000.0 MB)"

[エラー]
- "Failed to load progress for session abc123def456"
- "Invalid video: Validation error"
- "Processing failed progressdialog object has no attribute root"
```

#### 詳細ログウィンドウの活用
```
確認項目:
□ GPU使用状況（100% ↔ 20%の変動パターン）
□ フレーム処理速度（1.5秒/フレーム程度が正常）
□ メモリ使用量（2GB以下推奨）
□ FFmpegコマンドの成功・失敗
□ Waifu2xの戻り値（return code: 0が正常）
```

## 🛠️ 手動修復方法

### セッションの手動クリーンアップ

#### 破損セッションの削除
```bash
# 特定セッションの削除
rmdir /s "%TEMP%\upscale_app_sessions\[破損session_id]"

# 古いセッション全体の削除（注意: 全データ消失）
rmdir /s "%TEMP%\upscale_app_sessions"
```

#### 部分的な修復
```bash
# frames/ フォルダのみ削除（フレーム抽出をやり直し）
rmdir /s "%TEMP%\upscale_app_sessions\[session_id]\frames"

# upscaled/ フォルダのみ削除（AIアップスケーリングをやり直し）  
rmdir /s "%TEMP%\upscale_app_sessions\[session_id]\upscaled"
```

### 進行状況ファイルの手動編集

#### ステップ状態のリセット
```json
{
  "steps": {
    "extract": {
      "status": "pending",    // "completed" → "pending" に変更
      "progress": 0
    },
    "upscale": {
      "status": "pending",
      "progress": 0,
      "completed_frames": []  // 配列をクリア
    }
  }
}
```

## 🔋 パフォーマンス最適化

### システム設定の最適化

#### Windows電源設定
```
推奨設定:
1. コントロールパネル → 電源オプション
2. 「高パフォーマンス」または「バランス」選択
3. プラン設定の変更:
   - ディスプレイの電源を切る: なし
   - コンピューターをスリープ状態にする: なし
```

#### GPUドライバー最適化
```
AMD GPU (Radeon RX Vega):
1. AMD Radeon Software を最新版に更新
2. Vulkan ランタイムの確認・更新
3. GPU温度監視の有効化

NVIDIA GPU:
1. GeForce Experience/NVIDIA Control Panel更新
2. CUDAランタイムの確認
3. GPU温度・クロック監視
```

### リソース監視コマンド

#### リアルタイム監視
```bash
# GPU使用率の監視
wmic path win32_VideoController get name,CurrentHorizontalResolution,CurrentVerticalResolution

# メモリ使用量の監視
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value

# ディスク使用量の監視
wmic logicaldisk get size,freespace,caption
```

## 📞 サポート情報収集

### 問題報告時の必要情報

#### システム情報
```
□ OS バージョン: Windows 10/11
□ CPU: Intel/AMD型番
□ GPU: NVIDIA/AMD型番とドライバーバージョン
□ メモリ容量: XXXth
□ 利用可能ディスク容量: XXXGB
```

#### エラー詳細
```
□ エラーメッセージの正確な文言
□ エラー発生時の処理ステップ（validation/extract/upscale/combine）
□ 処理進行率（X.X%）
□ 動画ファイル情報（解像度、長さ、フレーム数）
□ 使用設定（拡大率、品質）
```

#### ログファイル
```
提供ファイル:
1. 詳細ログウィンドウの内容（スクリーンショット）
2. progress.json ファイル（個人情報削除後）
3. セッションディレクトリの構造（dir出力）
```

### 緊急回避方法

#### 処理の完全リセット
```
手順:
1. アプリケーション終了
2. %TEMP%\upscale_app_sessions フォルダを削除  
3. アプリケーション再起動
4. 「最初から開始」で新規処理
```

#### 外部ツールでの処理継続
```
FFmpeg直接実行（上級者向け）:
1. セッションのframes/フォルダから残存フレームを特定
2. waifu2x-ncnn-vulkanで個別処理
3. FFmpegで手動結合

注意: この方法は技術的知識が必要です
```

問題が解決しない場合は、GitHubのIssueページで詳細情報と共にお問い合わせください。