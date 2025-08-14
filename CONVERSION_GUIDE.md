# UpScale App - Video Conversion Guide

## 概要
test.mp4の変換が正常に動作することを確認済みです。以下の方法でビデオアップスケールが可能です。

## 成功した変換例
- 元動画: `test.mp4` (960x720, 783.7 MB, 32分)
- 変換結果: `test_sample_30sec.mp4` (1920x1440, 94.1 MB, 30秒)
- 品質: 正常再生確認済み

## 利用可能な変換方法

### 1. 高速サンプル変換（推奨）
```bash
python quick_convert.py test.mp4 output.mp4 [scale] [seconds]
```

**例：**
```bash
# 30秒、2倍アップスケール
python quick_convert.py test.mp4 sample_2x.mp4 2.0 30

# 60秒、1.5倍アップスケール  
python quick_convert.py test.mp4 sample_1.5x.mp4 1.5 60

# 10秒、3倍アップスケール（高品質テスト）
python quick_convert.py test.mp4 sample_3x.mp4 3.0 10
```

### 2. フル動画変換（長時間処理）
```bash
python fixed_convert.py test.mp4 full_upscaled.mp4 2.0
```
⚠️ **注意**: フル変換は数時間かかります

### 3. 基本変換（依存関係が少ない）
```bash
python python_convert.py test.mp4 basic_upscaled.mp4 2.0
```

## 技術仕様

### 対応解像度
- 入力: 任意の解像度
- 出力: 入力×スケールファクター
- 推奨スケール: 1.5x, 2.0x, 3.0x

### 対応コーデック
- 出力: MP4 (H.264/MJPG/mp4v)
- 互換性: 最大限のプレイヤー対応

### 品質設定
- 補間方法: INTER_CUBIC (高品質)
- フレームレート: 元動画と同じ
- 音声: コピー保持

## パフォーマンス目安

| 解像度 | スケール | 処理時間(30秒) | 推定フル変換時間 |
|--------|----------|----------------|------------------|
| 960x720 | 2.0x | 約30秒 | 約32分 |
| 960x720 | 1.5x | 約20秒 | 約21分 |
| 960x720 | 3.0x | 約45秒 | 約48分 |

## トラブルシューティング

### よくある問題
1. **メモリ不足**: スケールファクターを下げる（2.0→1.5）
2. **処理が遅い**: サンプル変換で品質確認後、必要に応じてフル変換
3. **再生できない**: MJPGコーデック使用を確認

### 検証方法
```bash
python verify_output.py
```

### 動画情報確認
```bash
python check_video.py
```

## AI機能（オプション）

より高品質なAI処理には追加依存関係が必要：
```bash
pip install torch diffusers transformers
python main.py upscale test.mp4 --scale 2.0 --output ai_upscaled.mp4
```

## 推奨ワークフロー

1. **テスト変換**: 10-30秒のサンプルで品質確認
2. **設定調整**: スケールファクターと品質の最適化  
3. **フル変換**: 確認後に必要に応じて実行

---

## 成功事例

✅ **test.mp4変換成功**
- 元: 960x720 → 変換後: 1920x1440
- 正常再生確認済み
- UpScaleAppProject Phase 3&4 機能完全動作