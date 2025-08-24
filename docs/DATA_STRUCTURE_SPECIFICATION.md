# UpScaleApp データ構造仕様書

## 1. 概要

### 1.1 目的
UpScaleAppのレジューム機能における全データ構造の詳細仕様を定義する。

### 1.2 データ格納形式
- **永続化**: JSON形式でファイルシステムに保存
- **メモリ**: Python辞書オブジェクト
- **文字エンコード**: UTF-8
- **日時形式**: ISO 8601 (YYYY-MM-DDTHH:MM:SS)

## 2. 主要データ構造

### 2.1 Session オブジェクト
レジューム機能の中核となるセッション情報

```typescript
interface Session {
    // === 基本情報 ===
    session_id: string;           // MD5ハッシュ（32文字）
    video_file: string;           // 動画ファイル絶対パス
    created_at: string;           // セッション作成日時
    last_updated: string;         // 最終更新日時
    status: SessionStatus;        // セッション全体のステータス
    
    // === 動画情報 ===
    video_info: VideoInfo;
    
    // === 処理設定 ===
    settings: ProcessingSettings;
    
    // === 処理ステップ ===
    steps: {
        validate: ValidateStep;
        extract: ExtractStep;
        upscale: UpscaleStep;
        combine: CombineStep;
    };
    
    // === メタデータ ===
    metadata?: SessionMetadata;
}
```

### 2.2 VideoInfo オブジェクト
動画ファイルのメタ情報

```typescript
interface VideoInfo {
    // === 基本情報 ===
    file_path: string;            // ファイルパス
    file_size: number;            // ファイルサイズ（バイト）
    file_modified: number;        // 最終更新時刻（Unix timestamp）
    
    // === 動画特性 ===
    duration: number;             // 再生時間（秒）
    frame_count: number;          // 総フレーム数
    fps: number;                  // フレームレート
    width: number;                // 幅（ピクセル）
    height: number;               // 高さ（ピクセル）
    
    // === コーデック情報 ===
    video_codec: string;          // 動画コーデック
    audio_codec?: string;         // 音声コーデック
    bitrate: number;              // ビットレート
    
    // === 追加情報 ===
    format: string;               // ファイル形式
    has_audio: boolean;           // 音声トラック有無
    color_space?: string;         // 色空間
    aspect_ratio: string;         // アスペクト比
}
```

### 2.3 ProcessingSettings オブジェクト
処理設定パラメータ

```typescript
interface ProcessingSettings {
    // === アップスケール設定 ===
    scale_factor: number;         // 拡大率（1.0-4.0）
    quality: QualityLevel;        // 品質レベル
    noise_reduction: NoiseLevel;  // ノイズ除去レベル
    
    // === AI バックエンド ===
    backend: AIBackend;           // AI処理バックエンド
    backend_version?: string;     // バックエンドバージョン
    
    // === 処理オプション ===
    parallel_processing: boolean; // 並列処理有効
    gpu_acceleration: boolean;    // GPU加速有効
    max_threads?: number;         // 最大スレッド数
    batch_size?: number;          // バッチサイズ
    
    // === 出力設定 ===
    output_format: string;        // 出力形式
    output_quality: number;       // 出力品質（0-100）
    preserve_audio: boolean;      // 音声保持
    
    // === フレーム処理 ===
    frame_format: string;         // フレーム形式（PNG/JPG）
    temp_compression: boolean;    // 一時ファイル圧縮
}

// 列挙型定義
type QualityLevel = "Draft" | "Quality" | "High Quality";
type NoiseLevel = 0 | 1 | 2 | 3;
type AIBackend = "real_cugan" | "waifu2x_python" | "waifu2x_executable" | "simple";
```

### 2.4 Processing Step オブジェクト

#### 2.4.1 BaseStep（基底クラス）
```typescript
interface BaseStep {
    // === 基本ステータス ===
    status: StepStatus;           // ステップ状態
    progress: number;             // 進捗率（0-100）
    
    // === 時刻管理 ===
    start_time?: string;          // 開始時刻
    end_time?: string;            // 終了時刻
    duration?: number;            // 処理時間（秒）
    
    // === エラー管理 ===
    error?: ErrorInfo;            // エラー情報
    retry_count: number;          // リトライ回数
    
    // === 統計情報 ===
    statistics?: StepStatistics;  // 処理統計
}

type StepStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled" | "paused";
```

#### 2.4.2 ValidateStep
```typescript
interface ValidateStep extends BaseStep {
    // === 検証結果 ===
    file_exists: boolean;         // ファイル存在確認
    file_readable: boolean;       // ファイル読み取り可能
    format_supported: boolean;    // 形式サポート
    metadata_valid: boolean;      // メタデータ有効性
    
    // === 検証詳細 ===
    validation_details: {
        ffprobe_output?: string;  // FFprobe出力
        format_analysis: FormatAnalysis;
        compatibility_check: CompatibilityCheck;
    };
    
    // === 推定値 ===
    estimated_processing_time: number; // 推定処理時間（秒）
    estimated_output_size: number;     // 推定出力サイズ（バイト）
}

interface FormatAnalysis {
    container_format: string;
    video_streams: number;
    audio_streams: number;
    subtitle_streams: number;
    chapters: number;
}

interface CompatibilityCheck {
    ffmpeg_compatible: boolean;
    backend_compatible: boolean;
    gpu_compatible: boolean;
    warnings: string[];
}
```

#### 2.4.3 ExtractStep
```typescript
interface ExtractStep extends BaseStep {
    // === フレーム管理 ===
    total_frames: number;         // 総フレーム数
    extracted_frames: number;     // 抽出済みフレーム数
    failed_extractions: number;   // 抽出失敗数
    
    // === バッチ処理 ===
    batch_size: number;           // バッチサイズ
    total_batches: number;        // 総バッチ数
    completed_batches: number[];  // 完了済みバッチ
    current_batch: number;        // 現在処理中バッチ
    
    // === 処理統計 ===
    extraction_rate: number;      // 抽出レート（frames/sec）
    average_frame_size: number;   // 平均フレームサイズ（バイト）
    total_extracted_size: number; // 総抽出サイズ（バイト）
    
    // === 品質情報 ===
    frame_quality_stats: FrameQualityStats;
    
    // === エラー詳細 ===
    batch_errors: BatchError[];   // バッチ別エラー
}

interface FrameQualityStats {
    min_size: number;             // 最小フレームサイズ
    max_size: number;             // 最大フレームサイズ
    avg_size: number;             // 平均フレームサイズ
    std_deviation: number;        // 標準偏差
    corrupted_frames: string[];   // 破損フレーム一覧
}

interface BatchError {
    batch_id: number;
    error_message: string;
    failed_frames: number[];
    timestamp: string;
    recovery_attempted: boolean;
}
```

#### 2.4.4 UpscaleStep
```typescript
interface UpscaleStep extends BaseStep {
    // === フレーム処理 ===
    total_frames: number;         // 総フレーム数
    completed_frames: string[];   // 完了フレーム一覧
    failed_frames: FailedFrame[]; // 失敗フレーム詳細
    skipped_frames: string[];     // スキップフレーム一覧
    
    // === 並列処理 ===
    worker_count: number;         // ワーカー数
    active_workers: number;       // アクティブワーカー数
    worker_statistics: WorkerStats[];
    
    // === 処理統計 ===
    processing_rate: number;      // 処理レート（frames/sec）
    average_processing_time: number; // 平均処理時間/フレーム（秒）
    gpu_utilization?: number;     // GPU使用率（%）
    memory_usage: MemoryUsage;    // メモリ使用量
    
    // === 品質管理 ===
    quality_metrics: QualityMetrics;
    
    // === バックエンド情報 ===
    backend_info: BackendInfo;
}

interface FailedFrame {
    frame_path: string;
    error_message: string;
    error_code?: string;
    retry_count: number;
    last_attempt: string;
    recovery_possible: boolean;
}

interface WorkerStats {
    worker_id: number;
    processed_frames: number;
    processing_time: number;
    average_time_per_frame: number;
    error_count: number;
    current_frame?: string;
}

interface MemoryUsage {
    peak_memory_mb: number;
    current_memory_mb: number;
    memory_efficiency: number;    // MB per frame
    gc_collections: number;
}

interface QualityMetrics {
    upscale_ratio_actual: number; // 実際の拡大率
    image_quality_score?: number; // 画質スコア
    processing_artifacts: number; // アーティファクト数
    quality_consistency: number;  // 品質一貫性（0-1）
}

interface BackendInfo {
    name: string;
    version: string;
    model_used: string;
    gpu_device?: string;
    optimization_level: string;
}
```

#### 2.4.5 CombineStep
```typescript
interface CombineStep extends BaseStep {
    // === 出力情報 ===
    output_path?: string;         // 出力ファイルパス
    output_size?: number;         // 出力ファイルサイズ
    output_duration?: number;     // 出力動画長（秒）
    
    // === エンコード設定 ===
    encoder_used: string;         // 使用エンコーダー
    encoding_preset: string;      // エンコードプリセット
    bitrate_target: number;       // 目標ビットレート
    bitrate_actual?: number;      // 実際のビットレート
    
    // === 処理統計 ===
    encoding_speed: number;       // エンコード速度（x倍速）
    compression_ratio: number;    // 圧縮率
    
    // === 音声処理 ===
    audio_processing: AudioProcessing;
    
    // === 品質情報 ===
    output_quality_metrics: OutputQualityMetrics;
}

interface AudioProcessing {
    has_audio: boolean;
    audio_codec: string;
    audio_bitrate: number;
    audio_channels: number;
    audio_sample_rate: number;
    audio_synchronized: boolean;
    lip_sync_offset?: number;     // 音声同期オフセット（ms）
}

interface OutputQualityMetrics {
    psnr?: number;                // Peak Signal-to-Noise Ratio
    ssim?: number;                // Structural Similarity Index
    vmaf?: number;                // Video Multimethod Assessment Fusion
    file_integrity_verified: boolean;
}
```

## 3. エラー管理データ構造

### 3.1 ErrorInfo オブジェクト
```typescript
interface ErrorInfo {
    // === エラー基本情報 ===
    error_code: string;           // エラーコード
    error_message: string;        // エラーメッセージ
    error_type: ErrorType;        // エラー種別
    
    // === 発生情報 ===
    timestamp: string;            // 発生日時
    step_name: string;            // 発生ステップ
    frame_context?: string;       // 関連フレーム
    
    // === 技術詳細 ===
    stack_trace?: string;         // スタックトレース
    system_info: SystemInfo;      // システム情報
    
    // === 復旧情報 ===
    recoverable: boolean;         // 復旧可能性
    suggested_action?: string;    // 推奨アクション
    retry_after?: number;         // リトライ推奨時間（秒）
}

type ErrorType = "system" | "user" | "network" | "hardware" | "software" | "configuration";

interface SystemInfo {
    platform: string;            // OS情報
    python_version: string;       // Pythonバージョン
    memory_available_mb: number;  // 利用可能メモリ
    disk_space_available_gb: number; // 利用可能ディスク容量
    gpu_memory_mb?: number;       // GPU メモリ
    cpu_usage_percent: number;    // CPU使用率
}
```

## 4. 統計データ構造

### 4.1 SessionMetadata オブジェクト
```typescript
interface SessionMetadata {
    // === 処理統計 ===
    total_processing_time: number; // 総処理時間（秒）
    actual_vs_estimated_ratio: number; // 実績/推定時間比
    
    // === リソース使用量 ===
    peak_memory_usage_mb: number;
    peak_disk_usage_gb: number;
    total_cpu_time: number;       // 総CPU時間（秒）
    total_gpu_time?: number;      // 総GPU時間（秒）
    
    // === 品質指標 ===
    overall_quality_score?: number;
    user_satisfaction?: number;   // ユーザー満足度（1-5）
    
    // === 効率指標 ===
    frames_per_second_average: number;
    mb_per_second_throughput: number;
    efficiency_score: number;     // 効率スコア（0-1）
    
    // === 中断・再開統計 ===
    interruption_count: number;  // 中断回数
    resume_success_rate: number; // 再開成功率
    data_recovery_count: number; // データ復旧回数
}
```

### 4.2 StepStatistics オブジェクト
```typescript
interface StepStatistics {
    // === 時間統計 ===
    min_processing_time: number;
    max_processing_time: number;
    avg_processing_time: number;
    std_dev_processing_time: number;
    
    // === スループット ===
    throughput_items_per_second: number;
    throughput_mb_per_second: number;
    
    // === エラー統計 ===
    error_rate: number;           // エラー率（0-1）
    retry_success_rate: number;   // リトライ成功率（0-1）
    
    // === リソース統計 ===
    avg_memory_usage_mb: number;
    peak_memory_usage_mb: number;
    avg_cpu_usage_percent: number;
    avg_gpu_usage_percent?: number;
}
```

## 5. 設定・管理データ構造

### 5.1 SessionConfig オブジェクト
システム全体の設定

```typescript
interface SessionConfig {
    // === ストレージ設定 ===
    max_sessions: number;         // 最大セッション数
    max_session_age_days: number; // セッション最大保持日数
    max_disk_usage_gb: number;    // 最大ディスク使用量
    
    // === パフォーマンス設定 ===
    auto_cleanup_enabled: boolean;
    progress_save_interval_sec: number;
    memory_cleanup_threshold_mb: number;
    
    // === セキュリティ設定 ===
    session_encryption_enabled: boolean;
    log_sensitive_data: boolean;
    
    // === 復旧設定 ===
    auto_recovery_enabled: boolean;
    max_auto_retry_count: number;
    corruption_detection_enabled: boolean;
}
```

### 5.2 SessionIndex オブジェクト
セッション管理用インデックス

```typescript
interface SessionIndex {
    // === インデックス情報 ===
    index_version: string;
    last_updated: string;
    
    // === セッション一覧 ===
    sessions: SessionSummary[];
    
    // === 統計情報 ===
    total_sessions: number;
    active_sessions: number;
    completed_sessions: number;
    failed_sessions: number;
    
    // === クリーンアップ情報 ===
    last_cleanup: string;
    cleanup_statistics: CleanupStats;
}

interface SessionSummary {
    session_id: string;
    video_file: string;
    status: SessionStatus;
    created_at: string;
    last_updated: string;
    progress_percent: number;
    estimated_time_remaining?: number;
    file_size_mb: number;
}

interface CleanupStats {
    sessions_cleaned: number;
    disk_space_freed_gb: number;
    last_cleanup_duration_sec: number;
}
```

## 6. バリデーション仕様

### 6.1 データ整合性チェック
```python
def validate_session_data(session: Session) -> ValidationResult:
    """セッションデータの整合性を検証"""
    errors = []
    warnings = []
    
    # 基本フィールド検証
    if not session.session_id or len(session.session_id) != 32:
        errors.append("Invalid session_id format")
    
    # 日時形式検証
    try:
        datetime.fromisoformat(session.created_at)
        datetime.fromisoformat(session.last_updated)
    except ValueError:
        errors.append("Invalid datetime format")
    
    # ファイル存在確認
    if not Path(session.video_file).exists():
        warnings.append("Video file not found")
    
    # ステップ整合性確認
    for step_name, step_data in session.steps.items():
        step_errors = validate_step_data(step_name, step_data)
        errors.extend(step_errors)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

### 6.2 スキーマバージョニング
```python
# データ構造バージョン管理
SCHEMA_VERSIONS = {
    "1.0": "初期バージョン",
    "1.1": "統計情報追加", 
    "1.2": "エラーハンドリング強化",
    "2.0": "並列処理対応"
}

def migrate_session_data(session_data: dict, from_version: str, to_version: str):
    """セッションデータのマイグレーション"""
    if from_version == "1.0" and to_version == "1.1":
        # 統計情報フィールド追加
        session_data["metadata"] = create_default_metadata()
    
    # その他のマイグレーション処理...
```

---

## 変更履歴
- v1.0: 初版作成 (2024-08-24)
- v1.1: エラー管理データ構造追加
- v1.2: 統計・メタデータ構造詳細化