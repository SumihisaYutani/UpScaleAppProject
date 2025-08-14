# UpScale App Project Architecture

このドキュメントはUpScale Appプロジェクトのファイル構成、アーキテクチャ、および処理フローについて詳しく解説します。

## 📁 プロジェクト構造

```
UpScaleAppProject/
├── 📄 メインファイル
│   ├── simple_gui.py          # メインGUIアプリケーション
│   ├── simple_convert.py      # 基本変換スクリプト  
│   ├── main.py               # コマンドライン版メイン
│   └── main_gui.py           # GUI版メイン
│
├── 🔧 設定・環境
│   ├── config/
│   │   └── settings.py       # アプリケーション設定
│   ├── setup_environment.py  # 環境セットアップ
│   ├── install_dependencies.py # 依存関係インストール
│   └── requirements*.txt     # 依存関係定義
│
├── 🎯 コアモジュール (src/)
│   ├── gui/                  # GUI コンポーネント
│   │   ├── main_window.py    # メインウィンドウ
│   │   ├── batch_window.py   # バッチ処理ウィンドウ
│   │   ├── preview_window.py # プレビューウィンドウ
│   │   └── plugin_manager_window.py # プラグイン管理
│   │
│   ├── modules/              # 処理モジュール
│   │   ├── video_processor.py    # 動画処理エンジン
│   │   ├── video_builder.py      # 動画再構築
│   │   ├── waifu2x_processor.py  # Waifu2x統合
│   │   ├── amd_waifu2x_backend.py # AMD GPU対応
│   │   ├── batch_processor.py    # バッチ処理
│   │   ├── ai_processor.py       # AI処理統合
│   │   └── settings_manager.py   # 設定管理
│   │
│   └── plugins/              # プラグインシステム
│       └── plugin_system.py # プラグイン管理
│
├── 📊 出力・ログ
│   ├── output/              # 処理済み動画出力
│   ├── logs/               # アプリケーションログ
│   └── temp/               # 一時ファイル（フレーム画像等）
│
└── 📚 ドキュメント
    ├── README.md           # プロジェクト概要
    ├── CHANGELOG.md        # 変更履歴
    └── docs/              # 詳細ドキュメント
```

## 🚀 アプリケーション処理フロー

```mermaid
graph TD
    A[ユーザー起動] --> B{起動方法選択}
    B -->|GUI| C[simple_gui.py]
    B -->|CLI| D[simple_convert.py]
    
    C --> E[ファイル選択]
    D --> E
    
    E --> F[動画分析]
    F --> G[処理設定]
    G --> H{処理方式選択}
    
    H -->|Waifu2x| I[Waifu2x処理]
    H -->|Simple| J[簡単処理]
    
    I --> K[フレーム抽出]
    J --> K
    
    K --> L[フレーム処理]
    L --> M[動画再構築]
    M --> N[音声合成]
    N --> O[出力完了]
    
    L -.->|キャンセル| P[プロセス終了]
    M -.->|キャンセル| P
    N -.->|キャンセル| P
```

## 🎨 GUI アーキテクチャ

```mermaid
graph LR
    subgraph "GUI Layer"
        A[simple_gui.py<br/>メインGUI]
        B[main_window.py<br/>メインウィンドウ]
        C[batch_window.py<br/>バッチ処理]
        D[preview_window.py<br/>プレビュー]
    end
    
    subgraph "Processing Layer"
        E[video_processor.py<br/>動画処理]
        F[waifu2x_processor.py<br/>AI処理]
        G[batch_processor.py<br/>バッチ処理]
    end
    
    subgraph "Core Layer"
        H[video_builder.py<br/>動画構築]
        I[settings_manager.py<br/>設定管理]
        J[plugin_system.py<br/>プラグイン]
    end
    
    A --> E
    B --> E
    C --> G
    D --> E
    
    E --> H
    F --> H
    G --> H
    
    E --> I
    F --> I
    G --> I
    
    H --> J
```

## ⚙️ 動画処理パイプライン

```mermaid
sequenceDiagram
    participant U as ユーザー
    participant GUI as simple_gui.py
    participant VP as video_processor.py
    participant WP as waifu2x_processor.py
    participant VB as video_builder.py
    
    U->>GUI: 動画選択
    GUI->>VP: 動画分析開始
    VP->>GUI: 動画情報返却
    
    U->>GUI: 処理開始
    GUI->>VP: フレーム抽出開始
    VP->>VP: フレーム保存
    
    VP->>WP: フレーム処理開始
    loop 各フレーム
        WP->>WP: AI upscaling
        WP->>GUI: 進捗更新
    end
    
    WP->>VB: 処理済みフレーム
    VB->>VB: 動画再構築
    VB->>VB: 音声合成
    VB->>GUI: 完了通知
    GUI->>U: 処理完了
```

## 🔧 モジュール間依存関係

```mermaid
graph TB
    subgraph "UI Layer"
        A[simple_gui.py]
        B[GUI Components]
    end
    
    subgraph "Business Logic"
        C[video_processor.py]
        D[waifu2x_processor.py]
        E[batch_processor.py]
    end
    
    subgraph "Core Services"
        F[video_builder.py]
        G[settings_manager.py]
        H[ai_processor.py]
    end
    
    subgraph "External Dependencies"
        I[FFmpeg]
        J[OpenCV]
        K[Waifu2x-ncnn-vulkan]
    end
    
    A --> C
    A --> E
    B --> C
    B --> D
    
    C --> F
    C --> G
    D --> H
    E --> C
    E --> F
    
    F --> I
    C --> J
    H --> K
    D --> K
```

## 📋 主要コンポーネント詳細

### 1. simple_gui.py - メインGUIアプリケーション
- **役割**: ユーザーインターフェースとメイン制御
- **主な機能**:
  - ファイル選択・動画分析
  - 処理設定UI（スケール、品質等）
  - 進捗表示とキャンセル処理
  - FFmpeg/Waifu2xテスト機能

### 2. video_processor.py - 動画処理エンジン
- **役割**: 動画ファイルの分析・フレーム抽出
- **主な機能**:
  - 動画メタデータ取得
  - フレーム単位での分解・保存
  - 処理進捗管理

### 3. waifu2x_processor.py - AI処理統合
- **役割**: Waifu2x AIモデルとの統合
- **主な機能**:
  - 複数バックエンド対応（NCNN-Vulkan等）
  - フレーム単位での高品質アップスケール
  - GPU処理最適化

### 4. video_builder.py - 動画再構築
- **役割**: 処理済みフレームからの動画再構築
- **主な機能**:
  - フレーム結合・動画生成
  - 音声トラック保持・同期
  - FFmpeg統合とプロセス管理

### 5. AMD GPU対応モジュール群
- **amd_gpu_detector.py**: AMD GPU検出
- **amd_waifu2x_backend.py**: AMD用Waifu2xバックエンド
- **amd_vulkan_waifu2x.py**: Vulkan最適化実装

## 🔄 処理フェーズと状態管理

```mermaid
stateDiagram-v2
    [*] --> 初期化
    初期化 --> ファイル選択
    ファイル選択 --> 動画分析
    動画分析 --> 設定調整
    設定調整 --> 処理準備
    
    処理準備 --> フレーム抽出
    フレーム抽出 --> AI処理
    AI処理 --> 動画再構築
    動画再構築 --> 音声合成
    音声合成 --> 完了
    
    AI処理 --> キャンセル : ユーザー操作
    動画再構築 --> キャンセル : ユーザー操作
    音声合成 --> キャンセル : ユーザー操作
    
    キャンセル --> クリーンアップ
    クリーンアップ --> ファイル選択
    完了 --> ファイル選択
```

## 🚨 エラーハンドリングとキャンセル処理

### プロセス管理機能
- **subprocess.Popen()**: プロセスID保持による適切な制御
- **キャンセル機能**: ユーザー操作によるいつでも中断可能
- **プロセス終了**: Windows/Unix対応の強制終了処理
- **リソース管理**: 一時ファイル・メモリの適切なクリーンアップ

### エラー処理フロー
```mermaid
graph TD
    A[処理開始] --> B{エラー発生？}
    B -->|No| C[正常処理継続]
    B -->|Yes| D[エラーキャッチ]
    
    D --> E{エラー種別}
    E -->|ファイルエラー| F[ファイル復旧試行]
    E -->|プロセスエラー| G[プロセス強制終了]
    E -->|メモリエラー| H[メモリクリーンアップ]
    
    F --> I[ユーザー通知]
    G --> I
    H --> I
    
    I --> J[安全な状態に復帰]
```

このアーキテクチャにより、スケーラブルで保守性の高い動画アップスケールアプリケーションを実現しています。各モジュールは独立性を保ちながら、効率的な処理パイプラインを構成しています。