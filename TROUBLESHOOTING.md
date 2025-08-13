# 🚨 トラブルシューティングガイド

## ❌ POOR Status (<40%) - 重大な問題

環境テストでPOORステータスが表示される場合、以下の重大な問題が検出されています：

### 🔴 Critical Issues (重大な問題)

#### 1. Python環境の根本的な問題
```
❌ Python installation not found or severely broken
❌ Core Python modules (sys, os, pathlib) not working
❌ Working directory or project structure corrupted
```

**症状:**
- `python` コマンドが全く動作しない
- 基本的なPythonモジュールのインポートに失敗
- プロジェクトディレクトリが破損している

**解決策:**
```bash
# Python再インストール
# Windows: https://python.org からダウンロード
# Linux: sudo apt install python3 python3-pip
# macOS: brew install python3

# プロジェクトの再クローン
git clone https://github.com/SumihisaYutani/UpScaleAppProject.git
```

#### 2. プロジェクト構造の破損
```
❌ Missing critical files: src/, config/, main.py
❌ Configuration import completely failed
❌ Core modules cannot be loaded
```

**症状:**
- 必須ファイル/ディレクトリが存在しない
- `config.settings` のインポートが失敗
- すべてのモジュールインポートが失敗

**解決策:**
```bash
# プロジェクトの完全な再取得
rm -rf UpScaleAppProject
git clone https://github.com/SumihisaYutani/UpScaleAppProject.git
cd UpScaleAppProject
python quick_test.py
```

#### 3. 権限・アクセスの問題
```
❌ Permission denied on project directories
❌ Cannot create temporary directories
❌ File system access completely blocked
```

**症状:**
- ディレクトリの読み書きができない
- temp/, output/ ディレクトリが作成できない
- ファイルの実行権限がない

**解決策:**
```bash
# Windows: 管理者権限で実行
# Linux/macOS: 権限の修正
chmod -R 755 UpScaleAppProject/
sudo chown -R $USER:$USER UpScaleAppProject/
```

### 🟠 Severe Issues (深刻な問題)

#### 4. 依存関係の完全な不足
```
❌ No packages available (PIL, numpy, click, etc.)
❌ pip installation completely broken
❌ Internet connectivity issues blocking downloads
```

**診断:**
```bash
python -c "import sys; print(sys.path)"
python -m pip --version
python -c "import PIL, numpy" # Should fail
```

**解決策:**
```bash
# pip修復
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# 基本パッケージの強制インストール
python -m pip install --user Pillow numpy click tqdm
```

#### 5. システムレベルの制限
```
❌ Corporate firewall blocking package downloads
❌ Antivirus software interfering with Python
❌ System-wide Python policy restrictions
```

**回避策:**
```bash
# オフラインインストール用パッケージの準備
pip download -r requirements_minimal.txt -d packages/
pip install --find-links packages/ --no-index -r requirements_minimal.txt

# 企業環境での代替
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_minimal.txt
```

### 📊 POORステータスの具体的な条件

環境テストでは以下の条件でPOOR判定されます：

```python
# test_environment.pyでの計算例
basic_imports: 0/5 passed     # Python基本機能
structure: 1/5 passed         # プロジェクト構造  
config: Failed                # 設定インポート
modules: 0/4 passed           # モジュール読み込み
functionality: Failed         # 基本機能
cli: 2/4 passed               # CLIファイル存在

# 総合: 3/20 = 15% → POOR
```

### 🔧 段階的回復手順

#### Stage 1: 基本確認
```bash
# 1. Python動作確認
python --version
python -c "print('Hello World')"

# 2. 作業ディレクトリ確認  
pwd
ls -la

# 3. 基本テスト
python quick_test.py
```

#### Stage 2: 最小限の修復
```bash
# 1. プロジェクト再取得
git pull origin main

# 2. 最小依存関係インストール
python -m pip install Pillow tqdm click

# 3. 設定確認
python -c "import sys; sys.path.append('src'); from config.settings import VIDEO_SETTINGS; print('OK')"
```

#### Stage 3: 完全回復
```bash
# 1. 環境リセット
python install_dependencies.py

# 2. 完全テスト
python test_environment.py

# 3. 機能確認
python main.py system
```

### 🚨 緊急時の最終手段

すべてが失敗した場合：

```bash
# 完全クリーン
rm -rf UpScaleAppProject/
rm -rf venv/

# Python再インストール確認
python --version || echo "Python installation required"

# プロジェクト再作成
git clone https://github.com/SumihisaYutani/UpScaleAppProject.git
cd UpScaleAppProject
python quick_test.py

# もしまだ失敗するなら：
# 1. システム管理者に相談
# 2. 別のマシンで試行
# 3. Docker環境での実行を検討
```

### 📞 サポート情報

POORステータスが継続する場合、以下の情報と共にサポートを求めてください：

```bash
# 診断情報の収集
python test_environment.py > diagnostic_report.txt 2>&1
python -c "import sys, platform; print(f'Python: {sys.version}'); print(f'Platform: {platform.platform()}'); print(f'Path: {sys.path}')" >> diagnostic_report.txt
echo "Working Directory: $(pwd)" >> diagnostic_report.txt
ls -la >> diagnostic_report.txt
```

この診断レポートに以下も追加：
- オペレーティングシステム詳細
- 企業環境/制限の有無
- エラーメッセージの全文
- 試行した解決策

---

**重要**: POORステータスは基本的なPython環境やプロジェクト構造に根本的な問題があることを示しています。まずは`python quick_test.py`で基本動作を確認し、段階的に問題を解決してください。