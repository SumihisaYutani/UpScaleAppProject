# 🛠️ Environment Setup Guide

This guide helps you set up and troubleshoot the UpScale App Python environment.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Windows
run_test.bat

# Linux/macOS
./run_test.sh
```

### Option 2: Manual Setup
```bash
# 1. Test environment
python quick_test.py

# 2. Full environment test
python test_environment.py

# 3. Install dependencies
python install_dependencies.py

# 4. Complete setup
python setup_environment.py
```

## 🔧 Python Environment Issues

### Problem: "Python command not found"

This happens when Python is not in your system PATH or not installed.

#### Windows Solutions:

1. **Try different commands:**
   ```cmd
   python --version
   python3 --version
   py --version
   ```

2. **Install Python with PATH:**
   - Download from [python.org](https://python.org)
   - ⚠️ **Check "Add Python to PATH" during installation**
   - Restart command prompt after installation

3. **Find existing Python:**
   ```cmd
   where python
   dir C:\Users\%USERNAME%\AppData\Local\Programs\Python\*
   ```

4. **Manual PATH setup:**
   - Open System Properties → Environment Variables
   - Add to PATH: `C:\Users\YourName\AppData\Local\Programs\Python\Python39`
   - Add to PATH: `C:\Users\YourName\AppData\Local\Programs\Python\Python39\Scripts`

#### Linux/macOS Solutions:

1. **Install Python:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip
   
   # macOS (Homebrew)
   brew install python3
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip
   ```

2. **Check installation:**
   ```bash
   which python3
   python3 --version
   ```

## 📦 Dependency Management

### Staged Installation Approach

The project uses a staged dependency installation to handle different environments:

1. **Core dependencies** (always required)
2. **Video processing** (for video handling)
3. **AI dependencies** (for AI upscaling)
4. **Optional dependencies** (for enhanced features)

### Installation Scripts

- `install_dependencies.py` - Interactive dependency installer
- `requirements_minimal.txt` - Core dependencies only
- `requirements.txt` - Full dependencies

### Fallback System

The application includes fallback modules when heavy dependencies are missing:

- **torch_fallback.py** - Basic torch functions without PyTorch
- **cv2_fallback.py** - Basic OpenCV functions using PIL

## 🧪 Testing Your Environment

### Quick Test (No dependencies)
```bash
python quick_test.py
```
Tests basic Python and project structure.

### Full Environment Test
```bash
python test_environment.py  
```
Comprehensive test of all components and dependencies.

### Expected Results

#### Minimal Setup (Core only):
- ✅ Basic Python functionality
- ✅ Project structure
- ✅ Configuration loading
- ❌ Video processing
- ❌ AI features

#### Complete Setup:
- ✅ All basic features
- ✅ Video processing
- ✅ AI upscaling
- ✅ Enhanced monitoring

## 🔍 Troubleshooting Common Issues

### Issue 1: "Module not found" errors
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
python install_dependencies.py
# Select 'Y' for AI dependencies
```

### Issue 2: "Permission denied" errors
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
- **Windows:** Run as Administrator
- **Linux/macOS:** Use `sudo` or virtual environment
- **All platforms:** Use virtual environment (recommended)

### Issue 3: "FFmpeg not found"
```
ffmpeg: command not found
```

**Solutions:**
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Ubuntu:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### Issue 4: GPU/CUDA issues
```
CUDA out of memory
```

**Solutions:**
- Use CPU processing: `--no-ai` flag
- Reduce batch size in settings
- Use enhanced CLI with monitoring
- Free GPU memory: restart application

## 🐍 Virtual Environment Setup

### Why Use Virtual Environments?
- Isolate project dependencies
- Avoid conflicts with system Python
- Easy cleanup and reinstallation

### Setup Virtual Environment

#### Windows:
```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Activation Scripts
The setup creates activation scripts:
- `activate_env.bat` (Windows)
- `activate_env.sh` (Linux/macOS)

## 📊 Environment Status

### Check Current Status
```bash
# Quick status
python main.py system

# Detailed status  
python main_enhanced.py system --save-report

# Environment test report
python test_environment.py
```

### Status Indicators

- **🎉 EXCELLENT (80-100%)**: Full functionality
- **✅ GOOD (60-79%)**: Basic functionality works
- **⚠️ LIMITED (40-59%)**: Some features missing
- **❌ POOR (<40%)**: Significant issues

## 🔄 Environment Reset

### Clean Reset
```bash
# Remove virtual environment
rmdir /s venv          # Windows
rm -rf venv            # Linux/macOS

# Clean Python cache
python -m pip cache purge

# Fresh setup
python setup_environment.py
```

### Partial Reset
```bash
# Reinstall dependencies only
python -m pip uninstall -r requirements.txt -y
python install_dependencies.py
```

## 📱 Platform-Specific Notes

### Windows
- Use `py` launcher if available
- May need to run as Administrator for some operations
- Windows Defender might flag AI model downloads
- Use PowerShell for better Unicode support

### macOS
- Use Homebrew for system dependencies
- May need to install Xcode Command Line Tools
- M1/M2 Macs: Use ARM64 Python builds when possible

### Linux
- Install system packages with package manager
- May need `python3-dev` for building packages
- Docker containers: install system dependencies

## 🆘 Getting Help

### Diagnostic Information
Always include when reporting issues:

```bash
# Generate diagnostic report
python test_environment.py > diagnostic_report.txt

# System information
python main.py system

# Platform info
python -c "import sys, platform; print(f'Python: {sys.version}'); print(f'Platform: {platform.platform()}')"
```

### Common Solutions Summary

| Problem | Quick Fix |
|---------|-----------|
| Python not found | Install Python, add to PATH |
| Module not found | Run `python install_dependencies.py` |
| Permission error | Use virtual environment |
| FFmpeg missing | Install FFmpeg system-wide |
| CUDA issues | Use `--no-ai` flag |
| Memory errors | Use enhanced CLI with monitoring |

### Support Resources
- 📖 **README.md**: Main documentation
- 📋 **CHANGELOG.md**: Version changes
- 🔧 **PROJECT_DESIGN.md**: Technical details
- 🐛 **GitHub Issues**: Report bugs
- 💬 **Discussions**: Ask questions

---

**Need immediate help?** Run `python quick_test.py` for a fast diagnosis!