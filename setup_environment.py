#!/usr/bin/env python3
"""
Environment Setup Script for UpScale App
Automatically detects and configures Python environment with dependencies
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
import json

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🎬 UpScale App - Environment Setup")
    print("=" * 60)

def detect_python():
    """Detect available Python installations"""
    python_commands = ['python', 'python3', 'py']
    python_paths = []
    
    print("🔍 Detecting Python installations...")
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                path = shutil.which(cmd)
                python_paths.append({
                    'command': cmd,
                    'version': version,
                    'path': path
                })
                print(f"  ✅ {cmd}: {version} ({path})")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            print(f"  ❌ {cmd}: Not found")
    
    return python_paths

def check_python_version(python_cmd):
    """Check if Python version is compatible"""
    try:
        result = subprocess.run([python_cmd, '-c', 
                               'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = float(result.stdout.strip())
            return version >= 3.8
    except:
        pass
    return False

def create_virtual_environment(python_cmd):
    """Create virtual environment"""
    print("\n📦 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("  ⚠️  Virtual environment already exists")
        return str(venv_path)
    
    try:
        # Try creating venv
        subprocess.run([python_cmd, '-m', 'venv', 'venv'], check=True)
        print("  ✅ Virtual environment created successfully")
        return str(venv_path)
    except subprocess.CalledProcessError:
        print("  ❌ Failed to create virtual environment")
        return None

def get_venv_python():
    """Get virtual environment Python path"""
    system = platform.system()
    if system == "Windows":
        return Path("venv/Scripts/python.exe")
    else:
        return Path("venv/bin/python")

def install_dependencies(python_cmd):
    """Install project dependencies"""
    print("\n📥 Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("  ❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("  🔧 Upgrading pip...")
        subprocess.run([python_cmd, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("  📦 Installing requirements...")
        result = subprocess.run([python_cmd, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Dependencies installed successfully")
            return True
        else:
            print(f"  ❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Installation error: {e}")
        return False

def install_optional_dependencies(python_cmd):
    """Install optional dependencies for enhanced features"""
    print("\n🔧 Installing optional dependencies...")
    
    optional_packages = [
        ('torch', 'PyTorch for GPU acceleration'),
        ('diffusers', 'Hugging Face Diffusers for AI models'),
        ('pynvml', 'NVIDIA GPU monitoring'),
        ('psutil', 'System monitoring'),
    ]
    
    for package, description in optional_packages:
        try:
            print(f"  📦 Installing {package} ({description})...")
            result = subprocess.run([python_cmd, '-m', 'pip', 'install', package],
                                   capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"    ✅ {package} installed")
            else:
                print(f"    ⚠️  {package} installation failed (optional)")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  {package} installation timeout (skipped)")
        except Exception as e:
            print(f"    ⚠️  {package} installation error: {e}")

def test_basic_functionality(python_cmd):
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    test_script = '''
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from config.settings import VIDEO_SETTINGS, AI_SETTINGS
    print("✅ Settings import successful")
    
    from modules.video_processor import VideoProcessor
    processor = VideoProcessor()
    print("✅ Video processor initialization successful")
    
    from modules.ai_processor import SimpleUpscaler
    upscaler = SimpleUpscaler()
    print("✅ Simple upscaler initialization successful")
    
    print("🎉 Basic functionality test passed")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run([python_cmd, '-c', test_script], 
                               capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("  ❌ Basic functionality test failed")
            print(f"  Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Test execution failed: {e}")
        return False

def create_activation_scripts():
    """Create environment activation scripts"""
    print("\n📜 Creating activation scripts...")
    
    # Windows batch script
    windows_script = '''@echo off
echo 🎬 UpScale App Environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
echo ✅ Environment activated!
echo.
echo Available commands:
echo   python main.py upscale video.mp4
echo   python main_enhanced.py upscale video.mp4 --show-system-stats
echo   python main.py info video.mp4
echo   python main.py system
echo.
'''
    
    # Unix shell script  
    unix_script = '''#!/bin/bash
echo "🎬 UpScale App Environment"
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Environment activated!"
echo ""
echo "Available commands:"
echo "  python main.py upscale video.mp4"
echo "  python main_enhanced.py upscale video.mp4 --show-system-stats" 
echo "  python main.py info video.mp4"
echo "  python main.py system"
echo ""
'''
    
    try:
        # Windows
        with open("activate_env.bat", "w", encoding="utf-8") as f:
            f.write(windows_script)
        
        # Unix
        with open("activate_env.sh", "w", encoding="utf-8") as f:
            f.write(unix_script)
        
        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod("activate_env.sh", 0o755)
        
        print("  ✅ Activation scripts created:")
        print("    - activate_env.bat (Windows)")
        print("    - activate_env.sh (Linux/macOS)")
        
    except Exception as e:
        print(f"  ⚠️  Script creation failed: {e}")

def create_environment_info():
    """Create environment information file"""
    print("\n📋 Creating environment information...")
    
    python_paths = detect_python()
    
    env_info = {
        "setup_date": str(Path().cwd()),
        "platform": platform.system(),
        "python_version": sys.version,
        "available_pythons": python_paths,
        "project_structure": {
            "main_cli": "main.py",
            "enhanced_cli": "main_enhanced.py", 
            "config": "config/settings.py",
            "modules": "src/modules/",
            "tests": "tests/",
            "docs": ["README.md", "PROJECT_DESIGN.md", "CHANGELOG.md"]
        },
        "quick_start": [
            "Activate environment: ./activate_env.bat (Windows) or ./activate_env.sh (Unix)",
            "Basic usage: python main.py upscale video.mp4",
            "Enhanced usage: python main_enhanced.py upscale video.mp4 --show-system-stats",
            "Get info: python main.py info video.mp4",
            "System check: python main.py system"
        ]
    }
    
    try:
        with open("environment_info.json", "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2, ensure_ascii=False, default=str)
        
        print("  ✅ Environment info saved to environment_info.json")
        
    except Exception as e:
        print(f"  ⚠️  Environment info creation failed: {e}")

def print_summary(success, python_cmd):
    """Print setup summary"""
    print("\n" + "=" * 60)
    if success:
        print("🎉 Environment setup completed successfully!")
        print("")
        print("📁 Project structure ready")
        print("🐍 Python environment configured")
        print("📦 Dependencies installed")
        print("🧪 Basic functionality tested")
        print("")
        print("🚀 Quick Start:")
        system = platform.system()
        if system == "Windows":
            print("  1. Run: activate_env.bat")
        else:
            print("  1. Run: ./activate_env.sh")
        print("  2. Test: python main.py system")
        print("  3. Process video: python main.py upscale your_video.mp4")
        print("")
        print("📚 Documentation:")
        print("  - README.md: Full documentation")
        print("  - CHANGELOG.md: Version history")  
        print("  - environment_info.json: Environment details")
        
    else:
        print("❌ Environment setup encountered issues")
        print("")
        print("🔧 Troubleshooting:")
        print("  - Ensure Python 3.8+ is installed")
        print("  - Check internet connection for package downloads")
        print("  - Try manual installation: pip install -r requirements.txt")
        print("  - Check environment_info.json for details")
    
    print("=" * 60)

def main():
    """Main setup function"""
    print_header()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Detect Python
    python_paths = detect_python()
    
    if not python_paths:
        print("❌ No Python installation found!")
        print("Please install Python 3.8+ from https://python.org")
        return False
    
    # Find suitable Python
    suitable_python = None
    for python_info in python_paths:
        if check_python_version(python_info['command']):
            suitable_python = python_info['command']
            print(f"✅ Using {python_info['command']}: {python_info['version']}")
            break
    
    if not suitable_python:
        print("❌ No suitable Python version found (requires 3.8+)")
        return False
    
    # Create virtual environment
    venv_path = create_virtual_environment(suitable_python)
    if not venv_path:
        print("⚠️  Continuing without virtual environment...")
        python_cmd = suitable_python
    else:
        python_cmd = str(get_venv_python())
    
    # Install dependencies
    deps_success = install_dependencies(python_cmd)
    if not deps_success:
        print("⚠️  Some dependencies failed to install")
    
    # Install optional dependencies
    install_optional_dependencies(python_cmd)
    
    # Test functionality
    test_success = test_basic_functionality(python_cmd)
    
    # Create helper scripts
    create_activation_scripts()
    create_environment_info()
    
    # Print summary
    success = deps_success and test_success
    print_summary(success, python_cmd)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        sys.exit(1)