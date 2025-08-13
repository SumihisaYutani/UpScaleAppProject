#!/usr/bin/env python3
"""
Dependency Installation Script
Installs dependencies in stages with fallback options
"""

import subprocess
import sys
import importlib
from pathlib import Path

def run_pip_install(packages, description="packages", timeout=300):
    """Install packages with error handling"""
    print(f"📦 Installing {description}...")
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode == 0:
                print(f"  ✅ {package} installed successfully")
            else:
                print(f"  ❌ {package} failed: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  {package} installation timeout (skipped)")
        except Exception as e:
            print(f"  ❌ {package} installation error: {e}")

def check_package(package_name, import_name=None):
    """Check if package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_core_dependencies():
    """Install core dependencies"""
    core_packages = [
        'pip',  # Upgrade pip first
        'wheel',  # For building packages
        'Pillow',
        'numpy',
        'tqdm',
        'click',
        'python-dotenv',
        'psutil',
        'pytest'
    ]
    
    run_pip_install(core_packages, "core dependencies")

def install_video_dependencies():
    """Install video processing dependencies"""
    video_packages = [
        'opencv-python',
        'ffmpeg-python'
    ]
    
    print("\n📹 Installing video processing dependencies...")
    print("Note: These packages are larger and may take longer to install")
    
    run_pip_install(video_packages, "video processing dependencies", timeout=600)

def install_ai_dependencies():
    """Install AI dependencies (optional)"""
    print("\n🤖 Installing AI dependencies (optional)...")
    print("Note: These are large packages and require significant disk space")
    
    # Try lightweight torch first
    try:
        print("  Attempting CPU-only PyTorch installation...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu'
        ], capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            print("  ✅ PyTorch (CPU) installed successfully")
        else:
            print("  ⚠️  PyTorch installation failed, trying standard version...")
            run_pip_install(['torch', 'torchvision'], "PyTorch (standard)", timeout=900)
            
    except Exception as e:
        print(f"  ❌ PyTorch installation error: {e}")
    
    # Install other AI packages
    ai_packages = [
        'diffusers',
        'transformers',
        'accelerate'
    ]
    
    run_pip_install(ai_packages, "AI packages", timeout=600)

def install_optional_dependencies():
    """Install optional monitoring dependencies"""
    optional_packages = [
        'pynvml'  # For GPU monitoring
    ]
    
    run_pip_install(optional_packages, "optional monitoring dependencies")

def test_imports():
    """Test that key packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('click', 'Click'),
        ('psutil', 'psutil'),
        ('cv2', 'OpenCV'),
        ('ffmpeg', 'ffmpeg-python'),
        ('torch', 'PyTorch'),
        ('diffusers', 'Diffusers')
    ]
    
    results = []
    for import_name, package_name in test_packages:
        available = check_package('', import_name)
        status = "✅" if available else "❌"
        print(f"  {status} {package_name}")
        results.append((package_name, available))
    
    return results

def create_fallback_modules():
    """Create fallback modules for missing dependencies"""
    print("\n🔧 Creating fallback modules for missing dependencies...")
    
    # Create fallback for torch if not available
    if not check_package('', 'torch'):
        fallback_torch = '''
# Fallback torch module
import warnings

def cuda():
    class CudaMock:
        def is_available(self): return False
        def device_count(self): return 0
    return CudaMock()

def Generator(device=None):
    import random
    return random.Random()

def autocast(device_type):
    class AutocastMock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    return AutocastMock()

warnings.warn("PyTorch not available. Using fallback implementation.")
'''
        
        Path("src/fallbacks").mkdir(parents=True, exist_ok=True)
        with open("src/fallbacks/torch_fallback.py", "w") as f:
            f.write(fallback_torch)
        print("  ✅ Created torch fallback")
    
    # Create fallback for cv2 if not available
    if not check_package('', 'cv2'):
        fallback_cv2 = '''
# Fallback cv2 module  
import warnings
from PIL import Image

INTER_LANCZOS = "LANCZOS"

def imread(path):
    """Fallback imread using PIL"""
    return Image.open(path)

def imwrite(path, image):
    """Fallback imwrite using PIL"""
    if hasattr(image, 'save'):
        image.save(path)
    return True

warnings.warn("OpenCV not available. Using PIL fallback.")
'''
        
        with open("src/fallbacks/cv2_fallback.py", "w") as f:
            f.write(fallback_cv2)
        print("  ✅ Created OpenCV fallback")

def main():
    """Main installation function"""
    print("🎬 UpScale App - Dependency Installation")
    print("=" * 50)
    
    # Upgrade pip first
    print("🔧 Upgrading pip...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        print("  ✅ pip upgraded")
    except:
        print("  ⚠️  pip upgrade failed, continuing...")
    
    # Install in stages
    install_core_dependencies()
    
    # Ask about video dependencies
    print("\n" + "?" * 30)
    install_video = input("Install video processing dependencies? (Y/n): ").strip().lower()
    if install_video != 'n':
        install_video_dependencies()
    
    # Ask about AI dependencies
    print("\n" + "?" * 30)
    install_ai = input("Install AI dependencies? (large download, Y/n): ").strip().lower()
    if install_ai != 'n':
        install_ai_dependencies()
    
    # Install optional
    install_optional_dependencies()
    
    # Test imports
    test_results = test_imports()
    
    # Create fallbacks for missing packages
    create_fallback_modules()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    
    available_count = sum(1 for _, available in test_results if available)
    total_count = len(test_results)
    
    print(f"  Available packages: {available_count}/{total_count}")
    
    if available_count >= 5:  # Core packages available
        print("  ✅ Core functionality should work")
    else:
        print("  ⚠️  Some core packages missing")
    
    print("\n🚀 Next steps:")
    print("  1. Run: python setup_environment.py")
    print("  2. Test: python -c 'import sys; sys.path.append(\"src\"); from config.settings import VIDEO_SETTINGS; print(\"✅ Basic setup working\")'")
    print("  3. Try: python main.py system")

if __name__ == "__main__":
    main()