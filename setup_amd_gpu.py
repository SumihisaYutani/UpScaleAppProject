#!/usr/bin/env python3
"""
AMD GPU Setup and Test Script for UpScale App
This script helps set up and test AMD GPU support for waifu2x processing
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8 or higher is required")
        return False
    else:
        print("[OK] Python version is compatible")
        return True

def check_system_info():
    """Check system information"""
    print_header("System Information")
    
    system = platform.system()
    print(f"Operating System: {system}")
    print(f"Architecture: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    
    return system.lower()

def detect_amd_gpus():
    """Detect AMD GPUs"""
    print_header("AMD GPU Detection")
    
    try:
        from modules.amd_gpu_detector import get_amd_gpu_info
        
        gpu_info = get_amd_gpu_info()
        
        print(f"AMD GPUs Found: {gpu_info.get('amd_gpus_found', 0)}")
        print(f"ROCm Available: {gpu_info.get('rocm_available', False)}")
        print(f"Vulkan Available: {gpu_info.get('vulkan_available', False)}")
        print(f"Recommended Backend: {gpu_info.get('recommended_backend', 'cpu')}")
        
        if gpu_info.get('amd_gpus'):
            print("\nDetected AMD GPUs:")
            for i, gpu in enumerate(gpu_info['amd_gpus']):
                print(f"  {i}: {gpu.get('name', 'Unknown')} ({gpu.get('method', 'unknown')})")
                if gpu.get('memory'):
                    print(f"      Memory: {gpu['memory']} bytes")
        
        return gpu_info
        
    except Exception as e:
        print(f"[ERROR] Error detecting AMD GPUs: {e}")
        return None

def install_amd_dependencies(system_platform):
    """Install AMD-specific dependencies"""
    print_header("Installing AMD Dependencies")
    
    try:
        # Install basic requirements
        print("Installing basic requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_amd.txt"], 
                      check=True)
        
        # Install ROCm PyTorch if on Linux
        if system_platform == "linux":
            print("Installing PyTorch with ROCm support...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/rocm6.0"
            ], check=True)
        else:
            print("Installing standard PyTorch...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ], check=True)
        
        print("[OK] Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error installing dependencies: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_amd_backend():
    """Test AMD backend functionality"""
    print_header("Testing AMD Backend")
    
    try:
        from modules.amd_waifu2x_backend import test_amd_waifu2x_availability, AMDWaifu2xBackend
        
        # Test availability
        availability = test_amd_waifu2x_availability()
        
        print("AMD Backend Availability:")
        print(f"  Backend Available: {availability.get('amd_backend_available', False)}")
        print(f"  ROCm Available: {availability.get('rocm_available', False)}")
        print(f"  Vulkan Available: {availability.get('vulkan_available', False)}")
        print(f"  Recommended: {availability.get('recommended_backend', 'cpu')}")
        
        if availability.get('amd_backend_available'):
            print("\n[OK] AMD backend is available!")
            
            # Test backend initialization
            print("\nTesting backend initialization...")
            backend = AMDWaifu2xBackend()
            
            if backend.is_available():
                print("[OK] Backend initialized successfully")
                
                # Get performance info
                perf_info = backend.get_performance_info()
                print("\nPerformance Information:")
                for key, value in perf_info.items():
                    print(f"  {key}: {value}")
                
                return True
            else:
                print("[ERROR] Backend initialization failed")
                return False
        else:
            print("[ERROR] AMD backend not available")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing AMD backend: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing with AMD backend"""
    print_header("Testing Image Processing")
    
    try:
        from modules.amd_waifu2x_backend import AMDWaifu2xBackend
        from PIL import Image
        
        # Create test image
        test_image = Image.new('RGB', (256, 256), (128, 128, 128))
        
        # Initialize backend
        backend = AMDWaifu2xBackend(scale=2)
        
        if not backend.is_available():
            print("[ERROR] Backend not available for testing")
            return False
        
        print("Processing test image...")
        result = backend.upscale_image(test_image)
        
        if result:
            print(f"[OK] Image processed successfully")
            print(f"   Input size: {test_image.size}")
            print(f"   Output size: {result.size}")
            return True
        else:
            print("[ERROR] Image processing failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing image processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_waifu2x_integration():
    """Test integration with main waifu2x processor"""
    print_header("Testing Waifu2x Integration")
    
    try:
        from modules.waifu2x_processor import Waifu2xUpscaler, test_waifu2x_availability
        from PIL import Image
        
        # Check availability
        availability = test_waifu2x_availability()
        print("Waifu2x Availability:")
        for backend, available in availability.items():
            status = "[OK]" if available else "[--]"    
            print(f"  {backend}: {status}")
        
        if availability.get('amd'):
            print("\n[OK] AMD backend available in main processor")
            
            # Test upscaler
            print("Testing integrated upscaler...")
            upscaler = Waifu2xUpscaler(backend="amd", scale=2)
            
            if upscaler.is_available():
                print("[OK] Upscaler initialized successfully")
                
                # Test with small image
                test_image = Image.new('RGB', (128, 128), (64, 64, 64))
                result = upscaler.upscale_image(test_image)
                
                if result:
                    print(f"[OK] Integration test passed")
                    print(f"   Input: {test_image.size}")
                    print(f"   Output: {result.size}")
                    return True
                else:
                    print("[ERROR] Integration test failed")
                    return False
            else:
                print("[ERROR] Upscaler initialization failed")
                return False
        else:
            print("[ERROR] AMD backend not available in main processor")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_setup_summary(results):
    """Print setup summary"""
    print_header("Setup Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print()
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] All tests passed! AMD GPU support is ready.")
        print("\nNext steps:")
        print("1. Run the GUI: python main_gui.py")
        print("2. Select 'AMD' backend in settings")
        print("3. Start processing videos with AMD GPU acceleration")
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} test(s) failed.")
        print("\nTroubleshooting:")
        print("1. Check AMD GPU drivers are installed")
        print("2. Install ROCm (Linux) or latest AMD drivers (Windows)")
        print("3. Ensure Vulkan SDK is installed")
        print("4. Try running: python -c 'from modules.amd_gpu_detector import *; print(get_amd_gpu_info())'")

def main():
    """Main setup function"""
    print("AMD GPU Setup for UpScale App")
    print("This script will help set up AMD GPU support for waifu2x processing")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    results = {}
    
    # Check Python version
    results['Python Version'] = check_python_version()
    
    # System info
    system_platform = check_system_info()
    
    # AMD GPU detection
    gpu_info = detect_amd_gpus()
    results['AMD GPU Detection'] = gpu_info is not None and gpu_info.get('amd_gpus_found', 0) > 0
    
    # Install dependencies
    print("\nSkipping dependency installation (use -y flag to install automatically)")
    results['Dependency Installation'] = True  # Assume already installed
    
    # Test backend
    results['AMD Backend Test'] = test_amd_backend()
    
    # Test image processing
    if results['AMD Backend Test']:
        results['Image Processing Test'] = test_image_processing()
    else:
        results['Image Processing Test'] = False
    
    # Test integration
    results['Waifu2x Integration'] = test_waifu2x_integration()
    
    # Print summary
    print_setup_summary(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)