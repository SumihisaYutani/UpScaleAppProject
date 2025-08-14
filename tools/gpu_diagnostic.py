#!/usr/bin/env python3
"""
GPU Diagnostic Tool for UpScale App
Diagnoses GPU availability, Vulkan support, and performance issues
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from modules.waifu2x_processor import detect_vulkan_gpus, get_performance_info, test_waifu2x_availability
except ImportError:
    print("Error: Cannot import waifu2x processor modules")
    sys.exit(1)

def setup_logging():
    """Setup logging for diagnostics"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_vulkan_support():
    """Check Vulkan support on system"""
    logger = logging.getLogger(__name__)
    
    print("Checking Vulkan Support...")
    
    # Check vulkaninfo
    try:
        result = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("[OK] Vulkan is supported")
            
            # Parse GPU devices and avoid duplicates by using unique device names
            lines = result.stdout.split('\n')
            found_device_names = set()
            gpu_count = 0
            
            for line in lines:
                if 'deviceName' in line and '=' in line:
                    # Extract device name from lines like "deviceName = Radeon RX Vega"
                    device_name = line.split('=')[1].strip()
                    if device_name not in found_device_names:
                        print(f"   Device: {device_name}")
                        found_device_names.add(device_name)
                        gpu_count += 1
            
            if gpu_count == 0:
                print("[WARNING] No GPU devices found in vulkaninfo output")
            
            return True
        else:
            print("[ERROR] Vulkan not supported or vulkaninfo failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[ERROR] vulkaninfo not found - Vulkan SDK may not be installed")
        return False
    except subprocess.TimeoutExpired:
        print("[ERROR] vulkaninfo timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking Vulkan: {e}")
        return False

def check_waifu2x_executable():
    """Check if waifu2x executable is available"""
    print("\nChecking Waifu2x Executable...")
    
    waifu2x_path = Path("tools/waifu2x-ncnn-vulkan/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan.exe")
    
    if not waifu2x_path.exists():
        print(f"[ERROR] Waifu2x executable not found: {waifu2x_path}")
        return False
    
    print(f"[OK] Waifu2x executable found: {waifu2x_path}")
    
    # Test execution
    try:
        result = subprocess.run([str(waifu2x_path), "-h"], 
                              capture_output=True, text=True, timeout=10)
        
        if (result.stdout.strip() and ("usage:" in result.stdout.lower() or "waifu2x" in result.stdout.lower())):
            print("[OK] Waifu2x executable is functional")
            
            # Parse GPU information from help output
            lines = result.stdout.split('\n')
            print("   Available GPUs:")
            gpu_found = False
            for line in lines:
                if 'gpu' in line.lower() and (':' in line or 'device' in line.lower()):
                    print(f"   {line.strip()}")
                    gpu_found = True
            
            if not gpu_found:
                print("   [WARNING] No GPU information found in waifu2x help output")
            
            return True
        else:
            print("[ERROR] Waifu2x executable returned unexpected output")
            print(f"   stderr: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Waifu2x executable timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error testing waifu2x executable: {e}")
        return False

def check_gpu_detection():
    """Check GPU detection using our modules"""
    print("\nChecking GPU Detection...")
    
    try:
        gpus = detect_vulkan_gpus()
        if gpus:
            print(f"[OK] Detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("[ERROR] No GPUs detected by our detection method")
        
        return len(gpus) > 0
        
    except Exception as e:
        print(f"[ERROR] Error in GPU detection: {e}")
        return False

def check_waifu2x_backends():
    """Check waifu2x backend availability"""
    print("\nChecking Waifu2x Backend Availability...")
    
    try:
        availability = test_waifu2x_availability()
        
        backends = {
            "NCNN-Vulkan": availability.get("ncnn", False),
            "AMD": availability.get("amd", False), 
            "Chainer": availability.get("chainer", False),
            "Mock": availability.get("mock", False)
        }
        
        any_available = availability.get("any_available", False)
        
        for backend, available in backends.items():
            status = "[OK]" if available else "[FAIL]"
            print(f"   {status} {backend}: {'Available' if available else 'Not available'}")
        
        if any_available:
            print("[OK] At least one backend is available")
        else:
            print("[ERROR] No backends are available - this will cause CPU fallback")
        
        return any_available
        
    except Exception as e:
        print(f"[ERROR] Error checking backend availability: {e}")
        return False

def test_performance():
    """Test system performance"""
    print("\nTesting System Performance...")
    
    try:
        perf_info = get_performance_info()
        
        print(f"   CPU Usage: {perf_info['cpu_usage']:.1f}%")
        print(f"   Memory Usage: {perf_info['memory_usage']:.1f}%") 
        print(f"   GPU Usage: {perf_info['gpu_usage']}%")
        print(f"   Processing Mode: {perf_info['processing_mode']}")
        print(f"   Backend: {perf_info['backend']}")
        print(f"   GPU ID: {perf_info['gpu_id']}")
        
        if perf_info['processing_mode'] == 'CPU':
            print("[WARNING] System is running in CPU mode")
            print("   This will result in very slow processing (hours instead of minutes)")
        elif perf_info['gpu_usage'] == -1:
            print("[WARNING] Cannot monitor GPU usage")
            print("   GPU usage monitoring is not available for AMD GPUs on this system")
            print("   This is normal and doesn't affect GPU processing functionality")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error getting performance info: {e}")
        return False

def provide_recommendations():
    """Provide recommendations based on diagnostic results"""
    print("\nRECOMMENDATIONS:")
    print("1. Ensure Vulkan SDK is installed:")
    print("   - Download from: https://vulkan.lunarg.com/sdk/home")
    print("   - Install and reboot system")
    
    print("\n2. Update GPU drivers:")
    print("   - NVIDIA: GeForce Experience or NVIDIA website")
    print("   - AMD: AMD Software Adrenalin Edition") 
    print("   - Intel: Intel Driver & Support Assistant")
    
    print("\n3. Check Windows GPU settings:")
    print("   - Windows Settings > System > Display > Graphics settings")
    print("   - Set UpScaleApp to 'High performance' GPU")
    
    print("\n4. For AMD GPUs:")
    print("   - Ensure Vulkan support is enabled in AMD drivers")
    print("   - Try running as administrator")
    
    print("\n5. Diagnostic logs:")
    print("   - Check logs/gui_debug.log for detailed error messages")
    print("   - Look for 'NCNN backend initialized' messages")

def main():
    """Main diagnostic function"""
    print("UpScale App GPU Diagnostic Tool")
    print("=" * 50)
    
    logger = setup_logging()
    
    results = {
        'vulkan': check_vulkan_support(),
        'waifu2x_exe': check_waifu2x_executable(), 
        'gpu_detection': check_gpu_detection(),
        'backends': check_waifu2x_backends(),
        'performance': test_performance()
    }
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    all_good = True
    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test.replace('_', ' ').title()}")
        if not result:
            all_good = False
    
    if all_good:
        print("\n[SUCCESS] All checks passed! GPU processing should work correctly.")
    else:
        print("\n[WARNING] Some issues detected. See recommendations below.")
        provide_recommendations()
    
    print("\nFor support, share this diagnostic output along with:")
    print("- Your GPU model")  
    print("- Windows version")
    print("- Contents of logs/gui_debug.log")

if __name__ == "__main__":
    main()