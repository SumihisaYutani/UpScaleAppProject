#!/usr/bin/env python3
"""
Waifu2x Testing Script
Test waifu2x functionality and performance
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_waifu2x_import():
    """Test waifu2x import and availability"""
    print("=== Waifu2x Import Test ===")
    
    try:
        from modules.waifu2x_processor import test_waifu2x_availability, Waifu2xUpscaler
        print("✓ Waifu2x processor module imported successfully")
        
        # Test availability
        availability = test_waifu2x_availability()
        print(f"NCNN backend available: {availability['ncnn']}")
        print(f"Chainer backend available: {availability['chainer']}")
        print(f"Any backend available: {availability['any_available']}")
        
        return availability['any_available']
        
    except ImportError as e:
        print(f"✗ Failed to import waifu2x processor: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing availability: {e}")
        return False

def test_waifu2x_initialization():
    """Test waifu2x initialization"""
    print("\n=== Waifu2x Initialization Test ===")
    
    try:
        from modules.waifu2x_processor import Waifu2xUpscaler
        
        # Test with default settings
        upscaler = Waifu2xUpscaler()
        
        if upscaler.is_available():
            print("✓ Waifu2x upscaler initialized successfully")
            
            # Get and display info
            info = upscaler.get_info()
            print(f"Backend: {info['backend']}")
            print(f"GPU ID: {info['gpu_id']}")
            print(f"Scale: {info['scale']}x")
            print(f"Noise Level: {info['noise']}")
            print(f"Model: {info['model']}")
            print(f"Supported scales: {info['supported_scales']}")
            print(f"Supported noise levels: {info['supported_noise_levels']}")
            
            return True
        else:
            print("✗ Waifu2x upscaler initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Error initializing waifu2x: {e}")
        return False

def test_waifu2x_image_processing():
    """Test waifu2x image processing with a small test image"""
    print("\n=== Waifu2x Image Processing Test ===")
    
    try:
        from modules.waifu2x_processor import Waifu2xUpscaler
        from PIL import Image
        import numpy as np
        
        # Create a small test image
        test_image = Image.new('RGB', (64, 64), color='red')
        
        # Add some pattern
        pixels = np.array(test_image)
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    pixels[i:i+8, j:j+8] = [0, 255, 0]  # Green squares
        test_image = Image.fromarray(pixels)
        
        print(f"Created test image: {test_image.size} pixels")
        
        # Initialize upscaler
        upscaler = Waifu2xUpscaler(scale=2, noise=1)
        
        if not upscaler.is_available():
            print("✗ Waifu2x not available for processing test")
            return False
        
        # Process image
        print("Processing test image...")
        start_time = time.time()
        
        upscaled = upscaler.upscale_image(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if upscaled is not None:
            print(f"✓ Image processed successfully")
            print(f"Original size: {test_image.size}")
            print(f"Upscaled size: {upscaled.size}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Save test result
            test_output = Path("test_waifu2x_output.png")
            upscaled.save(test_output)
            print(f"Test result saved to: {test_output}")
            
            return True
        else:
            print("✗ Image processing failed - returned None")
            return False
            
    except Exception as e:
        print(f"✗ Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_waifu2x_settings():
    """Test different waifu2x settings"""
    print("\n=== Waifu2x Settings Test ===")
    
    try:
        from modules.waifu2x_processor import Waifu2xUpscaler
        
        # Test different scale factors
        scales_to_test = [1, 2, 4]
        
        for scale in scales_to_test:
            print(f"\nTesting scale {scale}x...")
            try:
                upscaler = Waifu2xUpscaler(scale=scale, noise=0)
                if upscaler.is_available():
                    print(f"✓ Scale {scale}x initialization successful")
                else:
                    print(f"✗ Scale {scale}x initialization failed")
            except Exception as e:
                print(f"✗ Scale {scale}x failed: {e}")
        
        # Test different noise levels
        noise_levels = [-1, 0, 1, 2, 3]
        
        for noise in noise_levels:
            print(f"\nTesting noise level {noise}...")
            try:
                upscaler = Waifu2xUpscaler(scale=2, noise=noise)
                if upscaler.is_available():
                    print(f"✓ Noise level {noise} initialization successful")
                else:
                    print(f"✗ Noise level {noise} initialization failed")
            except Exception as e:
                print(f"✗ Noise level {noise} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing settings: {e}")
        return False

def main():
    """Main test function"""
    print("Waifu2x Testing Script")
    print("=" * 50)
    
    # Test 1: Import and availability
    if not test_waifu2x_import():
        print("\n❌ Basic import test failed. Waifu2x is not available.")
        print("\nTo install Waifu2x support:")
        print("pip install waifu2x-ncnn-vulkan-python")
        print("\nNote: Requires Vulkan-compatible GPU for best performance")
        return 1
    
    # Test 2: Initialization
    if not test_waifu2x_initialization():
        print("\n❌ Initialization test failed.")
        return 1
    
    # Test 3: Settings
    if not test_waifu2x_settings():
        print("\n❌ Settings test failed.")
        return 1
    
    # Test 4: Image processing
    if not test_waifu2x_image_processing():
        print("\n❌ Image processing test failed.")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 All Waifu2x tests passed successfully!")
    print("Waifu2x is ready for high-quality image upscaling.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())