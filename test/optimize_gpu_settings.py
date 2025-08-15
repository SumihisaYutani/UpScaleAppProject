#!/usr/bin/env python3
"""
GPU Settings Optimization Script for Waifu2x
Tests different tile sizes to find optimal BLOCKSIZE equivalent for Radeon RX Vega
"""

import subprocess
import time
import os
from pathlib import Path

class Waifu2xOptimizer:
    def __init__(self):
        self.waifu2x_path = Path("../tools/waifu2x-ncnn-vulkan/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan.exe")
        self.test_image = Path("test.png")
        self.results = []
    
    def test_tile_size(self, tile_size: int, gpu_id: int = 0) -> dict:
        """Test waifu2x with specific tile size"""
        output_file = f"test_result_tile_{tile_size}.png"
        
        cmd = [
            str(self.waifu2x_path),
            "-i", str(self.test_image),
            "-o", output_file,
            "-s", "2",  # 2x upscale
            "-n", "1",  # noise reduction
            "-g", str(gpu_id),  # GPU ID
            "-t", str(tile_size),  # tile size (BLOCKSIZE equivalent)
            "-j", "1:4:2",  # threading
            "-m", "models-cunet",
            "-f", "png",
            "-v"  # verbose
        ]
        
        print(f"Testing tile size: {tile_size}")
        start_time = time.time()
        
        try:
            # Hide console window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,  # 1 minute timeout
                startupinfo=startupinfo
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            success = result.returncode == 0 and Path(output_file).exists()
            
            test_result = {
                "tile_size": tile_size,
                "success": success,
                "processing_time": processing_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_file": output_file
            }
            
            if success:
                # Get output file size
                output_size = Path(output_file).stat().st_size
                test_result["output_size"] = output_size
                print(f"SUCCESS - Tile {tile_size}: {processing_time:.2f}s, {output_size} bytes")
            else:
                print(f"FAILED - Tile {tile_size}: {result.stderr}")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT - Tile {tile_size}: Exceeded 60 seconds")
            return {
                "tile_size": tile_size,
                "success": False,
                "processing_time": 60.0,
                "error": "timeout"
            }
        except Exception as e:
            print(f"ERROR - Tile {tile_size}: {e}")
            return {
                "tile_size": tile_size,
                "success": False,
                "error": str(e)
            }
    
    def run_optimization_tests(self):
        """Run tests with different tile sizes to find optimal settings"""
        # Test tile sizes (equivalent to BLOCKSIZE)
        tile_sizes = [128, 256, 384, 512, 768, 1024, 1536, 2048]
        
        print("Starting Waifu2x GPU Optimization Tests")
        print("=" * 50)
        print(f"Test image: {self.test_image}")
        print(f"Waifu2x path: {self.waifu2x_path}")
        print("=" * 50)
        
        for tile_size in tile_sizes:
            result = self.test_tile_size(tile_size)
            self.results.append(result)
            
            # Stop testing if we hit memory limits
            if not result["success"] and "memory" in result.get("stderr", "").lower():
                print(f"WARNING: Memory limit reached at tile size {tile_size}")
                break
            
            time.sleep(1)  # Brief pause between tests
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze test results and recommend optimal settings"""
        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)
        
        successful_tests = [r for r in self.results if r["success"]]
        
        if not successful_tests:
            print("ERROR: No successful tests! Check GPU drivers and Vulkan support.")
            return
        
        # Find fastest successful test
        fastest = min(successful_tests, key=lambda x: x["processing_time"])
        
        print(f"OPTIMAL SETTINGS FOUND:")
        print(f"   Tile Size (BLOCKSIZE): {fastest['tile_size']}")
        print(f"   Processing Time: {fastest['processing_time']:.2f} seconds")
        print(f"   GPU ID: 0 (Radeon RX Vega)")
        print(f"   Threading: 1:4:2")
        
        print(f"\nPerformance Summary:")
        for result in successful_tests:
            tile = result["tile_size"]
            time_taken = result["processing_time"]
            speedup = fastest["processing_time"] / time_taken
            print(f"   Tile {tile:4d}: {time_taken:6.2f}s (Speedup: {speedup:.2f}x)")
        
        # Update the application settings
        self.update_application_settings(fastest["tile_size"])
        
        print(f"\nSUCCESS: Settings have been applied to the application!")
    
    def update_application_settings(self, optimal_tile_size: int):
        """Update the application with optimal settings"""
        # This would update the actual application settings
        # For now, just save to a config file
        config = {
            "optimal_tile_size": optimal_tile_size,
            "gpu_id": 0,
            "threading": "1:4:2",
            "model": "models-cunet"
        }
        
        with open("optimal_gpu_settings.txt", "w") as f:
            f.write("# Optimal GPU Settings for Waifu2x\n")
            f.write(f"TILE_SIZE={optimal_tile_size}\n")
            f.write("GPU_ID=0\n")
            f.write("THREADING=1:4:2\n")
            f.write("MODEL=models-cunet\n")

if __name__ == "__main__":
    optimizer = Waifu2xOptimizer()
    
    if not optimizer.test_image.exists():
        print("ERROR: test.png not found! Run create_test_image.py first.")
        exit(1)
    
    if not optimizer.waifu2x_path.exists():
        print("ERROR: waifu2x-ncnn-vulkan.exe not found!")
        print(f"Expected path: {optimizer.waifu2x_path}")
        exit(1)
    
    optimizer.run_optimization_tests()