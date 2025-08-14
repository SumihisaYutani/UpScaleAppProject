#!/usr/bin/env python3
"""
Simple Video Converter
Basic video upscaling without AI dependencies
"""

import sys
import subprocess
import os
from pathlib import Path

def simple_upscale(input_file, output_file, scale_factor=2.0):
    """
    Simple video upscaling using FFmpeg
    """
    print(f"Converting {input_file} with scale factor {scale_factor}x")
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"ERROR: Input file {input_file} not found")
            return False
        
        # Build FFmpeg command for simple upscaling
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vf', f'scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos',
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-y',  # Overwrite output file
            output_file
        ]
        
        print("Running FFmpeg command:")
        print(" ".join(cmd))
        print()
        
        # Execute FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: Video converted successfully!")
            print(f"Output saved to: {output_file}")
            
            # Show file sizes
            input_size = os.path.getsize(input_file) / (1024 * 1024)
            output_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Input size: {input_size:.1f} MB")
            print(f"Output size: {output_size:.1f} MB")
            
            return True
        else:
            print("ERROR: FFmpeg conversion failed")
            print("STDERR:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("ERROR: FFmpeg not found. Please install FFmpeg:")
        print("  - Download from: https://ffmpeg.org/download.html")
        print("  - Or install via package manager")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_convert.py <input_file> [output_file] [scale_factor]")
        print("Example: python simple_convert.py test.mp4 test_upscaled.mp4 2.0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"upscaled_{input_file}"
    scale_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    
    success = simple_upscale(input_file, output_file, scale_factor)
    sys.exit(0 if success else 1)