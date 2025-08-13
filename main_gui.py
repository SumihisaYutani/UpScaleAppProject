#!/usr/bin/env python3
"""
UpScale App GUI - Main Entry Point
Graphical User Interface for AI Video Upscaling
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from gui.main_window import main, GUI_AVAILABLE
    
    if __name__ == "__main__":
        if not GUI_AVAILABLE:
            print("UpScale App GUI")
            print("=" * 50)
            print("GUI dependencies not available")
            print("")
            print("To use the GUI version, install dependencies:")
            print("   pip install -r requirements_gui.txt")
            print("")
            print("Alternative options:")
            print("   python main.py upscale video.mp4          # Basic CLI")
            print("   python main_enhanced.py upscale video.mp4  # Enhanced CLI")
            print("")
            print("For setup help:")
            print("   python quick_test.py                       # Quick environment test")
            print("   python setup_environment.py               # Full setup")
            sys.exit(1)
        
        print("Starting UpScale App GUI...")
        sys.exit(main())
        
except ImportError as e:
    print("UpScale App GUI")
    print("=" * 50)
    print(f"Import Error: {e}")
    print("")
    print("Possible solutions:")
    print("   1. Install GUI dependencies: pip install -r requirements_gui.txt")
    print("   2. Check environment: python test_environment.py")
    print("   3. Run setup: python setup_environment.py")
    print("")
    print("Alternative: Use CLI version")
    print("   python main.py --help")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)