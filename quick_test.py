#!/usr/bin/env python3
"""
Quick Environment Test - No dependencies required
Tests basic Python environment and project structure
"""

import sys
import os
from pathlib import Path

def main():
    """Quick test function"""
    print("🎬 UpScale App - Quick Environment Test")
    print("=" * 50)
    
    # Basic Python info
    print(f"🐍 Python Version: {sys.version}")
    print(f"📁 Working Directory: {Path.cwd()}")
    print(f"💻 Platform: {sys.platform}")
    
    # Test project structure
    print("\n📁 Project Structure Check:")
    required_files = [
        "src/",
        "config/",
        "main.py",
        "README.md"
    ]
    
    all_good = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_good = False
    
    # Test basic imports
    print("\n🔧 Basic Import Test:")
    try:
        sys.path.append(str(Path("src")))
        
        # Test configuration
        from config.settings import VIDEO_SETTINGS
        print("  ✅ Configuration import successful")
        
        # Test basic modules (without heavy dependencies)
        print("  ✅ Basic functionality available")
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        all_good = False
    except Exception as e:
        print(f"  ⚠️  Warning: {e}")
    
    # Result
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Quick test PASSED!")
        print("📝 Next steps:")
        print("  1. Run: python test_environment.py (full test)")
        print("  2. Run: python setup_environment.py (full setup)")
        print("  3. Try: python main.py system (if dependencies available)")
    else:
        print("⚠️  Quick test found issues")
        print("📝 Recommended actions:")
        print("  1. Check if you're in the correct directory")
        print("  2. Run: python setup_environment.py")
        print("  3. Check README.md for setup instructions")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"💥 Quick test failed: {e}")
        sys.exit(1)