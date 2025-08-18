#!/usr/bin/env python3
"""
UpScale App - Simple Executable Test
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def main():
    """Simple test application"""
    print("UpScale App - Resume Functionality v2.0.0")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {Path.cwd()}")
    print(f"Script Path: {Path(__file__).parent}")
    
    # Test GUI availability
    try:
        root = tk.Tk()
        root.withdraw()
        
        messagebox.showinfo(
            "UpScale App", 
            "UpScale App Resume Functionality\n\n"
            "Version: 2.0.0\n"
            "Build: Resume Feature Branch\n\n"
            "✅ Executable test successful!"
        )
        
        root.destroy()
        print("✅ GUI test successful")
        return 0
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())