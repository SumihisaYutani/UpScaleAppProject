#!/usr/bin/env python3
"""
UpScale App - Standalone Executable
Environment-independent video upscaling tool
Version: 2.0.0 (Executable Edition)
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# Detect if running as executable or script
if getattr(sys, 'frozen', False):
    # Running as PyInstaller executable
    BUNDLE_DIR = Path(sys.executable).parent
    RESOURCE_DIR = BUNDLE_DIR / "resources"
    IS_EXECUTABLE = True
else:
    # Running as script
    BUNDLE_DIR = Path(__file__).parent
    RESOURCE_DIR = BUNDLE_DIR / "resources"
    IS_EXECUTABLE = False

# Ensure resource directory exists
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Add resource paths to environment
os.environ['PATH'] = str(RESOURCE_DIR / "binaries") + os.pathsep + os.environ.get('PATH', '')
os.environ['RESOURCE_DIR'] = str(RESOURCE_DIR)
os.environ['BUNDLE_DIR'] = str(BUNDLE_DIR)

# Configure Python path
sys.path.insert(0, str(BUNDLE_DIR))

def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        import tkinter
        return True, ""
    except ImportError:
        return False, "GUI libraries not available"

def show_startup_error(message):
    """Show startup error dialog"""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("UpScale App - Startup Error", message)
    root.destroy()

def main():
    """Main application entry point"""
    try:
        print("UpScale App - Executable Edition v2.0.0")
        print(f"Bundle Directory: {BUNDLE_DIR}")
        print(f"Resource Directory: {RESOURCE_DIR}")
        print(f"Running Mode: {'Executable' if IS_EXECUTABLE else 'Script'}")
    except UnicodeEncodeError:
        # Fallback for systems with encoding issues
        print("UpScale App - Executable Edition v2.0.0")
        print("Bundle Directory: " + str(BUNDLE_DIR))
        print("Resource Directory: " + str(RESOURCE_DIR))
        print("Running Mode: " + ('Executable' if IS_EXECUTABLE else 'Script'))
    
    # Check system requirements
    requirements_ok, error_msg = check_system_requirements()
    if not requirements_ok:
        show_startup_error(f"System requirements not met:\n{error_msg}")
        return 1
    
    try:
        # Import and initialize main application
        from core.app import UpScaleApp
        
        # Create and run application
        app = UpScaleApp()
        return app.run()
        
    except ImportError as e:
        error_msg = f"Failed to import required modules:\n{e}\n\nThis may indicate a packaging issue."
        print(f"❌ Import Error: {e}")
        show_startup_error(error_msg)
        return 1
    
    except Exception as e:
        error_msg = f"Unexpected error during startup:\n{e}"
        print(f"💥 Startup Error: {e}")
        import traceback
        traceback.print_exc()
        show_startup_error(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())