"""
UpScale App - Main Application Class
Consolidated application for executable distribution
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import tkinter as tk

# Configure logging for debug version with file output
def setup_logging():
    """Set up logging to both console and file"""
    import datetime
    
    # Create logs directory
    logs_dir = Path.home() / "UpScaleApp_Logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"upscale_app_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log file: {log_file}")
    return log_file

log_file_path = setup_logging()
logger = logging.getLogger(__name__)

class UpScaleApp:
    """Main application class for standalone executable"""
    
    def __init__(self):
        # Get environment paths
        self.bundle_dir = Path(os.environ.get('BUNDLE_DIR', '.'))
        self.resource_dir = Path(os.environ.get('RESOURCE_DIR', './resources'))
        
        # Initialize components
        self.video_processor = None
        self.ai_processor = None
        self.gpu_detector = None
        self.gui = None
        
        # Application state
        self.temp_dir = None
        self.is_running = False
        
        logger.info(f"UpScale App initialized - Bundle: {self.bundle_dir}")
        
    def initialize(self):
        """Initialize all application components"""
        try:
            # Create temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="upscale_"))
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Import and initialize core modules
            from .utils import ResourceManager
            from .video_processor import VideoProcessor  
            from .ai_processor import AIProcessor
            from .gpu_detector import GPUDetector
            from .session_manager import SessionManager
            from .gui import MainGUI
            
            # Initialize resource manager
            self.resource_manager = ResourceManager(self.resource_dir)
            
            # Initialize GPU detection
            self.gpu_detector = GPUDetector(self.resource_manager)
            gpu_info = self.gpu_detector.detect_gpus()
            logger.info(f"GPU Detection: {gpu_info}")
            
            # Initialize processors with GPU support
            self.video_processor = VideoProcessor(
                self.resource_manager, 
                self.temp_dir,
                gpu_info
            )
            
            # Initialize AI processor with automatic backend selection
            self.ai_processor = AIProcessor(
                self.resource_manager,
                gpu_info
            )
            
            # Initialize session manager
            self.session_manager = SessionManager()
            logger.info("SessionManager initialized")
            
            # Initialize GUI
            self.gui = MainGUI(
                video_processor=self.video_processor,
                ai_processor=self.ai_processor,
                gpu_info=gpu_info,
                session_manager=self.session_manager,
                log_file_path=log_file_path  # ログファイルパスを渡す
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self) -> int:
        """Run the application"""
        try:
            logger.info("Starting UpScale App...")
            
            # Initialize tkinter environment first
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the root window initially
            
            # Initialize components
            if not self.initialize():
                root.destroy()
                return 1
            
            # Start GUI
            self.is_running = True
            logger.info("Launching GUI...")
            
            result = self.gui.run()
            root.destroy()
            return result
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            return 0
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error dialog
            try:
                root = tk.Tk()
                root.withdraw()
                from tkinter import messagebox
                messagebox.showerror(
                    "UpScale App Error", 
                    f"An error occurred:\n{e}\n\nCheck the log file for details."
                )
                root.destroy()
            except:
                pass
            
            return 1
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up application resources...")
            
            # Cleanup processors
            if self.ai_processor:
                self.ai_processor.cleanup()
            
            if self.video_processor:
                self.video_processor.cleanup()
            
            # Cleanup GUI
            if self.gui:
                self.gui.cleanup()
            
            # Cleanup temp directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            
            self.is_running = False
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "app_version": "2.0.0",
            "bundle_dir": str(self.bundle_dir),
            "resource_dir": str(self.resource_dir),
            "temp_dir": str(self.temp_dir) if self.temp_dir else None,
            "platform": sys.platform,
            "python_version": sys.version,
        }
        
        # Add GPU info if available
        if self.gpu_detector:
            info["gpu_info"] = self.gpu_detector.get_gpu_summary()
        
        return info