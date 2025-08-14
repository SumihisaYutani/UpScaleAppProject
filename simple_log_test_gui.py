#!/usr/bin/env python3
"""
Simple Log Test GUI
Test the log display functionality in a minimal GUI
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

class SimpleLogTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Log Display Test")
        self.root.geometry("600x400")
        
        # Create GUI elements
        self.setup_gui()
        
        # Set up logging
        self.setup_logging()
        
    def setup_gui(self):
        # Button to trigger frame extraction
        self.extract_button = tk.Button(
            self.root, 
            text="Test Frame Extraction", 
            command=self.test_frame_extraction,
            font=("Arial", 12)
        )
        self.extract_button.pack(pady=10)
        
        # Log display area
        log_label = tk.Label(self.root, text="Log Messages:", font=("Arial", 10, "bold"))
        log_label.pack(anchor="w", padx=10)
        
        self.log_display = scrolledtext.ScrolledText(
            self.root, 
            height=20, 
            width=70,
            font=("Consolas", 9)
        )
        self.log_display.pack(fill="both", expand=True, padx=10, pady=5)
        
    def setup_logging(self):
        # Custom handler to send logs to GUI
        class GUIHandler(logging.Handler):
            def __init__(self, gui_callback):
                super().__init__()
                self.gui_callback = gui_callback
                self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.gui_callback(msg)
                except Exception:
                    pass
        
        # Set up the handler
        def add_log_message(message):
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\\n"
            
            # Update GUI in main thread
            self.root.after(0, lambda: self._insert_log(formatted_message))
        
        self.gui_handler = GUIHandler(add_log_message)
        self.gui_handler.setLevel(logging.INFO)
        
        # Add handler to relevant loggers
        loggers_to_track = [
            'modules.video_processor',
            'modules.waifu2x_processor',
            'enhanced_upscale_app'
        ]
        
        for logger_name in loggers_to_track:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self.gui_handler)
            logger.setLevel(logging.INFO)
        
        # Add initial message
        self.add_log_message("Log system initialized - ready for testing")
    
    def _insert_log(self, message):
        """Insert log message into GUI (called from main thread)"""
        self.log_display.insert(tk.END, message)
        self.log_display.see(tk.END)
        
    def add_log_message(self, message):
        """Add log message directly"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\\n"
        self._insert_log(formatted_message)
    
    def test_frame_extraction(self):
        """Test frame extraction in background thread"""
        self.extract_button.config(state="disabled", text="Processing...")
        
        def extraction_test():
            try:
                self.add_log_message("Starting frame extraction test...")
                
                # Import and test frame extraction
                from modules.video_processor import VideoFrameExtractor
                import tempfile
                
                temp_dir = tempfile.mkdtemp()
                extractor = VideoFrameExtractor(temp_dir)
                
                if os.path.exists('test_gui_video.mp4'):
                    self.add_log_message("Found test_gui_video.mp4, extracting frames...")
                    frames = extractor.extract_frames('test_gui_video.mp4')
                    self.add_log_message(f"Extraction completed! Got {len(frames)} frames")
                else:
                    self.add_log_message("test_gui_video.mp4 not found, testing manual logging...")
                    # Manual test
                    logger = logging.getLogger('modules.video_processor')
                    logger.info("Using FFmpeg for frame extraction...")
                    logger.info("FFmpeg successfully extracted 15 frames")
                
                # Clean up
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
                self.add_log_message("Test completed successfully!")
                
            except Exception as e:
                self.add_log_message(f"Error during test: {e}")
                import traceback
                self.add_log_message(f"Traceback: {traceback.format_exc()}")
            
            finally:
                # Re-enable button
                self.root.after(0, lambda: self.extract_button.config(
                    state="normal", text="Test Frame Extraction"
                ))
        
        # Run in background thread
        threading.Thread(target=extraction_test, daemon=True).start()
    
    def run(self):
        print("Starting Simple Log Test GUI...")
        print("Click the 'Test Frame Extraction' button to see logs in action!")
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleLogTestGUI()
    app.run()