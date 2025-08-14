#!/usr/bin/env python3
"""
Test GUI Log Functionality
Simple test to verify that log messages appear in the GUI
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_frame_extraction():
    """Test frame extraction with GUI logging simulation"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Import after path setup
    from modules.video_processor import VideoFrameExtractor
    import tempfile
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory: {temp_dir}")
    
    # Create extractor
    extractor = VideoFrameExtractor(temp_dir)
    
    # Test with existing test video
    if os.path.exists('test_video.mp4'):
        print("\\n=== Testing Frame Extraction ===")
        print("This will show the log messages that would appear in the GUI:")
        print("-" * 50)
        
        # Extract frames (this will generate the log messages we want to see)
        frames = extractor.extract_frames('test_video.mp4')
        
        print("-" * 50)
        print(f"[SUCCESS] Successfully extracted {len(frames)} frames")
        print("\\nThe messages above would now appear in the GUI log area!")
        
        # Clean up a few test frames to save space
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        print("[ERROR] test_video.mp4 not found. Please run the video creation script first.")

if __name__ == "__main__":
    test_frame_extraction()