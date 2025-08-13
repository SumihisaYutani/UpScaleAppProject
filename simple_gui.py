#!/usr/bin/env python3
"""
Simple UpScale App GUI - Lightweight version without AI dependencies
For testing GUI functionality without heavy ML dependencies
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI dependencies not available. Install with: pip install -r requirements_gui.txt")

# Configure CustomTkinter
if GUI_AVAILABLE:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")


class SimpleUpScaleGUI:
    """Simple GUI for UpScale App without AI dependencies"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available")
        
        self._setup_window()
        self._setup_ui()
        
    def _setup_window(self):
        """Setup main window"""
        self.root = ctk.CTk()
        self.root.title("UpScale App - Simple GUI v0.1.0")
        self.root.geometry("600x500")
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def _setup_ui(self):
        """Setup user interface"""
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="UpScale App - Simple GUI", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Basic Video File Processing Tool",
            font=ctk.CTkFont(size=14)
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # File selection section
        self._setup_file_section()
        
        # Settings section
        self._setup_settings_section()
        
        # Actions section
        self._setup_actions_section()
        
        # Status section
        self._setup_status_section()
        
    def _setup_file_section(self):
        """Setup file selection section"""
        
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        # Input file
        input_label = ctk.CTkLabel(file_frame, text="Input Video:", font=ctk.CTkFont(weight="bold"))
        input_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        input_container = ctk.CTkFrame(file_frame)
        input_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.input_entry = ctk.CTkEntry(
            input_container, 
            placeholder_text="Select a video file...",
            height=35
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        self.browse_button = ctk.CTkButton(
            input_container,
            text="Browse",
            command=self._browse_input_file,
            width=80
        )
        self.browse_button.pack(side="right", padx=(5, 10), pady=10)
        
    def _setup_settings_section(self):
        """Setup settings section"""
        
        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        settings_label = ctk.CTkLabel(settings_frame, text="Settings", font=ctk.CTkFont(size=16, weight="bold"))
        settings_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Settings container
        settings_container = ctk.CTkFrame(settings_frame)
        settings_container.pack(fill="x", padx=15, pady=(0, 15))
        
        # Scale factor
        ctk.CTkLabel(settings_container, text="Scale Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.scale_var = ctk.StringVar(value="1.5x")
        scale_menu = ctk.CTkComboBox(
            settings_container,
            values=["1.2x", "1.5x", "2.0x", "2.5x"],
            variable=self.scale_var,
            state="readonly",
            width=100
        )
        scale_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Processing mode
        ctk.CTkLabel(settings_container, text="Mode:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.mode_var = ctk.StringVar(value="Simple")
        mode_menu = ctk.CTkComboBox(
            settings_container,
            values=["Simple", "Basic Processing"],
            variable=self.mode_var,
            state="readonly",
            width=120
        )
        mode_menu.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
    def _setup_actions_section(self):
        """Setup actions section"""
        
        actions_frame = ctk.CTkFrame(self.main_frame)
        actions_frame.pack(fill="x", padx=20, pady=10)
        
        actions_label = ctk.CTkLabel(actions_frame, text="Actions", font=ctk.CTkFont(size=16, weight="bold"))
        actions_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Button container
        button_container = ctk.CTkFrame(actions_frame)
        button_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.analyze_button = ctk.CTkButton(
            button_container,
            text="Analyze Video",
            command=self._analyze_video,
            height=35,
            state="disabled"
        )
        self.analyze_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        self.test_button = ctk.CTkButton(
            button_container,
            text="Test FFmpeg",
            command=self._test_ffmpeg,
            height=35,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90")
        )
        self.test_button.pack(side="right", padx=10, pady=10)
        
    def _setup_status_section(self):
        """Setup status section"""
        
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        status_label = ctk.CTkLabel(status_frame, text="Status", font=ctk.CTkFont(size=16, weight="bold"))
        status_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Status display
        self.status_text = ctk.CTkTextbox(status_frame, height=100)
        self.status_text.pack(fill="x", padx=15, pady=(0, 15))
        self.status_text.insert("0.0", "Ready. Select a video file to begin.")
        
    def _browse_input_file(self):
        """Browse for input file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)
            self.analyze_button.configure(state="normal")
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", f"Selected: {os.path.basename(file_path)}")
            
    def _analyze_video(self):
        """Analyze selected video"""
        input_path = self.input_entry.get().strip()
        
        if not input_path or not os.path.exists(input_path):
            CTkMessagebox(
                title="Error",
                message="Please select a valid video file",
                icon="cancel"
            )
            return
            
        try:
            # Basic file info
            file_size = os.path.getsize(input_path) / (1024*1024)  # MB
            
            # Try to get video info with opencv
            try:
                import cv2
                cap = cv2.VideoCapture(input_path)
                
                if not cap.isOpened():
                    raise Exception("Could not open video file")
                    
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                # Calculate target resolution
                scale_factor = float(self.scale_var.get().rstrip('x'))
                target_width = int(width * scale_factor)
                target_height = int(height * scale_factor)
                
                info_text = f"""Video Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB
Resolution: {width}x{height}
Frame Rate: {fps:.2f} fps
Duration: {duration:.2f} seconds
Frames: {frame_count}

Upscaling Preview:
Scale Factor: {self.scale_var.get()}
Target Resolution: {target_width}x{target_height}
Processing Mode: {self.mode_var.get()}

Status: Analysis complete - Ready for processing"""
                
            except ImportError:
                info_text = f"""Basic File Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB

Note: Install opencv-python for detailed video analysis
Scale Factor: {self.scale_var.get()}
Processing Mode: {self.mode_var.get()}

Status: Basic analysis complete"""
                
            except Exception as e:
                info_text = f"""File Information:
File: {os.path.basename(input_path)}
Size: {file_size:.1f} MB

Error analyzing video: {str(e)}
This may not be a valid video file or the codec is not supported.

Status: Analysis failed"""
            
            self.status_text.delete("0.0", "end")
            self.status_text.insert("0.0", info_text)
            
        except Exception as e:
            CTkMessagebox(
                title="Analysis Error",
                message=f"Failed to analyze video: {str(e)}",
                icon="cancel"
            )
            
    def _test_ffmpeg(self):
        """Test FFmpeg availability"""
        try:
            import subprocess
            
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                CTkMessagebox(
                    title="FFmpeg Test",
                    message=f"FFmpeg is available!\n\n{version_line}",
                    icon="check"
                )
            else:
                CTkMessagebox(
                    title="FFmpeg Test",
                    message="FFmpeg found but returned error",
                    icon="warning"
                )
                
        except FileNotFoundError:
            CTkMessagebox(
                title="FFmpeg Test",
                message="FFmpeg not found in PATH.\n\nPlease install FFmpeg to enable video processing.",
                icon="cancel"
            )
        except Exception as e:
            CTkMessagebox(
                title="FFmpeg Test",
                message=f"Error testing FFmpeg: {str(e)}",
                icon="cancel"
            )
            
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    if not GUI_AVAILABLE:
        print("GUI dependencies not available")
        print("Install GUI dependencies with:")
        print("   pip install -r requirements_gui.txt")
        return 1
        
    try:
        app = SimpleUpScaleGUI()
        app.run()
        return 0
        
    except Exception as e:
        print(f"GUI Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())