"""
UpScale App - GUI Module
Modern GUI using CustomTkinter for better compatibility
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    # Fallback to standard tkinter
    from tkinter import ttk, messagebox, scrolledtext
    ctk = None

from .utils import ProgressCallback, SimpleTimer, format_duration
from .session_manager import SessionManager
from .resume_dialog import ResumeDialog, SessionSelectionDialog

logger = logging.getLogger(__name__)

# Configure CustomTkinter
if GUI_AVAILABLE and ctk:
    logger.info("CustomTkinter loaded successfully - Modern GUI enabled")
    ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
else:
    logger.warning("CustomTkinter not available - Falling back to standard Tkinter")

class ProcessingStepTracker:
    """実行ステップごとのステータス追跡クラス"""
    
    STEPS = [
        {"id": "validate", "name": "動画検証", "emoji": "🔍", "weight": 5},
        {"id": "extract", "name": "フレーム抽出", "emoji": "📸", "weight": 15}, 
        {"id": "upscale", "name": "AI処理", "emoji": "🤖", "weight": 70},
        {"id": "combine", "name": "動画結合", "emoji": "🎬", "weight": 10}
    ]
    
    def __init__(self):
        self.current_step = 0
        self.step_progress = {}
        self.step_timers = {}
        self.step_status = {}
        
        # Initialize all steps
        for i, step in enumerate(self.STEPS):
            self.step_progress[step["id"]] = 0.0
            self.step_timers[step["id"]] = None
            self.step_status[step["id"]] = "pending"  # pending, active, completed, error
    
    def start_step(self, step_id: str):
        """指定ステップの開始"""
        import time
        if step_id in self.step_timers:
            self.step_timers[step_id] = time.time()
            self.step_status[step_id] = "active"
            
    def update_step(self, step_id: str, progress: float):
        """ステップ内の進捗更新"""
        if step_id in self.step_progress:
            self.step_progress[step_id] = min(100.0, max(0.0, progress))
            
    def complete_step(self, step_id: str):
        """ステップ完了"""
        import time
        if step_id in self.step_status:
            self.step_status[step_id] = "completed"
            self.step_progress[step_id] = 100.0
            if self.step_timers[step_id]:
                elapsed = time.time() - self.step_timers[step_id]
                self.step_timers[step_id] = elapsed
    
    def error_step(self, step_id: str):
        """ステップエラー"""
        if step_id in self.step_status:
            self.step_status[step_id] = "error"
    
    def get_overall_progress(self) -> float:
        """全体の進捗計算"""
        total_weight = sum(step["weight"] for step in self.STEPS)
        completed_weight = 0.0
        
        for step in self.STEPS:
            step_id = step["id"]
            if self.step_status[step_id] == "completed":
                completed_weight += step["weight"]
            elif self.step_status[step_id] == "active":
                step_progress = self.step_progress[step_id] / 100.0
                completed_weight += step["weight"] * step_progress
                
        return (completed_weight / total_weight) * 100.0
    
    def get_step_info(self, step_id: str) -> dict:
        """ステップ情報取得"""
        for step in self.STEPS:
            if step["id"] == step_id:
                return {
                    "name": step["name"],
                    "emoji": step["emoji"],
                    "progress": self.step_progress[step_id],
                    "status": self.step_status[step_id],
                    "timer": self.step_timers[step_id]
                }
        return {}

class ProgressDialog:
    """Progress dialog for processing operations"""
    
    def __init__(self, parent, temp_dir=None, log_file_path=None):
        logger.info("DEBUG: ProgressDialog.__init__ called")
        self.parent = parent
        self.step_tracker = ProcessingStepTracker()
        self.temp_dir = temp_dir  # Store temp directory for cleanup
        self.video_processor = None  # Store video processor reference for cleanup
        self.log_file_path = log_file_path  # Store log file path
        
        # Initialize log messages list and window
        self.log_messages = []
        self.log_window = None
        
        logger.info(f"DEBUG: GUI_AVAILABLE={GUI_AVAILABLE}, ctk={ctk is not None}")
        if GUI_AVAILABLE and ctk:
            # Use CustomTkinter
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Processing Video")
            self.window.geometry("875x704")  # 横幅は元のまま875、縦幅だけ938 * 0.75 = 704
            self.window.resizable(False, False)
            
            # Make modal
            self.window.transient(parent)
            self.window.grab_set()
            
            # Center on parent
            parent.update_idletasks()
            x = (parent.winfo_x() + parent.winfo_width() // 2) - 437  # 875/2 = 437.5
            y = (parent.winfo_y() + parent.winfo_height() // 2) - 352  # 704/2 = 352
            self.window.geometry(f"+{x}+{y}")
        else:
            # Fallback to standard tkinter
            self.window = tk.Toplevel(parent)
            self.window.title("Processing Video")
            self.window.geometry("675x630")  # 横幅は元のまま675、縦幅だけ840 * 0.75 = 630
            self.window.resizable(False, False)
            
            # Make modal
            self.window.transient(parent)
            self.window.grab_set()
            
            # Center on parent
            parent.update_idletasks()
            x = parent.winfo_x() + (parent.winfo_width() // 2) - 337  # 675/2 = 337.5
            y = parent.winfo_y() + (parent.winfo_height() // 2) - 315  # 630/2 = 315
            self.window.geometry(f"+{x}+{y}")
        
        self.cancelled = False
        self.current_process = None  # Track current subprocess
        logger.info("DEBUG: About to call _setup_ui()")
        try:
            self._setup_ui()
            logger.info("DEBUG: _setup_ui() completed successfully")
        except Exception as e:
            logger.error(f"DEBUG: Error in _setup_ui(): {e}")
            import traceback
            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        
    def _setup_ui(self):
        """Setup progress dialog UI"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Main frame
            self.main_frame = ctk.CTkFrame(self.window)
            self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            self.title_label = ctk.CTkLabel(
                self.main_frame, 
                text="🎬 Processing Video with AI Upscaling", 
                font=ctk.CTkFont(size=18, weight="bold")
            )
            self.title_label.pack(pady=(10, 20))
            
            # Overall progress section
            overall_frame = ctk.CTkFrame(self.main_frame)
            overall_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            # Overall status label
            self.status_label = ctk.CTkLabel(
                overall_frame, 
                text="Initializing...",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            self.status_label.pack(pady=(10, 5))
            
            # Overall progress bar
            self.progress_bar = ctk.CTkProgressBar(overall_frame)
            self.progress_bar.pack(fill="x", padx=20, pady=5)
            self.progress_bar.set(0)
            
            # Overall progress percentage
            self.progress_label = ctk.CTkLabel(
                overall_frame, 
                text="0%",
                font=ctk.CTkFont(size=12)
            )
            self.progress_label.pack(pady=(0, 10))
            
            # GPU Status section
            self.gpu_frame = ctk.CTkFrame(self.main_frame)
            self.gpu_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            # GPU Status title
            gpu_title = ctk.CTkLabel(
                self.gpu_frame,
                text="🔥 GPU Processing Status",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            gpu_title.pack(pady=(10, 5))
            
            # GPU indicator container
            gpu_indicator_frame = ctk.CTkFrame(self.gpu_frame)
            gpu_indicator_frame.pack(fill="x", padx=10, pady=5)
            
            # GPU Status indicator (red lamp when active)
            self.gpu_status_label = ctk.CTkLabel(
                gpu_indicator_frame,
                text="⚫ GPU: Idle",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="gray"
            )
            self.gpu_status_label.pack(side="left", padx=10, pady=5)
            
            # GPU utilization bar
            self.gpu_utilization_bar = ctk.CTkProgressBar(gpu_indicator_frame)
            self.gpu_utilization_bar.pack(side="right", fill="x", expand=True, padx=10, pady=5)
            self.gpu_utilization_bar.set(0)
            
            # GPU details label
            self.gpu_details_label = ctk.CTkLabel(
                self.gpu_frame,
                text="Ready for AI processing...",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            self.gpu_details_label.pack(pady=(0, 10))
            
            # Step progress section
            steps_frame = ctk.CTkFrame(self.main_frame)
            steps_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            steps_title = ctk.CTkLabel(
                steps_frame,
                text="📋 Processing Steps",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            steps_title.pack(pady=(10, 5))
            
            # Create step displays
            self.step_displays = {}
            for step in self.step_tracker.STEPS:
                step_container = ctk.CTkFrame(steps_frame)
                step_container.pack(fill="x", padx=10, pady=2)
                
                # Step info frame (left side)
                info_frame = ctk.CTkFrame(step_container)
                info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
                
                # Step name with emoji and status
                step_label = ctk.CTkLabel(
                    info_frame,
                    text=f"{step['emoji']} {step['name']}",
                    font=ctk.CTkFont(size=12),
                    anchor="w"
                )
                step_label.pack(side="left", padx=5)
                
                # Step status indicator
                status_label = ctk.CTkLabel(
                    info_frame,
                    text="⏳",
                    font=ctk.CTkFont(size=12)
                )
                status_label.pack(side="right", padx=5)
                
                # Step progress bar (right side)
                progress_frame = ctk.CTkFrame(step_container)
                progress_frame.pack(side="right", padx=5, pady=5)
                
                step_progress = ctk.CTkProgressBar(progress_frame, width=100)
                step_progress.pack(padx=5, pady=2)
                step_progress.set(0)
                
                step_percent = ctk.CTkLabel(
                    progress_frame,
                    text="0%",
                    font=ctk.CTkFont(size=10)
                )
                step_percent.pack(padx=5)
                
                self.step_displays[step["id"]] = {
                    "container": step_container,
                    "status_label": status_label,
                    "progress_bar": step_progress,
                    "percent_label": step_percent
                }
            
            steps_frame.pack(pady=(0, 10))
            
            # Buttons section - horizontal layout to save space
            buttons_frame = ctk.CTkFrame(self.main_frame)
            buttons_frame.pack(fill="x", padx=10, pady=(5, 10))
            
            # ログ表示ボタン（左側）
            self.log_button = ctk.CTkButton(
                buttons_frame,
                text="📋 詳細ログ",
                command=self._show_log_window,
                width=120,
                height=28
            )
            self.log_button.pack(side="left", padx=(10, 5), pady=8)
            
            # ログファイルを開くボタン（中央）
            if self.log_file_path:
                self.open_log_button = ctk.CTkButton(
                    buttons_frame,
                    text="📁 ログファイル",
                    command=self._open_log_file,
                    width=120,
                    height=28
                )
                self.open_log_button.pack(side="left", padx=5, pady=8)
            
            # Cancel button（右側、目立つ色）
            self.cancel_button = ctk.CTkButton(
                buttons_frame,
                text="❌ キャンセル",
                command=self._on_cancel,
                fg_color="#dc3545",
                hover_color="#c82333",
                width=120,
                height=28
            )
            self.cancel_button.pack(side="right", padx=(5, 10), pady=8)
        else:
            # Standard tkinter fallback
            main_frame = ttk.Frame(self.window)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            title_label = ttk.Label(
                main_frame, 
                text="Processing Video with AI Upscaling",
                font=("Arial", 14, "bold")
            )
            title_label.pack(pady=(0, 15))
            
            # Status
            self.status_var = tk.StringVar(value="Initializing...")
            self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
            self.status_label.pack(pady=5)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                main_frame, 
                variable=self.progress_var,
                maximum=100
            )
            self.progress_bar.pack(fill="x", pady=10)
            
            # Progress percentage
            self.progress_text_var = tk.StringVar(value="0%")
            progress_text_label = ttk.Label(main_frame, textvariable=self.progress_text_var)
            progress_text_label.pack(pady=5)
            
            # GPU Status section (Standard Tkinter)
            gpu_frame = ttk.LabelFrame(main_frame, text="GPU Processing Status", padding=10)
            gpu_frame.pack(fill="x", pady=(10, 0))
            
            # GPU Status indicator
            self.gpu_status_var = tk.StringVar(value="GPU: Idle")
            self.gpu_status_label = ttk.Label(
                gpu_frame, 
                textvariable=self.gpu_status_var,
                font=("Arial", 10, "bold"),
                foreground="gray"
            )
            self.gpu_status_label.pack(pady=2)
            
            # GPU utilization bar (standard)
            self.gpu_utilization_var = tk.DoubleVar()
            self.gpu_utilization_bar = ttk.Progressbar(
                gpu_frame,
                variable=self.gpu_utilization_var,
                maximum=100
            )
            self.gpu_utilization_bar.pack(fill="x", pady=2)
            
            # GPU details
            self.gpu_details_var = tk.StringVar(value="Ready for AI processing...")
            self.gpu_details_label = ttk.Label(
                gpu_frame,
                textvariable=self.gpu_details_var,
                font=("Arial", 9),
                foreground="gray"
            )
            self.gpu_details_label.pack(pady=2)
            
            # Log button
            log_button_frame = ttk.Frame(main_frame)
            log_button_frame.pack(fill="x", pady=(10, 0))
            
            self.log_button = ttk.Button(
                log_button_frame,
                text="詳細ログを表示",
                command=self._show_log_window
            )
            self.log_button.pack(pady=10)
            
            # Initialize log window as None
            self.log_window = None
            self.log_messages = []
            
            # Cancel button
            self.cancel_button = ttk.Button(
                main_frame,
                text="Cancel",
                command=self._on_cancel
            )
            self.cancel_button.pack(pady=(15, 0))
        
    def update_progress(self, progress: float, message: str):
        """Update progress display"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.progress_bar.set(progress / 100)
            self.progress_label.configure(text=f"{progress:.1f}%")
            self.status_label.configure(text=message)
            # Update window
            self.window.update()
        else:
            # Standard tkinter fallback
            self.progress_var.set(progress)
            self.progress_text_var.set(f"{progress:.1f}%")
            self.status_var.set(message)
            self.window.update_idletasks()
    
    def update_gpu_status(self, is_active: bool, utilization: float = 0, details: str = None):
        """Update GPU status indicator"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            if is_active:
                self.gpu_status_label.configure(
                    text="🔴 GPU: Processing",
                    text_color="red"
                )
                self.gpu_utilization_bar.set(utilization / 100)
            else:
                self.gpu_status_label.configure(
                    text="⚫ GPU: Idle",
                    text_color="gray"
                )
                self.gpu_utilization_bar.set(0)
            
            if details:
                self.gpu_details_label.configure(text=details)
                
            self.window.update_idletasks()
        else:
            # Standard tkinter fallback
            if is_active:
                self.gpu_status_var.set("🔴 GPU: Processing")
                self.gpu_status_label.configure(foreground="red")
                self.gpu_utilization_var.set(utilization)
            else:
                self.gpu_status_var.set("⚫ GPU: Idle")
                self.gpu_status_label.configure(foreground="gray")
                self.gpu_utilization_var.set(0)
            
            if details:
                self.gpu_details_var.set(details)
                
            self.window.update_idletasks()
    
    def update_step_progress(self, step_id: str, progress: float, status: str = None):
        """Update individual step progress"""
        if step_id not in self.step_displays:
            return
            
        # Update step tracker
        if status == "start":
            self.step_tracker.start_step(step_id)
        elif status == "complete":
            self.step_tracker.complete_step(step_id)
        elif status == "error":
            self.step_tracker.error_step(step_id)
        else:
            self.step_tracker.update_step(step_id, progress)
        
        # Update UI
        if GUI_AVAILABLE and ctk:
            display = self.step_displays[step_id]
            
            # Update progress bar and percentage
            display["progress_bar"].set(progress / 100)
            display["percent_label"].configure(text=f"{progress:.0f}%")
            
            # Update status indicator
            step_status = self.step_tracker.step_status[step_id]
            if step_status == "pending":
                display["status_label"].configure(text="⏳")
            elif step_status == "active":
                display["status_label"].configure(text="🔄")
            elif step_status == "completed":
                display["status_label"].configure(text="✅")
            elif step_status == "error":
                display["status_label"].configure(text="❌")
            
            # Update overall progress
            overall_progress = self.step_tracker.get_overall_progress()
            self.progress_bar.set(overall_progress / 100)
            self.progress_label.configure(text=f"{overall_progress:.1f}%")
            
            # Update window
            self.window.update()
        
    def add_log_message(self, message: str):
        """Add log message to display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Store message in list
        self.log_messages.append(formatted_message)
        
        # Update log window if it's open
        if self.log_window and hasattr(self.log_window, 'log_text'):
            try:
                self.log_window.log_text.insert("end", formatted_message + "\n")
                self.log_window.log_text.see("end")
                self.log_window.update()
            except:
                pass  # Window might be closed
    
    def _show_log_window(self):
        """Show separate log window"""
        try:
            logger.info("DEBUG: _show_log_window called")
            
            # Check if log window already exists and is valid
            if hasattr(self, 'log_window') and self.log_window is not None:
                try:
                    # Try to check if window still exists
                    if self.log_window.winfo_exists():
                        logger.info("DEBUG: Log window already exists, bringing to front")
                        self.log_window.lift()
                        self.log_window.focus_set()
                        return
                except (AttributeError, tk.TclError):
                    logger.info("DEBUG: Previous log window was destroyed")
                    self.log_window = None
            
            # Create new window only if none exists
            logger.info("DEBUG: Creating new log window")
            if GUI_AVAILABLE and ctk:
                # CustomTkinter version
                self.log_window = ctk.CTkToplevel(self.window)
                self.log_window.title("処理ログ")
                self.log_window.geometry("600x400")
                
                # Simple positioning
                self.log_window.geometry("+100+100")
                
                # Log display
                log_frame = ctk.CTkFrame(self.log_window)
                log_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                log_label = ctk.CTkLabel(
                    log_frame,
                    text="📋 処理ログメッセージ",
                    font=ctk.CTkFont(size=14, weight="bold")
                )
                log_label.pack(pady=(5, 10))
                
                # Scrollable text area
                self.log_window.log_text = ctk.CTkTextbox(
                    log_frame,
                    font=ctk.CTkFont(size=10, family="Courier")
                )
                self.log_window.log_text.pack(fill="both", expand=True, padx=5, pady=5)
                
            else:
                # Standard tkinter version
                import tkinter as tk
                from tkinter import ttk, scrolledtext
                
                self.log_window = tk.Toplevel(self.window)
                self.log_window.title("処理ログ")
                self.log_window.geometry("600x400+120+120")
                
                # Log display
                log_frame = ttk.LabelFrame(self.log_window, text="処理ログメッセージ", padding=10)
                log_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                self.log_window.log_text = scrolledtext.ScrolledText(
                    log_frame,
                    font=("Courier", 9)
                )
                self.log_window.log_text.pack(fill="both", expand=True)
            
            # Set up window close callback to clean up reference
            def on_log_window_close():
                logger.info("DEBUG: Log window closed by user")
                self.log_window = None
            
            self.log_window.protocol("WM_DELETE_WINDOW", on_log_window_close)
            
            # Add messages
            if hasattr(self, 'log_messages') and self.log_messages:
                for msg in self.log_messages:
                    self.log_window.log_text.insert("end", msg + "\n")
                self.log_window.log_text.see("end")
            else:
                self.log_window.log_text.insert("end", "[処理開始前] ログメッセージはありません\n")
            
            logger.info("DEBUG: Log window created successfully")
            
        except Exception as e:
            logger.error(f"DEBUG: Error in _show_log_window: {e}")
            # Simple fallback message
            try:
                import tkinter.messagebox as messagebox
                messagebox.showinfo("ログウィンドウ", f"ログウィンドウの表示でエラーが発生しました。\nログファイルを直接確認してください。")
            except:
                pass
        
    def _on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        
        # Update button state
        if GUI_AVAILABLE and ctk:
            self.cancel_button.configure(text="Cancelling...", state="disabled")
        else:
            self.cancel_button.config(text="Cancelling...", state="disabled")
        
        # Kill any running subprocess forcefully
        if self.current_process and self.current_process.poll() is None:
            try:
                # Try graceful termination first
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if terminate didn't work
                    self.current_process.kill()
                    self.current_process.wait(timeout=3)
                    
                # Also kill any child processes on Windows
                import os
                import signal
                if os.name == 'nt':  # Windows
                    try:
                        import psutil
                        parent = psutil.Process(self.current_process.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()
                    except:
                        # Fallback: use taskkill on Windows
                        try:
                            import subprocess
                            # Hide console window when running taskkill
                            startupinfo = subprocess.STARTUPINFO()
                            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                            startupinfo.wShowWindow = subprocess.SW_HIDE
                            
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)], 
                                         capture_output=True, timeout=5, startupinfo=startupinfo)
                        except:
                            pass
            except Exception as e:
                print(f"Error killing process: {e}")
        
        # Kill all FFmpeg processes as final measure
        try:
            import subprocess
            # Hide console window when running taskkill
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], capture_output=True, timeout=5, startupinfo=startupinfo)
        except:
            pass
        
        # Clean up temporary files (preserve frames for resume on error/cancel)
        self._cleanup_temp_files(preserve_frames=True)
        
        # Notify main GUI about cancellation BEFORE closing dialog
        if hasattr(self, 'main_gui') and self.main_gui:
            self.main_gui._reset_processing_state()
        
        # Close progress dialog after short delay
        def close_dialog():
            try:
                self.destroy()
            except:
                pass
        
        if GUI_AVAILABLE and ctk:
            self.window.after(1000, close_dialog)  # Reduced delay
        else:
            self.window.after(1000, close_dialog)
    
    def _cleanup_temp_files(self, preserve_frames=False):
        """Clean up temporary files and directories
        
        Args:
            preserve_frames: If True, preserve frame files for resume functionality
        """
        import shutil
        
        # === DEBUG: 一時ファイルクリーンアップの詳細ログ ===
        logger.info(f"=== CLEANUP DEBUG: Starting temp files cleanup (preserve_frames={preserve_frames}) ===")
        
        try:
            # Clean up video processor if available
            if self.video_processor:
                logger.info(f"Cleaning up video processor, temp_dir: {self.video_processor.temp_dir}")
                self.video_processor.cleanup(preserve_frames=preserve_frames)
            
            # Clean up temp directory if provided
            if self.temp_dir and os.path.exists(self.temp_dir) and not preserve_frames:
                logger.info(f"=== CLEANUP DEBUG: About to remove temp directory: {self.temp_dir} ===")
                
                # Check if this is a session directory - DO NOT DELETE session directories during cleanup
                if 'upscale_app_sessions' in str(self.temp_dir):
                    logger.warning(f"=== CLEANUP PROTECTION: Refusing to delete session directory ===")
                    logger.warning(f"Directory: {self.temp_dir}")
                    logger.warning(f"Session directories must be preserved for resume functionality")
                    # Skip deletion to preserve session data
                    return
                
                try:
                    # Remove all files in temp directory (only for non-session directories)
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    logger.info(f"=== CLEANUP DEBUG: Removed temp directory: {self.temp_dir} ===")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
            elif preserve_frames:
                logger.info(f"=== CLEANUP DEBUG: Preserving temp directory for resume: {self.temp_dir} ===")
                    
            # Additional cleanup for Windows temp files (SAFE VERSION)
            if os.name == 'nt' and not preserve_frames:
                try:
                    import tempfile
                    temp_root = tempfile.gettempdir()
                    
                    # === FIXED: Do NOT delete directories with frame files during processing ===
                    # This was a major source of frame deletion issues
                    logger.info("DEBUG: Windows temp cleanup - SAFE MODE (preserving session directories)")
                    
                    # Look for upscale related temp directories, but be much more conservative
                    for item in os.listdir(temp_root):
                        item_path = os.path.join(temp_root, item)
                        if os.path.isdir(item_path) and 'upscale' in item.lower():
                            # === CRITICAL FIX: DO NOT delete directories containing frame files ===
                            # Previously this was deleting active session directories
                            try:
                                files = os.listdir(item_path)
                                has_frames = any(f.startswith('frame_') and f.endswith('.png') for f in files)
                                has_session_files = any(f in ['progress.json', 'session.json'] for f in files)
                                
                                if has_frames or has_session_files:
                                    logger.info(f"DEBUG: Preserving active session directory: {item_path}")
                                    continue  # SKIP deletion of active session directories
                                
                                # Only clean up truly empty or old non-session temp directories
                                if len(files) == 0 or all(f.endswith('.tmp') for f in files):
                                    shutil.rmtree(item_path, ignore_errors=True)
                                    logger.info(f"Cleaned up empty temp directory: {item_path}")
                            except Exception as cleanup_e:
                                logger.warning(f"DEBUG: Error checking temp directory {item_path}: {cleanup_e}")
                                pass
                                
                except Exception as e:
                    logger.warning(f"Additional temp cleanup failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Temp file cleanup failed: {e}")
        
    def destroy(self):
        """Destroy progress dialog"""
        try:
            self.window.destroy()
        except:
            pass

class MainGUI:
    """Main GUI application window"""
    
    def __init__(self, video_processor, ai_processor, gpu_info, session_manager=None, log_file_path=None):
        self.video_processor = video_processor
        self.ai_processor = ai_processor  
        self.gpu_info = gpu_info
        self.log_file_path = log_file_path  # ログファイルパスを保存
        
        # Use provided session manager or create new one
        self.session_manager = session_manager or SessionManager()
        self.current_session_id = None
        
        self.root = None
        self.processing_thread = None
        self.is_processing = False
        
        # GUI variables
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.scale_var = tk.StringVar(value="2.0x")
        self.quality_var = tk.StringVar(value="Balanced")
        self.ai_backend_var = tk.StringVar(value="real_cugan")  # Default to Real-CUGAN
        self.thread_setting_var = tk.StringVar(value="2:2:1")  # Default thread setting
        
    def _create_window(self):
        """Create main window"""
        if GUI_AVAILABLE and ctk:
            # Use CustomTkinter
            self.root = ctk.CTk()
            self.root.title("UpScale App - AI Video Upscaling Tool v2.0.0")
            self.root.geometry("1200x1100")
            
            # Configure grid
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
        else:
            # Fallback to standard tkinter
            self.root = tk.Tk()
            self.root.title("UpScale App - AI Video Upscaling Tool v2.0.0")
            self.root.geometry("1000x875")
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')  # Modern looking theme
        
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        self._setup_ui()
    
    def _on_window_close(self):
        """Handle window close event"""
        try:
            # Cancel any ongoing processing
            if hasattr(self, 'processing_cancelled'):
                self.processing_cancelled = True
            
            # Cleanup and close
            self.cleanup()
            
            # Force exit if needed
            import sys
            sys.exit(0)
        except:
            # Force exit as last resort
            import os
            os._exit(0)
        
    def _setup_ui(self):
        """Setup main user interface"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Main container
            self.main_frame = ctk.CTkScrollableFrame(self.root)
            self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            
            # Title section
            self._setup_title_section(self.main_frame)
            
            # File selection section
            self._setup_file_section(self.main_frame)
            
            # Video information section
            self._setup_video_info_section(self.main_frame)
            
            # Settings section  
            self._setup_settings_section(self.main_frame)
            
            # System info section
            self._setup_system_section(self.main_frame)
            
            # Processing section
            self._setup_processing_section(self.main_frame)
            
            # Status section
            self._setup_status_section(self.main_frame)
        else:
            # Standard tkinter fallback
            # Main container with scrolling
            canvas = tk.Canvas(self.root)
            scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Title section
            self._setup_title_section(scrollable_frame)
            
            # File selection section
            self._setup_file_section(scrollable_frame)
            
            # Video information section
            self._setup_video_info_section(scrollable_frame)
            
            # Settings section  
            self._setup_settings_section(scrollable_frame)
            
            # System info section
            self._setup_system_section(scrollable_frame)
            
            # Processing section
            self._setup_processing_section(scrollable_frame)
        
    def _setup_title_section(self, parent):
        """Setup title section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # Title
            self.title_label = ctk.CTkLabel(
                parent, 
                text="🎬 UpScale App", 
                font=ctk.CTkFont(size=28, weight="bold")
            )
            self.title_label.pack(pady=(10, 5))
            
            self.subtitle_label = ctk.CTkLabel(
                parent,
                text="AI-Powered Video Upscaling Tool",
                font=ctk.CTkFont(size=16)
            )
            self.subtitle_label.pack(pady=(0, 20))
        else:
            # Standard tkinter fallback
            title_frame = ttk.Frame(parent)
            title_frame.pack(fill="x", padx=20, pady=20)
            
            title_label = ttk.Label(
                title_frame,
                text="UpScale App",
                font=("Arial", 24, "bold")
            )
            title_label.pack()
            
            subtitle_label = ttk.Label(
                title_frame,
                text="AI-Powered Video Upscaling Tool - Executable Edition",
                font=("Arial", 12)
            )
            subtitle_label.pack(pady=(5, 0))
        
    def _setup_file_section(self, parent):
        """Setup file selection section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            # File section frame
            file_frame = ctk.CTkFrame(parent)
            file_frame.pack(fill="x", padx=10, pady=10)
            
            # Input file
            input_label = ctk.CTkLabel(file_frame, text="📁 Input Video:", font=ctk.CTkFont(weight="bold"))
            input_label.pack(anchor="w", padx=15, pady=(15, 5))
            
            input_container = ctk.CTkFrame(file_frame)
            input_container.pack(fill="x", padx=15, pady=(0, 10))
            
            self.input_entry = ctk.CTkEntry(
                input_container, 
                placeholder_text="Select a video file...",
                height=35,
                textvariable=self.input_path_var
            )
            self.input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
            
            self.browse_input_button = ctk.CTkButton(
                input_container,
                text="Browse",
                command=self._browse_input,
                width=80
            )
            self.browse_input_button.pack(side="right", padx=(5, 10), pady=10)
            
            # Output folder
            output_label = ctk.CTkLabel(file_frame, text="📂 Output Folder:", font=ctk.CTkFont(weight="bold"))
            output_label.pack(anchor="w", padx=15, pady=(10, 5))
            
            output_container = ctk.CTkFrame(file_frame)
            output_container.pack(fill="x", padx=15, pady=(0, 15))
            
            self.output_entry = ctk.CTkEntry(
                output_container,
                placeholder_text="Select output folder...",
                height=35,
                textvariable=self.output_path_var
            )
            self.output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
            
            self.browse_output_button = ctk.CTkButton(
                output_container,
                text="Browse", 
                command=self._browse_output,
                width=80
            )
            self.browse_output_button.pack(side="right", padx=(5, 10), pady=10)
        else:
            # Standard tkinter fallback
            file_frame = ttk.LabelFrame(parent, text="File Selection", padding=15)
            file_frame.pack(fill="x", padx=20, pady=10)
            
            # Input file
            ttk.Label(file_frame, text="Input Video:").grid(row=0, column=0, sticky="w", pady=5)
            
            input_frame = ttk.Frame(file_frame)
            input_frame.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
            input_frame.columnconfigure(0, weight=1)
            
            self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path_var)
            self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            
            ttk.Button(
                input_frame, 
                text="Browse", 
                command=self._browse_input
            ).grid(row=0, column=1)
            
            # Output folder
            ttk.Label(file_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=5)
            
            output_frame = ttk.Frame(file_frame)
            output_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
            output_frame.columnconfigure(0, weight=1)
            
            self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
            self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            
            ttk.Button(
                output_frame,
                text="Browse",
                command=self._browse_output
            ).grid(row=0, column=1)
            
            file_frame.columnconfigure(1, weight=1)
        
    def _setup_video_info_section(self, parent):
        """Setup video information display section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            info_frame = ctk.CTkFrame(parent)
            info_frame.pack(fill="x", padx=10, pady=10)
            
            info_label = ctk.CTkLabel(info_frame, text="📹 動画情報", font=ctk.CTkFont(size=16, weight="bold"))
            info_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Video info container
            self.video_info_container = ctk.CTkFrame(info_frame)
            self.video_info_container.pack(fill="x", padx=15, pady=(0, 15))
            
            # Initially hidden message
            self.video_info_placeholder = ctk.CTkLabel(
                self.video_info_container, 
                text="動画ファイルを選択すると、詳細情報が表示されます。",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            self.video_info_placeholder.pack(pady=10)
            
            # Video info labels (initially hidden)
            self.video_info_labels = {}
            
        else:
            # Standard tkinter fallback
            info_frame = ttk.LabelFrame(parent, text="動画情報", padding=15)
            info_frame.pack(fill="x", padx=20, pady=10)
            
            # Video info container
            self.video_info_container = ttk.Frame(info_frame)
            self.video_info_container.pack(fill="x")
            
            # Initially hidden message
            self.video_info_placeholder = ttk.Label(
                self.video_info_container, 
                text="動画ファイルを選択すると、詳細情報が表示されます。",
                foreground="gray"
            )
            self.video_info_placeholder.pack(pady=10)
            
            # Video info labels (initially hidden)
            self.video_info_labels = {}
        
    def _setup_settings_section(self, parent):
        """Setup settings section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            settings_frame = ctk.CTkFrame(parent)
            settings_frame.pack(fill="x", padx=10, pady=10)
            
            settings_label = ctk.CTkLabel(settings_frame, text="⚙️ Settings", font=ctk.CTkFont(size=16, weight="bold"))
            settings_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Settings grid
            settings_grid = ctk.CTkFrame(settings_frame)
            settings_grid.pack(fill="x", padx=15, pady=(0, 15))
            
            # Scale factor
            ctk.CTkLabel(settings_grid, text="Scale Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
            scale_menu = ctk.CTkComboBox(
                settings_grid,
                values=["2.0x", "4.0x", "8.0x"],
                variable=self.scale_var,
                state="readonly",
                width=100
            )
            scale_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")
            
            # Quality preset
            ctk.CTkLabel(settings_grid, text="Quality:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
            quality_menu = ctk.CTkComboBox(
                settings_grid,
                values=["Fast", "Balanced", "Quality"],
                variable=self.quality_var,
                state="readonly",
                width=120
            )
            quality_menu.grid(row=0, column=3, padx=10, pady=10, sticky="w")
            
            # AI Backend selection (new row)
            ctk.CTkLabel(settings_grid, text="AI Backend:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
            
            # Get available backends for combobox
            available_backends = self.ai_processor.get_available_backends()
            logger.info(f"=== GUI BACKEND INITIALIZATION DEBUG (CTK) ===")
            logger.info(f"Available backends from AI processor: {list(available_backends.keys())}")
            logger.info(f"Backend details: {available_backends}")
            
            backend_options = []
            backend_descriptions = {}
            
            for backend_id, backend_info in available_backends.items():
                display_name = f"{backend_info['name']}"
                if backend_info.get('gpu_support'):
                    display_name += " (GPU)"
                else:
                    display_name += " (CPU)"
                backend_options.append(display_name)
                backend_descriptions[display_name] = backend_id
                logger.info(f"Mapped: '{display_name}' -> '{backend_id}' (name: {backend_info['name']}, gpu: {backend_info.get('gpu_support')})")
            
            self.backend_descriptions = backend_descriptions
            logger.info(f"Final backend_descriptions: {backend_descriptions}")
            logger.info(f"Final backend_options: {backend_options}")
            
            self.ai_backend_menu = ctk.CTkComboBox(
                settings_grid,
                values=backend_options,
                variable=self.ai_backend_var,
                state="readonly",
                width=200,
                command=self._on_backend_change
            )
            self.ai_backend_menu.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="w")
            
            # Thread setting (Real-ESRGAN only)
            ctk.CTkLabel(settings_grid, text="Thread Setting:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
            self.thread_setting_menu = ctk.CTkComboBox(
                settings_grid,
                values=["2:2:1 (Standard)", "4:2:2 (Fast)"],
                variable=self.thread_setting_var,
                state="readonly",
                width=150,
                command=self._on_thread_setting_change
            )
            self.thread_setting_menu.grid(row=2, column=1, padx=10, pady=10, sticky="w")
            
            # Set default backend display
            logger.info(f"Setting default backend display (CTK)...")
            default_set = False
            for display_name, backend_id in self.backend_descriptions.items():
                logger.info(f"Checking backend: '{backend_id}' (display: '{display_name}')")
                if backend_id == 'real_cugan':
                    logger.info(f"Setting default backend to Real-CUGAN: '{display_name}'")
                    self.ai_backend_var.set(display_name)
                    self._update_thread_setting_visibility(backend_id)
                    default_set = True
                    break
            if not default_set:
                logger.warning("Real-CUGAN not found, using first available backend")
                if self.backend_descriptions:
                    first_display = list(self.backend_descriptions.keys())[0]
                    first_backend = self.backend_descriptions[first_display]
                    logger.info(f"Setting first available backend: '{first_backend}' (display: '{first_display}')")
                    self.ai_backend_var.set(first_display)
                    self._update_thread_setting_visibility(first_backend)
        else:
            # Standard tkinter fallback
            settings_frame = ttk.LabelFrame(parent, text="Processing Settings", padding=15)
            settings_frame.pack(fill="x", padx=20, pady=10)
            
            # Scale factor
            ttk.Label(settings_frame, text="Scale Factor:").grid(row=0, column=0, sticky="w", padx=5)
            scale_combo = ttk.Combobox(
                settings_frame,
                textvariable=self.scale_var,
                values=["2.0", "4.0", "8.0"],
                state="readonly",
                width=10
            )
            scale_combo.grid(row=0, column=1, padx=5, sticky="w")
            
            # Quality setting
            ttk.Label(settings_frame, text="Quality:").grid(row=0, column=2, sticky="w", padx=5)
            quality_combo = ttk.Combobox(
                settings_frame,
                textvariable=self.quality_var,
                values=["Fast", "Balanced", "Quality"],
                state="readonly",
                width=12
            )
            quality_combo.grid(row=0, column=3, padx=5, sticky="w")
            
            # AI Backend selection
            ttk.Label(settings_frame, text="AI Backend:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            
            # Get available backends for combobox
            available_backends = self.ai_processor.get_available_backends()
            logger.info(f"=== GUI BACKEND INITIALIZATION DEBUG (TTK) ===")
            logger.info(f"Available backends from AI processor: {list(available_backends.keys())}")
            logger.info(f"Backend details: {available_backends}")
            
            backend_options = []
            backend_descriptions = {}
            
            for backend_id, backend_info in available_backends.items():
                display_name = f"{backend_info['name']}"
                if backend_info.get('gpu_support'):
                    display_name += " (GPU)"
                else:
                    display_name += " (CPU)"
                backend_options.append(display_name)
                backend_descriptions[display_name] = backend_id
                logger.info(f"Mapped: '{display_name}' -> '{backend_id}' (name: {backend_info['name']}, gpu: {backend_info.get('gpu_support')})")
            
            self.backend_descriptions = backend_descriptions
            logger.info(f"Final backend_descriptions: {backend_descriptions}")
            logger.info(f"Final backend_options: {backend_options}")
            
            self.ai_backend_combo = ttk.Combobox(
                settings_frame,
                textvariable=self.ai_backend_var,
                values=backend_options,
                state="readonly",
                width=25
            )
            self.ai_backend_combo.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="w")
            self.ai_backend_combo.bind('<<ComboboxSelected>>', self._on_backend_change)
            
            # Set default backend display
            logger.info(f"Setting default backend display (TTK)...")
            default_set = False
            for display_name, backend_id in self.backend_descriptions.items():
                logger.info(f"Checking backend: '{backend_id}' (display: '{display_name}')")
                if backend_id == 'real_cugan':
                    logger.info(f"Setting default backend to Real-CUGAN: '{display_name}'")
                    self.ai_backend_var.set(display_name)
                    self._update_thread_setting_visibility(backend_id)
                    default_set = True
                    break
            if not default_set:
                logger.warning("Real-CUGAN not found, using first available backend")
                if self.backend_descriptions:
                    first_display = list(self.backend_descriptions.keys())[0]
                    first_backend = self.backend_descriptions[first_display]
                    logger.info(f"Setting first available backend: '{first_backend}' (display: '{first_display}')")
                    self.ai_backend_var.set(first_display)
                    self._update_thread_setting_visibility(first_backend)
    
    def _on_backend_change(self, *args):
        """Handle AI backend selection change"""
        try:
            # === DETAILED DEBUG LOGGING ===
            logger.info(f"=== GUI BACKEND CHANGE DEBUG START ===")
            logger.info(f"Args received: {args}")
            
            # Try to get selection from args first (more reliable for events)
            selected_display = None
            if args and len(args) > 0:
                selected_display = args[0]
                logger.info(f"Selected display from args: '{selected_display}'")
            else:
                selected_display = self.ai_backend_var.get()
                logger.info(f"Selected display from variable: '{selected_display}'")
            logger.info(f"Available backend descriptions: {list(self.backend_descriptions.keys())}")
            logger.info(f"Backend descriptions mapping: {self.backend_descriptions}")
            
            if selected_display in self.backend_descriptions:
                backend_id = self.backend_descriptions[selected_display]
                logger.info(f"Mapped backend ID: '{backend_id}'")
                logger.info(f"Current AI processor backend: '{self.ai_processor.selected_backend}'")
                logger.info(f"User selected AI backend: {backend_id} ({selected_display})")
                
                # Update the variable to ensure consistency
                if self.ai_backend_var.get() != selected_display:
                    logger.info(f"Updating ai_backend_var from '{self.ai_backend_var.get()}' to '{selected_display}'")
                    self.ai_backend_var.set(selected_display)
                
                # Change the backend in AI processor
                logger.info(f"Attempting to switch backend from '{self.ai_processor.selected_backend}' to '{backend_id}'")
                success = self.ai_processor.set_backend(backend_id)
                logger.info(f"Backend switch success: {success}")
                logger.info(f"AI processor backend after switch: '{self.ai_processor.selected_backend}'")
                
                if success:
                    logger.info(f"Successfully switched to backend: {backend_id}")
                    # Update any UI status if needed
                    if hasattr(self, 'status_label'):
                        self.status_label.configure(text=f"AI Backend: {selected_display}")
                    # Update system information display with slight delay to ensure backend switch is complete
                    logger.info(f"=== GUI DEBUG: Attempting to schedule system info update ===")
                    logger.info(f"Has root attribute: {hasattr(self, 'root')}")
                    logger.info(f"Root exists: {getattr(self, 'root', None) is not None}")
                    if hasattr(self, 'root') and self.root:
                        logger.info("Scheduling system info update in 100ms")
                        self.root.after(100, self._update_system_info)
                        # Also try immediate update as backup
                        try:
                            self._update_system_info()
                        except Exception as e:
                            logger.warning(f"Immediate system info update failed: {e}")
                        
                        # Update thread setting visibility
                        self._update_thread_setting_visibility(backend_id)
                    else:
                        logger.warning("Cannot schedule system info update - root not available")
                        # Try immediate update as fallback
                        try:
                            self._update_system_info()
                            self._update_thread_setting_visibility(backend_id)
                        except Exception as e:
                            logger.warning(f"Fallback system info update failed: {e}")
                else:
                    logger.error(f"Failed to switch to backend: {backend_id}")
                    logger.error(f"Available backends in AI processor: {list(self.ai_processor.available_backends.keys())}")
                    
                    # Revert selection
                    logger.info("Reverting backend selection due to failure")
                    for display_name, orig_backend_id in self.backend_descriptions.items():
                        if orig_backend_id == self.ai_processor.selected_backend:
                            logger.info(f"Reverting to display: '{display_name}' (backend: '{orig_backend_id}')")
                            self.ai_backend_var.set(display_name)
                            break
            else:
                logger.error(f"Selected display '{selected_display}' not found in backend descriptions")
                logger.error(f"Available options: {list(self.backend_descriptions.keys())}")
            
            logger.info(f"=== GUI BACKEND CHANGE DEBUG END ===")
            
        except Exception as e:
            logger.error(f"Exception in _on_backend_change: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
    def _on_thread_setting_change(self, *args):
        """Handle thread setting change for Real-ESRGAN"""
        try:
            logger.info(f"=== THREAD SETTING CHANGE DEBUG START ===")
            logger.info(f"Args received: {args}")
            
            selected_display = args[0] if args else self.thread_setting_var.get()
            logger.info(f"Thread setting changed to: {selected_display}")
            logger.info(f"thread_setting_var.get(): {self.thread_setting_var.get()}")
            
            # Extract actual thread setting value
            thread_setting = selected_display.split(' ')[0]  # Extract "2:2:1" from "2:2:1 (Standard)"
            logger.info(f"Extracted thread setting: {thread_setting}")
            
            # Get current backend
            current_backend_display = self.ai_backend_var.get()
            logger.info(f"Current backend display: {current_backend_display}")
            logger.info(f"Backend descriptions: {self.backend_descriptions}")
            
            if current_backend_display in self.backend_descriptions:
                backend_id = self.backend_descriptions[current_backend_display]
                logger.info(f"Backend ID: {backend_id}")
                
                # Apply to Real-ESRGAN and Real-CUGAN backends
                if backend_id in ['real_esrgan', 'real_cugan']:
                    logger.info(f"Backend is {backend_id}, attempting to set thread setting...")
                    backend = self.ai_processor.available_backends.get(backend_id)
                    logger.info(f"Backend object: {backend}")
                    logger.info(f"Has set_thread_setting: {hasattr(backend, 'set_thread_setting') if backend else 'Backend is None'}")
                    
                    if backend and hasattr(backend, 'set_thread_setting'):
                        logger.info(f"Calling set_thread_setting with: {thread_setting}")
                        success = backend.set_thread_setting(thread_setting)
                        logger.info(f"set_thread_setting result: {success}")
                        if success:
                            logger.info(f"SUCCESS: {backend_id} thread setting updated to: {thread_setting}")
                            # Verify the setting was actually changed
                            current_setting = getattr(backend, 'thread_setting', 'UNKNOWN')
                            logger.info(f"Backend thread_setting after change: {current_setting}")
                        else:
                            logger.error(f"ERROR: Failed to set thread setting: {thread_setting}")
                    else:
                        logger.warning(f"ERROR: {backend_id} backend not found or doesn't support thread settings")
                        logger.warning(f"Available backends: {list(self.ai_processor.available_backends.keys())}")
                else:
                    logger.info(f"Thread setting only applies to Real-ESRGAN and Real-CUGAN, current backend is: {backend_id}")
            else:
                logger.error(f"Backend display not found in descriptions: {current_backend_display}")
            
            logger.info(f"=== THREAD SETTING CHANGE DEBUG END ===")
            
        except Exception as e:
            logger.error(f"Exception in _on_thread_setting_change: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_thread_setting_visibility(self, backend_id):
        """Update thread setting control visibility based on selected backend"""
        try:
            if hasattr(self, 'thread_setting_menu'):
                if backend_id in ['real_esrgan', 'real_cugan']:
                    # Show thread setting for Real-ESRGAN and Real-CUGAN
                    self.thread_setting_menu.grid()
                    logger.info(f"Thread setting control shown for {backend_id}")
                else:
                    # Hide thread setting for other backends
                    self.thread_setting_menu.grid_remove()
                    logger.info(f"Thread setting control hidden for backend: {backend_id}")
        except Exception as e:
            logger.error(f"Exception in _update_thread_setting_visibility: {e}")
    
    def _setup_system_section(self, parent):
        """Setup system information section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.system_frame = ctk.CTkFrame(parent)
            self.system_frame.pack(fill="x", padx=10, pady=10)
            
            system_label = ctk.CTkLabel(self.system_frame, text="💻 System Information", font=ctk.CTkFont(size=16, weight="bold"))
            system_label.pack(anchor="w", padx=15, pady=(15, 10))
        else:
            # Standard tkinter fallback  
            self.system_frame = ttk.LabelFrame(parent, text="System Information", padding=15)
            self.system_frame.pack(fill="x", padx=20, pady=10)
        
        # Get AI backend info
        backend_info = self.ai_processor.get_backend_info()
        gpu_summary = self.gpu_info.get('best_backend', 'cpu')
        
        # Count all available GPUs - ensure proper counting
        total_gpus = 0
        gpu_names = []
        all_gpu_details = []
        
        # Log GPU info for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"GPU Info in GUI: {self.gpu_info}")
        
        # Count NVIDIA GPUs
        nvidia_info = self.gpu_info.get('nvidia', {})
        if nvidia_info.get('available', False):
            nvidia_gpus = nvidia_info.get('gpus', [])
            for gpu in nvidia_gpus:
                gpu_name = gpu.get('name', 'Unknown NVIDIA')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"NVIDIA: {gpu_name}")
            logger.info(f"Found {len(nvidia_gpus)} NVIDIA GPUs: {[g.get('name') for g in nvidia_gpus]}")
        
        # Count AMD GPUs
        amd_info = self.gpu_info.get('amd', {})
        if amd_info.get('available', False):
            amd_gpus = amd_info.get('gpus', [])
            for gpu in amd_gpus:
                gpu_name = gpu.get('name', 'Unknown AMD')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"AMD: {gpu_name}")
            logger.info(f"Found {len(amd_gpus)} AMD GPUs: {[g.get('name') for g in amd_gpus]}")
        
        # Count Intel GPUs
        intel_info = self.gpu_info.get('intel', {})
        if intel_info.get('available', False):
            intel_gpus = intel_info.get('gpus', [])
            for gpu in intel_gpus:
                gpu_name = gpu.get('name', 'Unknown Intel')
                gpu_names.append(gpu_name)
                all_gpu_details.append(f"Intel: {gpu_name}")
            logger.info(f"Found {len(intel_gpus)} Intel GPUs: {[g.get('name') for g in intel_gpus]}")
        
        # Count Vulkan devices (include all Vulkan devices that aren't already counted)
        vulkan_info = self.gpu_info.get('vulkan', {})
        if vulkan_info.get('available', False):
            vulkan_devices = vulkan_info.get('devices', [])
            logger.info(f"Found {len(vulkan_devices)} Vulkan devices: {[d.get('name') for d in vulkan_devices]}")
            
            # Add Vulkan devices that aren't duplicates of already found GPUs
            for device in vulkan_devices:
                device_name = device.get('name', 'Vulkan Device')
                
                # Check if this device is not already in our list (avoid duplicates)
                is_duplicate = False
                for existing_name in gpu_names:
                    if device_name.lower() in existing_name.lower() or existing_name.lower() in device_name.lower():
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    gpu_names.append(device_name)
                    all_gpu_details.append(f"Vulkan: {device_name}")
                    logger.info(f"Added Vulkan device: {device_name}")
                else:
                    logger.info(f"Skipped duplicate Vulkan device: {device_name}")
        
        # Set total GPU count
        total_gpus = len(gpu_names)
        logger.info(f"Total GPU count: {total_gpus}")
        logger.info(f"All GPU details: {all_gpu_details}")
        
        # Determine if GPU mode is active - more comprehensive check
        gpu_mode_active = (
            backend_info.get('gpu_mode', False) or 
            gpu_summary != 'cpu' or 
            total_gpus > 0 or
            nvidia_info.get('available', False) or
            amd_info.get('available', False) or
            intel_info.get('available', False) or
            vulkan_info.get('available', False)
        )
        
        logger.info(f"GPU Mode Active: {gpu_mode_active}, Total GPUs: {total_gpus}, Best Backend: {gpu_summary}")
        
        # Format AI processor name with version info
        ai_name = backend_info.get('name', backend_info.get('backend', 'unknown'))
        ai_version = backend_info.get('version', '')
        ai_display = f"{ai_name} ({ai_version})" if ai_version else ai_name
        
        info_text = f"""GPU Backend: {gpu_summary.upper()}
AI Processor: {ai_display}
GPU Mode: {'Yes' if gpu_mode_active else 'No'}
Total GPUs: {total_gpus}"""
        
        # Add GPU details if available
        if gpu_names:
            # Show primary GPU
            primary_gpu = gpu_names[0][:35] + "..." if len(gpu_names[0]) > 35 else gpu_names[0]
            info_text += f"""
Primary GPU: {primary_gpu}"""
            
            # Show secondary GPU if available
            if len(gpu_names) > 1:
                secondary_gpu = gpu_names[1][:35] + "..." if len(gpu_names[1]) > 35 else gpu_names[1]
                info_text += f"""
Secondary GPU: {secondary_gpu}"""
            
            # Show additional GPUs count if more than 2
            if len(gpu_names) > 2:
                additional_count = len(gpu_names) - 2
                info_text += f"""
Additional GPUs: +{additional_count} more"""
        
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.system_info_label = ctk.CTkLabel(self.system_frame, text=info_text, justify="left", anchor="w")
            self.system_info_label.pack(anchor="w", padx=15, pady=(0, 15))
        else:
            # Standard tkinter fallback
            self.system_info_label = ttk.Label(self.system_frame, text=info_text, justify="left")
            self.system_info_label.pack(anchor="w")
        
        # Update system info for default selection on startup
        if hasattr(self, 'root') and self.root:
            self.root.after(100, self._update_system_info)
    
    def _update_system_info(self):
        """Update system information display based on current AI backend"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=== _update_system_info called ===")
        try:
            # Get current AI backend info
            backend_info = self.ai_processor.get_backend_info()
            gpu_summary = self.gpu_info.get('best_backend', 'cpu')
            
            # Log backend info for debugging
            logger.info(f"=== SYSTEM INFO UPDATE ===")
            logger.info(f"Backend info: {backend_info}")
            logger.info(f"Current selected backend: {self.ai_processor.selected_backend}")
            
            # Count all available GPUs - ensure proper counting
            total_gpus = 0
            gpu_names = []
            all_gpu_details = []
            
            # Log GPU info for debugging
            logger.info(f"Updating system info for backend: {backend_info.get('backend', 'unknown')}")
            
            # Count NVIDIA GPUs
            nvidia_info = self.gpu_info.get('nvidia', {})
            if nvidia_info.get('available', False):
                nvidia_gpus = nvidia_info.get('gpus', [])
                for gpu in nvidia_gpus:
                    gpu_name = gpu.get('name', 'Unknown NVIDIA')
                    gpu_names.append(gpu_name)
                    all_gpu_details.append(f"NVIDIA: {gpu_name}")
            
            # Count AMD GPUs
            amd_info = self.gpu_info.get('amd', {})
            if amd_info.get('available', False):
                amd_gpus = amd_info.get('gpus', [])
                for gpu in amd_gpus:
                    gpu_name = gpu.get('name', 'Unknown AMD')
                    gpu_names.append(gpu_name)
                    all_gpu_details.append(f"AMD: {gpu_name}")
            
            # Count Intel GPUs
            intel_info = self.gpu_info.get('intel', {})
            if intel_info.get('available', False):
                intel_gpus = intel_info.get('gpus', [])
                for gpu in intel_gpus:
                    gpu_name = gpu.get('name', 'Unknown Intel')
                    gpu_names.append(gpu_name)
                    all_gpu_details.append(f"Intel: {gpu_name}")
            
            # Count Vulkan devices (include all Vulkan devices that aren't already counted)
            vulkan_info = self.gpu_info.get('vulkan', {})
            if vulkan_info.get('available', False):
                vulkan_devices = vulkan_info.get('devices', [])
                
                # Add Vulkan devices that aren't duplicates of already found GPUs
                for device in vulkan_devices:
                    device_name = device.get('name', 'Vulkan Device')
                    
                    # Check if this device is not already in our list (avoid duplicates)
                    is_duplicate = False
                    for existing_name in gpu_names:
                        if device_name.lower() in existing_name.lower() or existing_name.lower() in device_name.lower():
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        gpu_names.append(device_name)
                        all_gpu_details.append(f"Vulkan: {device_name}")
            
            # Set total GPU count
            total_gpus = len(gpu_names)
            
            # Determine if GPU mode is active - more comprehensive check
            gpu_mode_active = (
                backend_info.get('gpu_mode', False) or 
                gpu_summary != 'cpu' or 
                total_gpus > 0 or
                nvidia_info.get('available', False) or
                amd_info.get('available', False) or
                intel_info.get('available', False) or
                vulkan_info.get('available', False)
            )
            
            # Format AI processor name with version info
            ai_name = backend_info.get('name', backend_info.get('backend', 'unknown'))
            ai_version = backend_info.get('version', '')
            ai_display = f"{ai_name} ({ai_version})" if ai_version else ai_name
            
            info_text = f"""GPU Backend: {gpu_summary.upper()}
AI Processor: {ai_display}
GPU Mode: {'Yes' if gpu_mode_active else 'No'}
Total GPUs: {total_gpus}"""
            
            # Add GPU details if available
            if gpu_names:
                # Show primary GPU
                primary_gpu = gpu_names[0][:35] + "..." if len(gpu_names[0]) > 35 else gpu_names[0]
                info_text += f"""
Primary GPU: {primary_gpu}"""
                
                # Show secondary GPU if available
                if len(gpu_names) > 1:
                    secondary_gpu = gpu_names[1][:35] + "..." if len(gpu_names[1]) > 35 else gpu_names[1]
                    info_text += f"""
Secondary GPU: {secondary_gpu}"""
                
                # Show additional GPUs count if more than 2
                if len(gpu_names) > 2:
                    additional_count = len(gpu_names) - 2
                    info_text += f"""
Additional GPUs: +{additional_count} more"""
            
            # Update the info label text
            logger.info(f"=== LABEL UPDATE DEBUG ===")
            logger.info(f"Has system_info_label: {hasattr(self, 'system_info_label')}")
            logger.info(f"New info_text: {info_text}")
            
            if hasattr(self, 'system_info_label'):
                logger.info("Updating system_info_label with new text")
                if GUI_AVAILABLE and ctk:
                    self.system_info_label.configure(text=info_text)
                    logger.info("Updated CustomTkinter label")
                else:
                    self.system_info_label.configure(text=info_text)
                    logger.info("Updated Tkinter label")
            else:
                logger.warning("system_info_label not found - cannot update display")
                    
        except Exception as e:
            logger.error(f"Error updating system info: {e}")
        
    def _setup_processing_section(self, parent):
        """Setup processing section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            processing_frame = ctk.CTkFrame(parent)
            processing_frame.pack(fill="x", padx=10, pady=10)
            
            processing_label = ctk.CTkLabel(processing_frame, text="🚀 Processing", font=ctk.CTkFont(size=16, weight="bold"))
            processing_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Button container
            button_container = ctk.CTkFrame(processing_frame)
            button_container.pack(fill="x", padx=15, pady=(0, 15))
            
            self.process_button = ctk.CTkButton(
                button_container,
                text="Start Processing",
                command=self._start_processing,
                height=40,
                font=ctk.CTkFont(size=16, weight="bold"),
                state="disabled"
            )
            self.process_button.pack(side="left", padx=10, pady=15, fill="x", expand=True)
        else:
            # Standard tkinter fallback
            process_frame = ttk.LabelFrame(parent, text="Processing", padding=15)
            process_frame.pack(fill="x", padx=20, pady=20)
            
            self.process_button = ttk.Button(
                process_frame,
                text="Start Processing",
                command=self._start_processing,
                state="disabled"
            )
            self.process_button.pack(pady=10)
        
        # Bind input change event
        self.input_path_var.trace("w", self._on_input_change)
        
    def _setup_status_section(self, parent):
        """Setup status section"""
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            status_frame = ctk.CTkFrame(parent)
            status_frame.pack(fill="x", padx=10, pady=10)
            
            status_label = ctk.CTkLabel(status_frame, text="📊 System Status", font=ctk.CTkFont(size=16, weight="bold"))
            status_label.pack(anchor="w", padx=15, pady=(15, 10))
            
            # Status display
            self.status_text = ctk.CTkTextbox(status_frame, height=80)
            self.status_text.pack(fill="x", padx=15, pady=(0, 15))
        else:
            # Standard tkinter fallback
            status_frame = ttk.LabelFrame(parent, text="Status", padding=15)
            status_frame.pack(fill="x", padx=20, pady=10)
            
            # Status display
            self.status_text = scrolledtext.ScrolledText(
                status_frame,
                height=6,
                width=70,
                font=("Consolas", 9)
            )
            self.status_text.pack(fill="both", expand=True, pady=(10, 0))
        
        self._add_status_message("Ready - Select input video to begin")
        
    def _add_status_message(self, message: str):
        """Add message to status display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        if GUI_AVAILABLE and ctk:
            # CustomTkinter implementation
            self.status_text.insert("end", formatted)
            self.status_text.see("end")
        else:
            # Standard tkinter fallback
            self.status_text.insert(tk.END, formatted)
            self.status_text.see(tk.END)
        
    def _browse_input(self):
        """Browse for input video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Log for debugging
            logger.info(f"Setting input file path: {file_path}")
            
            # Set the variable
            self.input_path_var.set(file_path)
            
            # Force multiple update methods
            self.input_entry.update()
            self.input_entry.update_idletasks()
            self.root.update()
            self.root.update_idletasks()
            
            # Direct manipulation of entry widget
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)
            # Note: self.input_entry.see(0) not available in tkinter Entry
            
            # Display video information
            self._display_video_info(file_path)
            
            # Check for resumable sessions
            self._check_resumable_sessions(file_path)
            
            # Force focus and refresh
            self.input_entry.focus_set()
            self.input_entry.focus_force()
            self.root.after(100, lambda: self.input_entry.focus_set())
            
            # Add status message
            self._add_status_message(f"Selected: {Path(file_path).name}")
            logger.info(f"Input entry content: '{self.input_entry.get()}'")
            
    def _browse_output(self):
        """Browse for output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        
        if folder_path:
            # Log for debugging
            logger.info(f"Setting output folder path: {folder_path}")
            
            # Set the variable
            self.output_path_var.set(folder_path)
            
            # Force multiple update methods
            self.output_entry.update()
            self.output_entry.update_idletasks()
            self.root.update()
            self.root.update_idletasks()
            
            # Direct manipulation of entry widget
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)
            # Note: self.output_entry.see(0) not available in tkinter Entry
            
            # Force focus and refresh
            self.output_entry.focus_set()
            self.output_entry.focus_force()
            self.root.after(100, lambda: self.output_entry.focus_set())
            
            # Add status message
            self._add_status_message(f"Output folder: {Path(folder_path).name}")
            logger.info(f"Output entry content: '{self.output_entry.get()}'")
            
    def _on_input_change(self, *args):
        """Handle input file change"""
        input_path = self.input_path_var.get().strip()
        
        if input_path and Path(input_path).exists():
            if GUI_AVAILABLE and ctk:
                self.process_button.configure(state="normal")
            else:
                self.process_button.config(state="normal")
            self._add_status_message(f"Video selected: {Path(input_path).name}")
        else:
            if GUI_AVAILABLE and ctk:
                self.process_button.configure(state="disabled")
            else:
                self.process_button.config(state="disabled")
            
    def _start_processing(self):
        """Start video processing"""
        logger.info(f"DEBUG: _start_processing called - is_processing: {self.is_processing}")
        if self.is_processing:
            logger.warning("DEBUG: Processing already in progress, returning early")
            return
            
        input_path = self.input_path_var.get().strip()
        output_folder = self.output_path_var.get().strip()
        
        if not input_path:
            if GUI_AVAILABLE and ctk:
                CTkMessagebox(
                    title="Error",
                    message="Please select an input video file",
                    icon="cancel"
                )
            else:
                messagebox.showerror("Error", "Please select an input video file")
            return
            
        if not output_folder:
            output_folder = str(Path(input_path).parent / "output")
            self.output_path_var.set(output_folder)
            
        # Create output directory
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{Path(input_path).stem}_upscaled.mp4"
        output_path = str(Path(output_folder) / output_filename)
        
        # Get video info for session creation
        logger.info(f"DEBUG: Starting video validation for: {input_path}")
        video_info = self.video_processor.validate_video(input_path)
        logger.info(f"DEBUG: Video validation result: valid={video_info['valid']}, error={video_info.get('error', 'None')}")
        
        if not video_info['valid']:
            logger.error(f"DEBUG: Video validation failed: {video_info['error']}")
            if GUI_AVAILABLE and ctk:
                CTkMessagebox(
                    title="Error",
                    message=f"Invalid video file: {video_info['error']}",
                    icon="cancel"
                )
            else:
                messagebox.showerror("Error", f"Invalid video file: {video_info['error']}")
            return
        
        # Initialize or resume session
        settings = self._get_current_settings()
        if not self.current_session_id:
            # Create new session
            self.current_session_id = self.session_manager.create_session(
                input_path, settings, video_info['info']
            )
            logger.info(f"Created new session: {self.current_session_id}")
        else:
            logger.info(f"Resuming session: {self.current_session_id}")
            
            # === RESUME FIX: セッション再開時の状態検証 ===
            progress_data = self.session_manager.load_progress(self.current_session_id)
            if not progress_data:
                logger.warning(f"Resume session {self.current_session_id} has no progress data, recreating...")
                # プログレスデータがない場合は新規セッションとして扱う
                self.current_session_id = self.session_manager.create_session(
                    input_path, settings, video_info['info']
                )
                logger.info(f"Recreated session: {self.current_session_id}")
            else:
                # セッション情報のログ出力を強化
                completed_frames = progress_data.get('steps', {}).get('upscale', {}).get('completed_frames', [])
                logger.info(f"Resume session verification:")
                logger.info(f"  - Session ID: {self.current_session_id}")
                logger.info(f"  - Completed frames in session: {len(completed_frames)}")
                logger.info(f"  - Extract status: {progress_data.get('steps', {}).get('extract', {}).get('status', 'unknown')}")
                logger.info(f"  - Upscale status: {progress_data.get('steps', {}).get('upscale', {}).get('status', 'unknown')}")
                
                # === RESUME FIX: 実際のファイルと完了フレーム情報を同期 ===
                if len(completed_frames) == 0:
                    logger.info("=== RESUME SYNC: No completed frames in session, checking filesystem ===")
                    session_dir = self.session_manager.get_session_dir(self.current_session_id)
                    upscaled_dir = session_dir / "upscaled"
                    
                    if upscaled_dir.exists():
                        # アップスケール済みフレームファイルを検索
                        upscaled_files = sorted([str(p) for p in upscaled_dir.glob("frame_*_upscaled.png")])
                        if upscaled_files:
                            logger.info(f"Found {len(upscaled_files)} upscaled frames in filesystem, syncing to session...")
                            
                            # セッションデータに実際のファイルを反映
                            for frame_file in upscaled_files:
                                self.session_manager.add_completed_frame(self.current_session_id, frame_file)
                            
                            # 更新後の状態を再ロード
                            progress_data = self.session_manager.load_progress(self.current_session_id)
                            completed_frames = progress_data.get('steps', {}).get('upscale', {}).get('completed_frames', [])
                            logger.info(f"After sync: {len(completed_frames)} completed frames in session")
                        else:
                            logger.info("No upscaled frames found in filesystem")
                    else:
                        logger.info("No upscaled directory found")
        
        # Update session with output path
        progress_data = self.session_manager.load_progress(self.current_session_id)
        if progress_data:
            progress_data['steps']['combine']['output_path'] = output_path
            self.session_manager.save_progress(self.current_session_id, progress_data)
        
        # Get settings
        try:
            scale_factor = float(self.scale_var.get())
        except ValueError:
            scale_factor = 2.0
        
        # Start processing thread first
        self.is_processing = True
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Processing...", state="disabled")
        else:
            self.process_button.config(text="Processing...", state="disabled")
        
        def process_video():
            """Background video processing"""
            timer = SimpleTimer()
            progress_dialog = None
            
            try:
                # Use session-specific temp directory
                session_dir = self.session_manager.get_session_dir(self.current_session_id)
                temp_dir = str(session_dir)
                
                # Update video processor temp directory - ONLY set frame directory, not session directory
                # This prevents session directory (with progress.json) from being deleted during cleanup
                session_dir = Path(temp_dir)
                frame_temp_dir = session_dir / "frames" 
                frame_temp_dir.mkdir(exist_ok=True)
                
                # Set video processor to use ONLY the frames subdirectory as temp_dir
                self.video_processor.temp_dir = frame_temp_dir
                self.video_processor.frame_dir = frame_temp_dir
                
                logger.info(f"=== SESSION PROTECTION: Video processor temp_dir set to frames subdir: {frame_temp_dir} ===")
                logger.info(f"Session directory preserved: {session_dir}")
                
                # Show progress dialog immediately on GUI thread
                def create_progress_dialog():
                    nonlocal progress_dialog
                    logger.info("DEBUG: Creating progress dialog...")
                    progress_dialog = ProgressDialog(self.root, temp_dir=str(frame_temp_dir))
                    progress_dialog.video_processor = self.video_processor  # Pass reference for cleanup
                    progress_dialog.main_gui = self  # Pass main GUI reference for button reset
                    progress_dialog.add_log_message("Initializing video processing...")
                    progress_dialog.update_progress(0, "Starting processing...")
                    logger.info("DEBUG: Progress dialog created and initialized")
                
                logger.info("DEBUG: Scheduling progress dialog creation...")
                self.root.after(0, create_progress_dialog)
                
                # Wait for progress dialog to be created
                import time
                while progress_dialog is None:
                    time.sleep(0.01)
                
                def progress_callback(progress, message):
                    if progress_dialog and not progress_dialog.cancelled:
                        self.root.after(0, lambda p=progress, m=message: progress_dialog.update_progress(p, m))
                        self.root.after(0, lambda m=message: progress_dialog.add_log_message(m))
                    
                    # Check for cancellation
                    if progress_dialog and progress_dialog.cancelled:
                        raise KeyboardInterrupt("Processing cancelled by user")
                
                # Step 1: Validate video (check if we can skip due to resume)
                progress_data = self.session_manager.load_progress(self.current_session_id)
                validate_status = progress_data['steps']['validate']['status'] if progress_data else 'pending'
                
                if validate_status != 'completed':
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 0, "start"))
                    self.session_manager.update_step_status(self.current_session_id, "validate", "in_progress", 0)
                    
                    progress_callback(5, "動画ファイル検証中...")
                    validation = self.video_processor.validate_video(input_path)
                    
                    if not validation['valid']:
                        self.session_manager.update_step_status(self.current_session_id, "validate", "failed", 0, 
                                                              {"error": validation['error']})
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 0, "error"))
                        raise RuntimeError(f"Invalid video: {validation['error']}")
                    
                    video_info = validation['info']
                    self.session_manager.update_step_status(self.current_session_id, "validate", "completed", 100)
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("validate", 100, "complete"))
                else:
                    # Skip validation - already completed
                    progress_callback(5, "動画検証完了済み（スキップ）")
                    video_info = progress_data.get('video_info', {})
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"動画情報: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}秒"))
                
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"DEBUG: Video validation complete. Frames: {video_info.get('frame_count', 'unknown')}"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message("DEBUG: About to start frame extraction step"))
                
                # Step 2: Extract frames (check if we can skip or resume)
                extract_status = progress_data['steps']['extract']['status'] if progress_data else 'pending'
                
                # === DEBUG: Extract status check ===
                logger.info(f"=== EXTRACT STATUS DEBUG ===")
                logger.info(f"Progress data exists: {progress_data is not None}")
                logger.info(f"Extract status: {extract_status}")
                if progress_data:
                    logger.info(f"Session ID: {progress_data.get('session_id', 'unknown')}")
                    logger.info(f"All steps: {progress_data.get('steps', {}).keys()}")
                    extract_data = progress_data.get('steps', {}).get('extract', {})
                    logger.info(f"Extract step data: {extract_data}")
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"DEBUG: Extract status = {extract_status}"))
                
                if extract_status == 'completed':
                    # Check if frames actually exist before skipping extraction
                    progress_callback(15, "フレーム抽出完了状態を確認中...")
                    
                    # === DEBUG: レジューム時のフレーム読み込み ===
                    logger.info(f"=== RESUME DEBUG: Loading existing frames ===")
                    logger.info(f"Video processor frame_dir: {self.video_processor.frame_dir}")
                    logger.info(f"Session directory: {session_dir}")
                    
                    # Load existing frame paths from session frames directory
                    session_frames_dir = session_dir / "frames"
                    frame_paths = []
                    
                    if session_frames_dir.exists():
                        frame_paths = sorted([str(p) for p in session_frames_dir.glob("frame_*.png")])
                        logger.info(f"Found {len(frame_paths)} frames in session frames directory")
                    else:
                        # Fallback to video processor frame_dir
                        frame_paths = sorted([str(p) for p in self.video_processor.frame_dir.glob("frame_*.png")])
                        logger.info(f"Found {len(frame_paths)} frames in video processor frame_dir (fallback)")
                    
                    # === EXTRACTION INTEGRITY CHECK ===
                    logger.info(f"DEBUG: video_info structure: {video_info}")
                    
                    # 安全に動画情報を取得
                    if isinstance(video_info, dict) and 'info' in video_info:
                        expected_frames = video_info['info']['frame_count']
                    else:
                        logger.warning(f"video_info does not have expected 'info' structure: {video_info}")
                        # フォールバック: 直接frame_countを探す
                        if isinstance(video_info, dict) and 'frame_count' in video_info:
                            expected_frames = video_info['frame_count']
                        else:
                            logger.error(f"Cannot determine frame count from video_info: {video_info}")
                            expected_frames = len(frame_paths)  # 安全なフォールバック
                    
                    actual_frames = len(frame_paths)
                    
                    logger.info(f"=== EXTRACTION INTEGRITY CHECK ===")
                    logger.info(f"Expected frames: {expected_frames}")
                    logger.info(f"Actual frames: {actual_frames}")
                    
                    # フレーム数の差が大きい場合は警告ログを出力
                    if actual_frames < expected_frames and (expected_frames - actual_frames) > 10:
                        missing_frames = expected_frames - actual_frames
                        completion_percentage = (actual_frames / expected_frames) * 100
                        
                        logger.warning(f"MAJOR FRAME COUNT MISMATCH DETECTED:")
                        logger.warning(f"  Expected: {expected_frames} frames")
                        logger.warning(f"  Actual: {actual_frames} frames")
                        logger.warning(f"  Missing: {missing_frames} frames")
                        logger.warning(f"  Completion: {completion_percentage:.1f}%")
                        logger.warning(f"  This indicates incomplete frame extraction")
                    
                    # 継続抽出機能を有効化
                    if actual_frames < expected_frames and (expected_frames - actual_frames) > 10:
                        missing_frames = expected_frames - actual_frames
                        completion_percentage = (actual_frames / expected_frames) * 100
                        
                        logger.warning(f"INCOMPLETE EXTRACTION DETECTED:")
                        logger.warning(f"  Missing frames: {missing_frames}")
                        logger.warning(f"  Extraction completion: {completion_percentage:.1f}%")
                        logger.warning(f"  Extract status was 'completed' but only {actual_frames}/{expected_frames} frames exist")
                        
                        # セッションの抽出ステータスを修正
                        progress_data = self.session_manager.load_progress(self.current_session_id)
                        if progress_data:
                            extract_step = progress_data['steps']['extract']
                            extract_step['status'] = 'incomplete'
                            extract_step['actual_frames'] = actual_frames
                            extract_step['expected_frames'] = expected_frames
                            extract_step['completion_percentage'] = completion_percentage
                            self.session_manager.save_progress(self.current_session_id, progress_data)
                            logger.info("Updated extract status to 'incomplete' in session data")
                        
                        # より簡単な解決策：フレーム抽出をやり直し
                        logger.info(f"=== RESTARTING FRAME EXTRACTION ===")
                        logger.info(f"Incomplete extraction detected - restarting full extraction")
                        
                        try:
                            # セッションのフレーム抽出ステータスを未完了に変更
                            progress_data = self.session_manager.load_progress(self.current_session_id)
                            if progress_data:
                                extract_step = progress_data['steps']['extract']
                                extract_step['status'] = 'pending'
                                extract_step['progress'] = 0
                                extract_step['error'] = 'Incomplete extraction detected, restarting'
                                self.session_manager.save_progress(self.current_session_id, progress_data)
                                logger.info("Reset extraction status to pending")
                            
                            # 既存のフレームをクリーンアップしてフレーム抽出をやり直し
                            import shutil
                            if session_frames_dir.exists():
                                logger.info(f"Cleaning up incomplete frames directory: {session_frames_dir}")
                                shutil.rmtree(session_frames_dir)
                                session_frames_dir.mkdir(parents=True, exist_ok=True)
                            
                            # フレーム抽出を再実行
                            logger.info("Restarting complete frame extraction...")
                            progress_callback(15, "不完全な抽出を検出 - フレーム抽出を再開中...")
                            
                            frame_paths = self.video_processor.extract_frames(
                                input_path,
                                str(session_frames_dir),
                                lambda p, m: progress_callback(15 + p * 0.35, f"フレーム抽出: {m}")
                            )
                            
                            if frame_paths:
                                logger.info(f"Frame extraction restart successful: {len(frame_paths)} frames extracted")
                                progress_callback(50, f"フレーム抽出完了: {len(frame_paths)}フレーム")
                                
                                # 抽出完了をセッションに記録
                                if progress_data:
                                    extract_step = progress_data['steps']['extract']
                                    extract_step['status'] = 'completed'
                                    extract_step['progress'] = 100
                                    extract_step['extracted_frames'] = len(frame_paths)
                                    extract_step['restarted'] = True
                                    self.session_manager.save_progress(self.current_session_id, progress_data)
                            else:
                                raise RuntimeError("Frame extraction restart failed")
                                
                        except Exception as e:
                            logger.error(f"Error in frame extraction restart: {e}")
                            logger.info("Proceeding with existing frames despite restart error")
                            # エラーが発生しても処理を継続
                    else:
                        logger.info(f"Extraction integrity: PASSED (all {expected_frames} frames present)")
                    
                    # === CRITICAL FIX: Check if frames actually exist ===
                    if len(frame_paths) == 0:
                        # Check if this is a true resume case or initial processing
                        progress_data = self.session_manager.load_progress(self.current_session_id)
                        is_true_resume = progress_data and any(
                            step.get('status') in ['completed', 'in_progress'] 
                            for step in progress_data.get('steps', {}).values()
                        )
                        
                        # === ENHANCED DEBUG: More detailed directory inspection ===
                        logger.warning("=== FRAME DETECTION FAILURE DEBUG ===")
                        logger.warning(f"Session frames directory: {session_frames_dir}")
                        logger.warning(f"Session frames directory exists: {session_frames_dir.exists()}")
                        if session_frames_dir.exists():
                            all_files = list(session_frames_dir.glob("*"))
                            logger.warning(f"All files in session frames dir: {[str(f) for f in all_files]}")
                            png_files = list(session_frames_dir.glob("*.png"))
                            logger.warning(f"PNG files in session frames dir: {[str(f) for f in png_files]}")
                        logger.warning(f"Video processor frame_dir: {self.video_processor.frame_dir}")
                        logger.warning(f"Video processor frame_dir exists: {self.video_processor.frame_dir.exists()}")
                        if self.video_processor.frame_dir.exists():
                            vp_files = list(self.video_processor.frame_dir.glob("*"))
                            logger.warning(f"All files in video processor frame_dir: {[str(f) for f in vp_files]}")
                        
                        if is_true_resume:
                            logger.warning("=== RESUME ISSUE: Extract marked as completed but no frame files found! ===")
                            logger.warning("Forcing frame re-extraction...")
                            self._safe_gui_update(lambda: progress_dialog.add_log_message("⚠️ 前回の抽出フレームが見つからないため、フレーム抽出を実行します"))
                        else:
                            logger.info("=== INITIAL PROCESSING: Starting frame extraction ===")
                            self._safe_gui_update(lambda: progress_dialog.add_log_message("フレーム抽出を開始します"))
                        
                        # Reset extract status and force re-extraction
                        self.session_manager.update_step_status(self.current_session_id, "extract", "pending", 0)
                        
                        # Force extraction by falling through to extraction logic
                        extract_status = 'pending'
                    else:
                        # Frames exist, can skip extraction
                        progress_callback(15, "フレーム抽出完了済み（スキップ）")
                        self._safe_gui_update(lambda: progress_dialog.add_log_message(f"Loaded {len(frame_paths)} existing frames"))
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 100, "complete"))
                
                if extract_status != 'completed':
                    # Need to extract frames
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 0, "start"))
                    self.session_manager.update_step_status(self.current_session_id, "extract", "in_progress", 0)
                    progress_callback(10, "フレーム抽出中...")
                    
                    self._safe_gui_update(lambda: progress_dialog.add_log_message("DEBUG: Frame extraction step initialized"))
                    
                    def extract_progress_callback(progress, message):
                        # Map extraction progress to step progress (0-100)
                        step_progress = progress
                        self.session_manager.update_step_status(self.current_session_id, "extract", "in_progress", step_progress)
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", step_progress))
                        progress_callback(5 + (progress * 0.1), message)  # 5-15% overall
                    
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"DEBUG: About to call extract_frames with: {input_path}"))
                    
                    # Add a small delay to ensure GUI updates are processed
                    import time
                    time.sleep(0.1)
                    
                    frame_paths = self.video_processor.extract_frames(input_path, extract_progress_callback, progress_dialog)
                    
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"DEBUG: extract_frames returned: {len(frame_paths) if frame_paths else 0} frames"))
                    
                    if not frame_paths:
                        self.session_manager.update_step_status(self.current_session_id, "extract", "failed", 0, 
                                                              {"error": "Failed to extract frames"})
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 0, "error"))
                        raise RuntimeError("Failed to extract frames")
                    
                    # === CRITICAL FIX: Copy frames to session directory for resume functionality ===
                    session_dir = self.session_manager.get_session_dir(self.current_session_id)
                    session_frames_dir = session_dir / "frames"
                    session_frames_dir.mkdir(parents=True, exist_ok=True)
                    
                    progress_callback(15.5, "フレームをセッションディレクトリに保存中...")
                    logger.info(f"=== RESUME FIX: Copying frames to session directory ===")
                    logger.info(f"Source frames: {len(frame_paths)} files")
                    logger.info(f"Destination: {session_frames_dir}")
                    
                    import shutil
                    copied_frames = []
                    for i, frame_path in enumerate(frame_paths):
                        source_frame = Path(frame_path)
                        dest_frame = session_frames_dir / source_frame.name
                        
                        try:
                            if not dest_frame.exists():
                                shutil.copy2(source_frame, dest_frame)
                                copied_frames.append(str(dest_frame))
                            else:
                                copied_frames.append(str(dest_frame))
                                
                            if i % 1000 == 0:  # Progress update every 1000 frames
                                progress_percent = 15.5 + (i / len(frame_paths)) * 0.5
                                progress_callback(progress_percent, f"フレーム保存中... ({i+1}/{len(frame_paths)})")
                        except Exception as e:
                            logger.error(f"Failed to copy frame {source_frame} to session directory: {e}")
                            # Use original path as fallback
                            copied_frames.append(frame_path)
                    
                    # Update frame_paths to point to session directory
                    frame_paths = copied_frames
                    logger.info(f"Successfully copied {len(copied_frames)} frames to session directory")
                    
                    # Update session with extracted frame count
                    self.session_manager.update_step_status(self.current_session_id, "extract", "completed", 100,
                                                          {"extracted_frames": len(frame_paths)})
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("extract", 100, "complete"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"フレーム抽出完了: {len(frame_paths)}枚"))
                
                # Memory cleanup after frame extraction
                progress_callback(16, "メモリクリーンアップ中...")
                self.video_processor.cleanup_memory_between_operations()
                
                # Step 3: Upscale frames (with resume support)
                upscale_status = progress_data['steps']['upscale']['status'] if progress_data else 'pending'
                
                if upscale_status == 'completed':
                    # Skip upscaling - already completed
                    progress_callback(85, "AIアップスケーリング完了済み（スキップ）")
                    # Load existing upscaled frames
                    session_dir = self.session_manager.get_session_dir(self.current_session_id)
                    upscaled_dir = session_dir / "upscaled"
                    upscaled_frames = sorted([str(p) for p in upscaled_dir.glob("*_upscaled.png")])
                    
                    # === ENHANCED DEBUG: Check upscaled frames directory ===
                    if len(upscaled_frames) == 0:
                        logger.warning("=== UPSCALED FRAME DETECTION FAILURE DEBUG ===")
                        logger.warning(f"Upscaled directory: {upscaled_dir}")
                        logger.warning(f"Upscaled directory exists: {upscaled_dir.exists()}")
                        if upscaled_dir.exists():
                            all_files = list(upscaled_dir.glob("*"))
                            logger.warning(f"All files in upscaled dir: {[str(f) for f in all_files]}")
                            png_files = list(upscaled_dir.glob("*.png"))
                            logger.warning(f"PNG files in upscaled dir: {[str(f) for f in png_files]}")
                        logger.warning("Upscale marked as completed but no upscaled files found! Forcing re-upscale...")
                        self._safe_gui_update(lambda: progress_dialog.add_log_message("⚠️ 前回のアップスケール結果が見つからないため、AI処理を再実行します"))
                        # Reset upscale status and force re-processing
                        self.session_manager.update_step_status(self.current_session_id, "upscale", "pending", 0)
                        upscale_status = 'pending'
                    else:
                        self._safe_gui_update(lambda: progress_dialog.add_log_message(f"Loaded {len(upscaled_frames)} existing upscaled frames"))
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 100, "complete"))
                else:
                    # Need to upscale frames (with resume)
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 0, "start"))
                    self.session_manager.update_step_status(self.current_session_id, "upscale", "in_progress", 0)
                    
                    # Check for existing upscaled frames (resume case)
                    completed_frames = self.session_manager.get_completed_frames(self.current_session_id)
                    remaining_frames = self.session_manager.get_remaining_frames(self.current_session_id, frame_paths)
                    
                    # === DEBUG: Resume status ===
                    logger.info(f"=== AI UPSCALE RESUME DEBUG ===")
                    logger.info(f"Total frames to process: {len(frame_paths)}")
                    logger.info(f"Completed frames: {len(completed_frames)}")
                    logger.info(f"Remaining frames: {len(remaining_frames)}")
                    
                    # Additional debug for path comparison
                    if completed_frames:
                        logger.info(f"Sample completed frame: {completed_frames[0]}")
                    if frame_paths:
                        logger.info(f"Sample frame path: {frame_paths[0]}")
                    
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"DEBUG: 完了済み {len(completed_frames)}フレーム、残り {len(remaining_frames)}フレーム"))
                    
                    if completed_frames:
                        progress_callback(20, f"AIアップスケーリング再開中... ({len(completed_frames)}/{len(frame_paths)} 完了済み)")
                        self._safe_gui_update(lambda: progress_dialog.add_log_message(f"Resume: {len(completed_frames)} frames already completed, {len(remaining_frames)} remaining"))
                        
                        # Calculate resume progress
                        initial_progress = (len(completed_frames) / len(frame_paths)) * 100
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", initial_progress))
                    else:
                        progress_callback(20, "AIアップスケーリング実行中...")
                        remaining_frames = frame_paths
                    
                    def upscale_progress_callback(progress, message):
                        # Calculate overall progress including completed frames
                        completed_count = len(completed_frames)
                        total_count = len(frame_paths)
                        remaining_count = len(remaining_frames)
                        
                        # Progress for remaining frames
                        remaining_progress = (progress / 100) * remaining_count
                        overall_progress = ((completed_count + remaining_progress) / total_count) * 100
                        
                        self.session_manager.update_step_status(self.current_session_id, "upscale", "in_progress", overall_progress)
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", overall_progress))
                        progress_callback(15 + (overall_progress * 0.7), message)  # 15-85% overall
                    
                    # Process remaining frames only
                    if remaining_frames:
                        # Continue processing remaining frames
                        self._safe_gui_update(lambda: progress_dialog.add_log_message(f"処理継続: 残り{len(remaining_frames)}フレーム"))
                        upscaled_frames = self._upscale_frames_with_resume(
                            remaining_frames, completed_frames, scale_factor, 
                            upscale_progress_callback, progress_dialog
                        )
                    else:
                        # All frames already completed - get all upscaled frames
                        session_dir = self.session_manager.get_session_dir(self.current_session_id)
                        upscaled_dir = session_dir / "upscaled"
                        upscaled_frames = sorted([str(p) for p in upscaled_dir.glob("*_upscaled.png")])
                        self._safe_gui_update(lambda: progress_dialog.add_log_message(f"全フレーム処理済み: {len(upscaled_frames)}フレーム"))
                    
                    if not upscaled_frames:
                        self.session_manager.update_step_status(self.current_session_id, "upscale", "failed", 0,
                                                              {"error": "Failed to upscale frames"})
                        self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 0, "error"))
                        raise RuntimeError("Failed to upscale frames")
                    
                    self.session_manager.update_step_status(self.current_session_id, "upscale", "completed", 100,
                                                          {"completed_frames": upscaled_frames})
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("upscale", 100, "complete"))
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"AIアップスケーリング完了: {len(upscaled_frames)}枚"))
                
                # Memory cleanup after AI upscaling
                progress_callback(87, "メモリクリーンアップ中...")
                self.ai_processor._cleanup_memory_between_batches()
                self.video_processor.cleanup_memory_between_operations()
                
                # Step 4: Combine to video
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 0, "start"))
                progress_callback(90, "動画結合中...")
                
                def combine_progress_callback(progress, message):
                    # Map combine progress to step progress (0-100)
                    step_progress = progress
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", step_progress))
                    progress_callback(85 + (progress * 0.15), message)  # 85-100% overall
                
                combine_result = self.video_processor.combine_frames_to_video(
                    upscaled_frames,
                    output_path,
                    input_path,
                    video_info.get('frame_rate', 30.0),
                    combine_progress_callback
                )
                
                # 戻り値がbool型（後方互換性）またはdict型（新しい形式）を処理
                if isinstance(combine_result, bool):
                    success = combine_result
                    is_partial = False
                else:
                    success = combine_result.get('success', False)
                    is_partial = combine_result.get('is_partial', False)
                
                if not success:
                    error_msg = combine_result.get('error', 'Failed to combine frames to video') if isinstance(combine_result, dict) else 'Failed to combine frames to video'
                    self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 0, "error"))
                    raise RuntimeError(error_msg)
                
                # 部分処理の警告を表示
                if is_partial:
                    completion_percentage = combine_result.get('completion_percentage', 0)
                    missing_frames = combine_result.get('missing_frames', 0)
                    original_frames = combine_result.get('original_frames', 0)
                    
                    warning_msg = (
                        f"⚠️ 部分処理が検出されました\n\n"
                        f"処理完了率: {completion_percentage:.1f}%\n"
                        f"処理済みフレーム: {combine_result.get('processed_frames', 0)}/{original_frames}\n"
                        f"未処理フレーム: {missing_frames}\n\n"
                        f"現在の状況:\n"
                        f"• 部分的な動画が作成されました\n"
                        f"• 作成された動画は元の動画よりも短くなります\n\n"
                        f"残りのフレームを処理して完全な動画を作成しますか？"
                    )
                    
                    def show_partial_warning():
                        continue_processing = False
                        
                        if self.use_modern_gui:
                            import customtkinter as ctk
                            dialog = ctk.CTkToplevel(self.window)
                            dialog.title("部分処理の確認")
                            dialog.geometry("500x400")
                            dialog.transient(self.window)
                            dialog.grab_set()
                            
                            # センタリング
                            dialog.update_idletasks()
                            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
                            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
                            dialog.geometry(f"+{x}+{y}")
                            
                            main_frame = ctk.CTkFrame(dialog)
                            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                            
                            text_widget = ctk.CTkTextbox(main_frame, wrap="word", height=250)
                            text_widget.pack(fill="both", expand=True, pady=(0, 15))
                            text_widget.insert("1.0", warning_msg)
                            text_widget.configure(state="disabled")
                            
                            button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
                            button_frame.pack(fill="x")
                            
                            def on_continue():
                                nonlocal continue_processing
                                continue_processing = True
                                dialog.destroy()
                            
                            def on_cancel():
                                dialog.destroy()
                            
                            continue_button = ctk.CTkButton(
                                button_frame, 
                                text="継続処理する", 
                                command=on_continue,
                                fg_color="#2B8B3D",  # Green color
                                hover_color="#1F6B2B"
                            )
                            continue_button.pack(side="left", padx=(0, 10))
                            
                            cancel_button = ctk.CTkButton(
                                button_frame, 
                                text="現在の結果で完了", 
                                command=on_cancel,
                                fg_color="#D32F2F",  # Red color
                                hover_color="#B71C1C"
                            )
                            cancel_button.pack(side="right")
                            
                            # ダイアログが閉じられるまで待機
                            dialog.wait_window()
                        else:
                            import tkinter.messagebox as messagebox
                            continue_processing = messagebox.askyesno(
                                "部分処理の確認", 
                                warning_msg + "\n\n継続処理しますか？",
                                icon='warning'
                            )
                        
                        # 継続処理が選択された場合
                        if continue_processing:
                            self._continue_partial_processing(
                                session_id, 
                                input_path, 
                                output_path, 
                                combine_result.get('processed_frames', 0),
                                original_frames
                            )
                    
                    self._safe_gui_update(show_partial_warning)
                
                # Completed
                self._safe_gui_update(lambda: progress_dialog.update_step_progress("combine", 100, "complete"))
                progress_callback(100, "処理完了!")
                processing_time = timer.elapsed_str()
                
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"処理完了: {processing_time}"))
                self._safe_gui_update(lambda: progress_dialog.add_log_message(f"出力先: {output_path}"))
                
                # Show completion dialog after short delay
                self._safe_gui_update(lambda: self.root.after(2000, lambda: self._on_processing_complete(output_path, progress_dialog)))
                
            except KeyboardInterrupt:
                # Handle user cancellation
                error_msg = "Processing cancelled by user"
                if progress_dialog:
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(error_msg))
                self._safe_gui_update(lambda: self.root.after(1000, lambda: self._on_processing_error(error_msg, progress_dialog)))
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                full_traceback = traceback.format_exc()
                
                # 詳細なエラーログを出力
                logger.error(f"=== PROCESSING THREAD EXCEPTION ===")
                logger.error(f"Error message: {error_msg}")
                logger.error(f"Full traceback:")
                logger.error(full_traceback)
                logger.error(f"=== END PROCESSING THREAD EXCEPTION ===")
                
                if progress_dialog:
                    self._safe_gui_update(lambda: progress_dialog.add_log_message(f"Error: {error_msg}"))
                self._safe_gui_update(lambda: self.root.after(1000, lambda: self._on_processing_error(error_msg, progress_dialog)))
        
        self.processing_thread = threading.Thread(target=process_video, daemon=True)
        self.processing_thread.start()
    
    def _safe_gui_update(self, callback):
        """Safely update GUI from background thread"""
        try:
            if hasattr(self, 'root') and self.root and hasattr(self.root, 'winfo_exists'):
                if self.root.winfo_exists():
                    self.root.after_idle(callback)
        except (AttributeError, RuntimeError, tk.TclError):
            pass  # Ignore errors if window is destroyed or in invalid state
        
    def _on_processing_complete(self, output_path, progress_dialog):
        """Handle processing completion"""
        # Mark session as completed
        if self.current_session_id:
            self.session_manager.update_step_status(self.current_session_id, "combine", "completed", 100,
                                                  {"output_path": output_path})
        
        # === FIXED: Full cleanup on successful completion (no need to preserve frames) ===
        logger.info("DEBUG: Processing completed successfully, performing full cleanup")
        try:
            if self.video_processor:
                self.video_processor.cleanup(preserve_frames=False)
                logger.info("DEBUG: Full cleanup completed - all temp files removed")
        except Exception as e:
            logger.warning(f"DEBUG: Error during completion cleanup: {e}")
            
            # Clean up session after successful completion
            self.session_manager.cleanup_session(self.current_session_id)
            self.current_session_id = None
            logger.info("Session completed successfully and cleaned up")
        
        # Clean up temporary files completely after successful completion
        progress_dialog._cleanup_temp_files(preserve_frames=False)
        progress_dialog.destroy()
        
        self.is_processing = False
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Start Processing", state="normal")
            CTkMessagebox(
                title="Success!",
                message=f"Video processing completed successfully!\n\nOutput saved to:\n{output_path}",
                icon="check"
            )
        else:
            self.process_button.config(text="Start Processing", state="normal")
            messagebox.showinfo(
                "Success!",
                f"Video processing completed successfully!\n\nOutput saved to:\n{output_path}"
            )
        
        self._add_status_message(f"Processing completed: {Path(output_path).name}")
        
    def _on_processing_error(self, error_msg, progress_dialog):
        """Handle processing error"""
        # Save session state for potential resume (don't clean up on error/cancellation)
        if self.current_session_id:
            # Update session with error info but keep it for resume
            progress_data = self.session_manager.load_progress(self.current_session_id)
            if progress_data:
                current_step = self.session_manager._get_current_step(progress_data.get('steps', {}))
                
                # Determine if this is a cancellation or error
                is_cancellation = "cancelled" in str(error_msg).lower()
                status = "cancelled" if is_cancellation else "failed"
                
                self.session_manager.update_step_status(self.current_session_id, current_step, status, 0,
                                                      {"error": str(error_msg)})
                logger.info(f"Session {self.current_session_id} saved for potential resume after {status}")
            else:
                logger.warning(f"Could not save session state - no progress data found for {self.current_session_id}")
        
        # Clean up temporary files (preserve frames for resume on error)
        progress_dialog._cleanup_temp_files(preserve_frames=True)
        progress_dialog.destroy()
        
        self.is_processing = False
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Start Processing", state="normal")
            
            # エラー時にログファイルのパスも表示
            error_message = f"処理中にエラーが発生しました:\n\n{error_msg}"
            if self.log_file_path:
                error_message += f"\n\n詳細なログは以下のファイルで確認できます:\n{self.log_file_path}"
            
            CTkMessagebox(
                title="Processing Error", 
                message=error_message,
                icon="cancel"
            )
        else:
            self.process_button.config(text="Start Processing", state="normal")
            messagebox.showerror("Processing Error", f"Processing failed:\n\n{error_msg}")
        
        self._add_status_message(f"Processing failed: {error_msg}")
        
    def _reset_processing_state(self):
        """Reset processing state when cancelled"""
        self.is_processing = False
        if GUI_AVAILABLE and ctk:
            self.process_button.configure(text="Start Processing", state="normal")
        else:
            self.process_button.config(text="Start Processing", state="normal")
        self._add_status_message("Processing cancelled by user")
    
    def __del__(self):
        """Destructor to ensure cleanup on application exit"""
        try:
            # Clean up old sessions on app exit
            if hasattr(self, 'session_manager') and self.session_manager:
                self.session_manager.cleanup_old_sessions()
            
            if hasattr(self, 'video_processor') and self.video_processor:
                self.video_processor.cleanup(preserve_frames=True)
        except:
            pass
        
    def run(self) -> int:
        """Run the GUI application"""
        try:
            self._create_window()
            
            logger.info("Starting GUI application")
            self.root.mainloop()
            
            return 0
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
            return 1
            
    def cleanup(self):
        """Cleanup GUI resources"""
        try:
            # Cancel any ongoing processing
            if hasattr(self, 'processing_cancelled'):
                self.processing_cancelled = True
            
            # Stop any running threads/processes
            if hasattr(self, 'video_processor') and self.video_processor:
                self.video_processor.cleanup(preserve_frames=True)
            
            # Destroy GUI safely
            if self.root:
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
        except:
            pass
    
    def _display_video_info(self, file_path):
        """Display video information when file is selected"""
        try:
            # Validate video and get information
            video_info = self.video_processor.validate_video(file_path)
            
            # Clear existing info labels
            for label in self.video_info_labels.values():
                label.destroy()
            self.video_info_labels.clear()
            
            if video_info['valid']:
                info = video_info['info']
                
                # Hide placeholder
                self.video_info_placeholder.pack_forget()
                
                # Prepare info data
                info_data = [
                    ("ファイル名", info['filename']),
                    ("解像度", f"{info['width']} × {info['height']}"),
                    ("時間", f"{info['duration']:.1f} 秒"),
                    ("フレームレート", f"{info['frame_rate']:.2f} fps"),
                    ("フレーム数", f"{info['frame_count']:,} フレーム"),
                    ("コーデック", info['codec_name']),
                    ("ファイルサイズ", f"{info['size'] / (1024*1024):.1f} MB"),
                    ("フォーマット", info['format'])
                ]
                
                if GUI_AVAILABLE and ctk:
                    # CustomTkinter version
                    for i, (label_text, value_text) in enumerate(info_data):
                        row_frame = ctk.CTkFrame(self.video_info_container)
                        row_frame.pack(fill="x", padx=5, pady=2)
                        
                        label = ctk.CTkLabel(
                            row_frame, 
                            text=f"{label_text}:", 
                            font=ctk.CTkFont(size=12, weight="bold"),
                            width=100,
                            anchor="w"
                        )
                        label.pack(side="left", padx=(10, 5), pady=5)
                        
                        value = ctk.CTkLabel(
                            row_frame, 
                            text=str(value_text),
                            font=ctk.CTkFont(size=12),
                            anchor="w"
                        )
                        value.pack(side="left", padx=(5, 10), pady=5, fill="x", expand=True)
                        
                        self.video_info_labels[f"row_{i}"] = row_frame
                        
                else:
                    # Standard tkinter version
                    for i, (label_text, value_text) in enumerate(info_data):
                        row_frame = ttk.Frame(self.video_info_container)
                        row_frame.pack(fill="x", pady=2)
                        
                        label = ttk.Label(
                            row_frame, 
                            text=f"{label_text}:",
                            font=("TkDefaultFont", 9, "bold"),
                            width=15,
                            anchor="w"
                        )
                        label.pack(side="left", padx=(5, 5))
                        
                        value = ttk.Label(
                            row_frame, 
                            text=str(value_text),
                            anchor="w"
                        )
                        value.pack(side="left", padx=(5, 5), fill="x", expand=True)
                        
                        self.video_info_labels[f"row_{i}"] = row_frame
                        
            else:
                # Show error message
                self.video_info_placeholder.configure(text=f"エラー: {video_info['error']}")
                self.video_info_placeholder.pack(pady=10)
                
        except Exception as e:
            logger.warning(f"Failed to display video info: {e}")
            # Show placeholder with error
            self.video_info_placeholder.configure(text="動画情報の取得に失敗しました。")
            self.video_info_placeholder.pack(pady=10)
    
    def _check_resumable_sessions(self, file_path: str):
        """Check for resumable sessions when video file is selected"""
        try:
            logger.info(f"DEBUG: Checking for resumable sessions for {file_path}")
            
            # Get current settings
            settings = self._get_current_settings()
            logger.info(f"DEBUG: Current settings: {settings}")
            
            # Check for resumable session
            resumable_session = self.session_manager.find_resumable_session(file_path, settings)
            logger.info(f"DEBUG: Resumable session found: {resumable_session is not None}")
            
            if resumable_session:
                logger.info(f"Found resumable session for {Path(file_path).name}")
                
                # Show resume dialog
                resume_dialog = ResumeDialog(self.root, resumable_session)
                choice = resume_dialog.show()
                
                if choice == "resume":
                    self.current_session_id = resumable_session['session_id']
                    self._add_status_message("前回のセッションから再開準備完了 - 「処理開始」ボタンを押してください")
                    logger.info(f"User chose to resume session {self.current_session_id}")
                    
                    # === FIXED: 自動処理開始を削除 - ユーザーが手動で開始する ===
                    logger.info("Resume session prepared - waiting for user to click start button")
                elif choice == "restart":
                    # Clean up old session
                    self.session_manager.cleanup_session(resumable_session['session_id'])
                    self.current_session_id = None
                    self._add_status_message("新しい処理として開始準備完了")
                    logger.info("User chose to restart processing")
                else:
                    # Cancel - clear session
                    self.current_session_id = None
                    logger.info("User cancelled resume dialog")
            else:
                # No resumable session found
                logger.info(f"DEBUG: No resumable session found for {file_path}")
                self.current_session_id = None
                
        except Exception as e:
            logger.error(f"DEBUG: Error checking resumable sessions: {e}")
            import traceback
            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            self.current_session_id = None
    
    def _get_current_settings(self) -> Dict[str, Any]:
        """Get current processing settings"""
        return {
            'scale_factor': float(self.scale_var.get().replace('x', '')),
            'quality': self.quality_var.get(),
            'noise_reduction': 3  # Default value
        }
    
    def _upscale_frames_with_resume(self, remaining_frames: List[str], completed_frames: List[str], 
                                   scale_factor: float, progress_callback: Callable, 
                                   progress_dialog) -> List[str]:
        """Upscale frames with resume functionality"""
        try:
            # Create custom progress callback that tracks completed frames
            def resume_progress_callback(progress, message):
                # Update session with newly completed frames during processing
                # This is called by ai_processor during frame processing
                progress_callback(progress, message)
            
            # Custom AI processor call that tracks individual frame completion
            all_upscaled_frames = self._upscale_with_tracking(
                remaining_frames, scale_factor, resume_progress_callback, progress_dialog
            )
            
            # === DEBUG: フレーム収集開始時の詳細情報 ===
            logger.info(f"=== FRAME COLLECTION DEBUG START ===")
            
            # Get completed frames from session management
            session_completed_frames = self.session_manager.get_completed_frames(self.current_session_id)
            logger.info(f"Session completed frames count: {len(session_completed_frames)}")
            
            # Also get frames from directory for verification
            session_dir = self.session_manager.get_session_dir(self.current_session_id)
            upscaled_dir = session_dir / "upscaled"
            directory_frames = [str(p) for p in upscaled_dir.glob("*_upscaled.png")]
            logger.info(f"Directory frames count: {len(directory_frames)}")
            
            # Use session management frames if available, otherwise fallback to directory scan
            if session_completed_frames:
                existing_upscaled = session_completed_frames
                logger.info(f"Using session management frames: {len(existing_upscaled)} frames")
            else:
                existing_upscaled = directory_frames
                logger.info(f"Fallback to directory scan: {len(existing_upscaled)} frames")
            
            # === DEBUG: 最終フレーム情報 ===
            if existing_upscaled:
                logger.info(f"First frame: {existing_upscaled[0] if existing_upscaled else 'None'}")
                logger.info(f"Last frame: {existing_upscaled[-1] if existing_upscaled else 'None'}")
                logger.info(f"Total frames returning: {len(existing_upscaled)}")
            
            return sorted(existing_upscaled)
            
        except Exception as e:
            logger.error(f"Error in resume upscaling: {e}")
            raise
    
    def _upscale_with_tracking(self, frame_paths: List[str], scale_factor: float, 
                              progress_callback: Callable, progress_dialog) -> List[str]:
        """Upscale frames with individual frame tracking for resume using parallel processing"""
        session_dir = self.session_manager.get_session_dir(self.current_session_id)
        output_dir = str(session_dir / "upscaled")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Filter out already processed frames for efficiency
        remaining_frames = []
        processed_frames = []
        
        for frame_path in frame_paths:
            frame_name = Path(frame_path).stem
            output_path = Path(output_dir) / f"{frame_name}_upscaled.png"
            
            if output_path.exists():
                processed_frames.append(str(output_path))
                self.session_manager.add_completed_frame(self.current_session_id, str(output_path))
            else:
                remaining_frames.append(frame_path)
        
        if not remaining_frames:
            logger.info("All frames already processed, skipping upscaling")
            return processed_frames
        
        logger.info(f"Processing {len(remaining_frames)} remaining frames (out of {len(frame_paths)} total)")
        
        # Use parallel processing through AIProcessor.upscale_frames instead of individual backend calls
        def tracking_progress_callback(progress, message):
            progress_callback(progress, message)
        
        try:
            # Call the main upscale_frames method to use parallel processing
            upscaled_frames = self.ai_processor.upscale_frames(
                remaining_frames, output_dir, scale_factor, 
                tracking_progress_callback, progress_dialog
            )
            
            # === DEBUG: 並列処理結果の確認 ===
            logger.info(f"Parallel processing returned {len(upscaled_frames)} frames")
            
            # Add completed frames to session tracking - use directory scan to ensure all frames are captured
            output_path = Path(output_dir)
            all_output_files = list(output_path.glob("*_upscaled.png"))
            logger.info(f"Directory contains {len(all_output_files)} upscaled files")
            
            # Add all files found in directory to session management
            logger.info(f"=== GUI DEBUG: Processing {len(all_output_files)} files for session management ===")
            added_count = 0
            for i, file_path in enumerate(all_output_files):
                full_path = str(file_path.absolute())
                
                # Debug log every 100th frame or first 5 frames
                if i < 5 or i % 100 == 0:
                    logger.info(f"Adding frame {i+1}/{len(all_output_files)}: {full_path}")
                
                self.session_manager.add_completed_frame(self.current_session_id, full_path)
                added_count += 1
                
                if full_path not in processed_frames:
                    processed_frames.append(full_path)
            
            logger.info(f"=== GUI DEBUG: Completed adding {added_count} frames to session management ===")
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"Error in parallel frame processing: {e}")
            # Fallback to sequential processing if parallel fails
            return self._upscale_sequential_fallback(remaining_frames, output_dir, scale_factor, progress_callback, progress_dialog, processed_frames)
    
    def _upscale_sequential_fallback(self, frame_paths: List[str], output_dir: str, scale_factor: float, 
                                   progress_callback: Callable, progress_dialog, processed_frames: List[str]) -> List[str]:
        """Sequential fallback for frame processing"""
        logger.info("Using sequential fallback processing")
        total_frames = len(frame_paths)
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Check for cancellation
                if progress_dialog and progress_dialog.cancelled:
                    raise KeyboardInterrupt("Processing cancelled by user")
                
                # Generate output path
                frame_name = Path(frame_path).stem
                output_path = Path(output_dir) / f"{frame_name}_upscaled.png"
                
                # Upscale single frame
                success = self.ai_processor.backend.upscale_image(
                    frame_path, str(output_path), scale_factor, 
                    progress_dialog=progress_dialog if i < 3 else None  # Debug for first 3 frames
                )
                
                if success:
                    processed_frames.append(str(output_path))
                    # Update session with completed frame
                    self.session_manager.add_completed_frame(self.current_session_id, str(output_path))
                    
                    # Update progress
                    frame_progress = ((i + 1) / total_frames) * 100
                    progress_callback(frame_progress, f"Processed frame {i + 1}/{total_frames}")
                else:
                    logger.warning(f"Failed to upscale frame: {frame_path}")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {e}")
                # Continue with next frame instead of failing completely
                continue
        
        return processed_frames
    
    def _open_log_file(self):
        """ログファイルをシステムのデフォルトエディタで開く"""
        try:
            if self.log_file_path and Path(self.log_file_path).exists():
                import subprocess
                import platform
                
                system = platform.system()
                if system == "Windows":
                    # Windowsでメモ帳やデフォルトエディタで開く
                    subprocess.run(['notepad', str(self.log_file_path)], check=False)
                elif system == "Darwin":  # macOS
                    subprocess.run(['open', str(self.log_file_path)], check=False)
                else:  # Linux
                    subprocess.run(['xdg-open', str(self.log_file_path)], check=False)
                    
                # ログファイルの場所もログに記録
                logger.info(f"Opened log file: {self.log_file_path}")
                
                # GUIにもメッセージ表示
                if GUI_AVAILABLE and ctk:
                    CTkMessagebox(
                        title="ログファイル",
                        message=f"ログファイルを開きました:\n{self.log_file_path}",
                        icon="info"
                    )
                
            else:
                error_msg = "ログファイルが見つかりません"
                logger.warning(error_msg)
                if GUI_AVAILABLE and ctk:
                    CTkMessagebox(
                        title="エラー",
                        message=error_msg,
                        icon="cancel"
                    )
                    
        except Exception as e:
            error_msg = f"ログファイルを開けませんでした: {e}"
            logger.error(error_msg)
            if GUI_AVAILABLE and ctk:
                CTkMessagebox(
                    title="エラー",
                    message=error_msg,
                    icon="cancel"
                )
    
    def _continue_partial_processing(self, session_id, input_path, output_path, processed_frames, total_frames):
        """部分処理の継続を実行する"""
        logger.info(f"=== CONTINUING PARTIAL PROCESSING ===")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Already processed: {processed_frames}/{total_frames} frames")
        logger.info(f"Remaining frames: {total_frames - processed_frames}")
        
        try:
            # 新しい進捗ダイアログを作成
            progress_dialog = None
            if self.use_modern_gui:
                progress_dialog = ProgressDialog(self.window, "継続処理中...")
                progress_dialog.add_step("upscale", "残りフレームの処理")
                progress_dialog.add_step("combine", "最終動画の作成")
                progress_dialog.show()
            
            def continue_progress_callback(progress, message):
                logger.info(f"Continue progress: {progress:.1f}% - {message}")
                if progress_dialog:
                    # 全体の進捗を更新
                    progress_dialog.update_progress(progress, message)
            
            # 残りのフレームを処理
            self._continue_upscaling_from_frame(
                session_id, 
                input_path, 
                output_path,
                processed_frames + 1,  # 次のフレームから開始
                total_frames,
                continue_progress_callback,
                progress_dialog
            )
            
            if progress_dialog:
                progress_dialog.hide()
            
            # 成功メッセージ
            success_msg = (
                f"継続処理が完了しました！\n\n"
                f"処理されたフレーム: {total_frames}/{total_frames}\n"
                f"完全な動画が作成されました。"
            )
            
            if self.use_modern_gui:
                import customtkinter as ctk
                CTkMessagebox(
                    title="継続処理完了",
                    message=success_msg,
                    icon="check"
                )
            else:
                import tkinter.messagebox as messagebox
                messagebox.showinfo("継続処理完了", success_msg)
                
        except Exception as e:
            logger.error(f"Continue processing failed: {e}")
            if progress_dialog:
                progress_dialog.hide()
                
            error_msg = f"継続処理中にエラーが発生しました:\n{str(e)}"
            if self.use_modern_gui:
                import customtkinter as ctk
                CTkMessagebox(
                    title="継続処理エラー",
                    message=error_msg,
                    icon="cancel"
                )
            else:
                import tkinter.messagebox as messagebox
                messagebox.showerror("継続処理エラー", error_msg)
    
    def _continue_upscaling_from_frame(self, session_id, input_path, output_path, start_frame, total_frames, progress_callback, progress_dialog):
        """指定されたフレーム番号から処理を継続する"""
        logger.info(f"=== CONTINUING UPSCALING FROM FRAME {start_frame} ===")
        
        # セッションの進捗データを取得
        session_progress = self.session_manager.get_session_progress(session_id)
        if not session_progress:
            raise RuntimeError(f"Session {session_id} not found")
        
        # ビデオ情報を取得
        video_info = self.video_processor.validate_video(input_path)
        if not video_info['valid']:
            raise RuntimeError("Invalid video file")
        
        # フレーム抽出がまだ完了していない場合は実行
        extract_step = session_progress['steps'].get('extract', {})
        if extract_step.get('status') != 'completed':
            logger.info("Frame extraction not completed, running extraction first...")
            self.video_processor.extract_frames(
                input_path,
                self.video_processor.temp_dir,
                lambda p, m: progress_callback(p * 0.3, f"フレーム抽出: {m}")
            )
        
        # 未処理のフレームを特定
        frames_dir = Path(self.video_processor.temp_dir)
        frame_files = sorted([f for f in frames_dir.glob("frame_*.png")])
        
        remaining_frames = frame_files[start_frame-1:]  # 0ベースのインデックス
        logger.info(f"Processing {len(remaining_frames)} remaining frames")
        
        if progress_dialog:
            progress_dialog.update_step_progress("upscale", 0, "準備中")
        
        # 残りのフレームを処理
        for i, frame_file in enumerate(remaining_frames):
            try:
                # AI処理
                result = self.ai_processor.upscale_image(str(frame_file))
                if result:
                    # セッションに追加
                    self.session_manager.add_completed_frame(session_id, result)
                    
                    # 進捗更新
                    current_frame = start_frame + i
                    frame_progress = (current_frame / total_frames) * 100
                    
                    if progress_dialog:
                        progress_dialog.update_step_progress("upscale", frame_progress)
                    
                    progress_callback(30 + (frame_progress * 0.55), f"フレーム処理: {current_frame}/{total_frames}")
                    
                    logger.info(f"Processed frame {current_frame}/{total_frames}")
                else:
                    logger.warning(f"Failed to process frame: {frame_file}")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_file}: {e}")
                continue
        
        # アップスケール完了をセッションに記録
        self.session_manager.update_step_status(session_id, "upscale", "completed")
        
        if progress_dialog:
            progress_dialog.update_step_progress("upscale", 100, "完了")
            progress_dialog.update_step_progress("combine", 0, "準備中")
        
        # 最終的な動画結合
        logger.info("Creating final complete video...")
        all_processed_frames = self.session_manager.get_completed_frames(session_id)
        
        def combine_progress_callback(progress, message):
            if progress_dialog:
                progress_dialog.update_step_progress("combine", progress)
            progress_callback(85 + (progress * 0.15), f"動画結合: {message}")
        
        combine_result = self.video_processor.combine_frames_to_video(
            all_processed_frames,
            output_path,
            input_path,
            video_info['info']['frame_rate'],
            combine_progress_callback
        )
        
        if isinstance(combine_result, dict) and not combine_result.get('success'):
            raise RuntimeError(combine_result.get('error', 'Failed to create final video'))
        elif combine_result == False:
            raise RuntimeError('Failed to create final video')
        
        # 結合完了をセッションに記録
        self.session_manager.update_step_status(session_id, "combine", "completed")
        
        if progress_dialog:
            progress_dialog.update_step_progress("combine", 100, "完了")
        
        progress_callback(100, "継続処理完了！")
        
        logger.info(f"Continue processing completed successfully for session {session_id}")
    
    def _prompt_continue_extraction(self, missing_frames, completion_percentage):
        """未完了のフレーム抽出を継続するかユーザーに確認"""
        message = (
            f"⚠️ フレーム抽出が未完了です\n\n"
            f"抽出完了率: {completion_percentage:.1f}%\n"
            f"未抽出フレーム: {missing_frames}フレーム\n\n"
            f"現在の状況:\n"
            f"• フレーム抽出が途中で停止しました\n"
            f"• このまま処理すると不完全な動画になります\n\n"
            f"残りのフレームを抽出しますか？"
        )
        
        try:
            if self.use_modern_gui:
                import customtkinter as ctk
                dialog = ctk.CTkToplevel(self.window)
                dialog.title("フレーム抽出の継続確認")
                dialog.geometry("480x350")
                dialog.transient(self.window)
                dialog.grab_set()
                
                # センタリング
                dialog.update_idletasks()
                x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_reqwidth() // 2)
                y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_reqheight() // 2)
                dialog.geometry(f"+{x}+{y}")
                
                main_frame = ctk.CTkFrame(dialog)
                main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                
                text_widget = ctk.CTkTextbox(main_frame, wrap="word", height=200)
                text_widget.pack(fill="both", expand=True, pady=(0, 15))
                text_widget.insert("1.0", message)
                text_widget.configure(state="disabled")
                
                button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
                button_frame.pack(fill="x")
                
                result = {"continue": False}
                
                def on_continue():
                    result["continue"] = True
                    dialog.destroy()
                
                def on_skip():
                    dialog.destroy()
                
                continue_button = ctk.CTkButton(
                    button_frame, 
                    text="継続抽出する", 
                    command=on_continue,
                    fg_color="#2B8B3D",
                    hover_color="#1F6B2B"
                )
                continue_button.pack(side="left", padx=(0, 10))
                
                skip_button = ctk.CTkButton(
                    button_frame, 
                    text="部分的な処理で続行", 
                    command=on_skip,
                    fg_color="#D32F2F",
                    hover_color="#B71C1C"
                )
                skip_button.pack(side="right")
                
                dialog.wait_window()
                return result["continue"]
            else:
                import tkinter.messagebox as messagebox
                return messagebox.askyesno(
                    "フレーム抽出の継続確認", 
                    message + "\n\n継続抽出しますか？",
                    icon='warning'
                )
        except Exception as e:
            logger.error(f"Error showing extraction prompt: {e}")
            return False
    
    def _continue_frame_extraction(self, input_path, output_dir, start_frame, total_frames, progress_callback):
        """未完了のフレーム抽出を継続する"""
        logger.info(f"=== CONTINUING FRAME EXTRACTION ===")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Start frame: {start_frame + 1}")
        logger.info(f"Total frames: {total_frames}")
        
        try:
            # 継続抽出用の進捗コールバック
            def extract_progress_callback(progress, message):
                # フレーム抽出の進捗を15-50%にマッピング
                adjusted_progress = 15 + (progress * 0.35)
                progress_callback(adjusted_progress, f"継続フレーム抽出: {message}")
                
            # FastFrameExtractorを使用して継続抽出
            remaining_frames = total_frames - start_frame
            logger.info(f"Extracting {remaining_frames} remaining frames starting from frame {start_frame + 1}")
            
            # 継続抽出を実行（範囲指定）
            success = self.video_processor.fast_frame_extractor.extract_frames_range(
                str(input_path),
                str(output_dir),
                start_frame + 1,  # 次のフレームから開始
                total_frames,
                extract_progress_callback
            )
            
            if not success:
                raise RuntimeError("Continue frame extraction failed")
                
            # セッションの抽出ステータスを更新
            progress_data = self.session_manager.load_progress(self.current_session_id)
            if progress_data:
                extract_step = progress_data['steps']['extract']
                extract_step['status'] = 'completed'
                extract_step['actual_frames'] = total_frames
                extract_step['completion_percentage'] = 100.0
                extract_step['continued_extraction'] = True
                self.session_manager.save_progress(self.current_session_id, progress_data)
                
            logger.info(f"Continue frame extraction completed successfully")
            progress_callback(50, "フレーム抽出継続完了")
            
        except Exception as e:
            logger.error(f"Continue frame extraction failed: {e}")
            progress_callback(15, f"継続抽出エラー: {str(e)}")
            raise