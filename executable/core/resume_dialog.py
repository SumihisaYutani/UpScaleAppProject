"""
UpScale App - Resume Dialog Module
Handles UI for session resume functionality
"""

import tkinter as tk
from tkinter import ttk
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

# Try to import CustomTkinter
try:
    import customtkinter as ctk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    ctk = None

logger = logging.getLogger(__name__)

class ResumeDialog:
    """Dialog for handling session resume options"""
    
    def __init__(self, parent, session_data: Dict[str, Any]):
        self.parent = parent
        self.session_data = session_data
        self.result = None
        self.dialog = None
        
    def show(self) -> Optional[str]:
        """Show resume dialog and return user choice"""
        if GUI_AVAILABLE and ctk:
            return self._show_ctk_dialog()
        else:
            return self._show_tk_dialog()
    
    def _show_ctk_dialog(self) -> Optional[str]:
        """Show CustomTkinter resume dialog"""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title("処理の再開")
        self.dialog.geometry("600x500")
        self.dialog.resizable(False, False)
        
        # Make modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.parent.update_idletasks()
        x = (self.parent.winfo_x() + self.parent.winfo_width() // 2) - 300
        y = (self.parent.winfo_y() + self.parent.winfo_height() // 2) - 250
        self.dialog.geometry(f"+{x}+{y}")
        
        self._setup_ctk_content()
        
        # Wait for user choice
        self.dialog.wait_window()
        return self.result
    
    def _show_tk_dialog(self) -> Optional[str]:
        """Show standard Tkinter resume dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("処理の再開")
        self.dialog.geometry("600x500")
        self.dialog.resizable(False, False)
        
        # Make modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.parent.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - 300
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - 250
        self.dialog.geometry(f"+{x}+{y}")
        
        self._setup_tk_content()
        
        # Wait for user choice
        self.dialog.wait_window()
        return self.result
    
    def _setup_ctk_content(self):
        """Setup CustomTkinter dialog content"""
        # Main frame
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="🔄 前回の処理が見つかりました",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Session info frame
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        # Session details
        self._add_ctk_session_info(info_frame)
        
        # Progress section
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        progress_title = ctk.CTkLabel(
            progress_frame,
            text="📊 処理進行状況",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        progress_title.pack(pady=(10, 5))
        
        self._add_ctk_progress_info(progress_frame)
        
        # Options section
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        options_title = ctk.CTkLabel(
            options_frame,
            text="⚙️ 選択してください",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        options_title.pack(pady=(10, 10))
        
        # Button frame
        button_frame = ctk.CTkFrame(options_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 15))
        
        # Resume button
        resume_btn = ctk.CTkButton(
            button_frame,
            text="🚀 途中から再開",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=lambda: self._on_choice("resume")
        )
        resume_btn.pack(side="left", padx=(10, 5), pady=5, fill="x", expand=True)
        
        # Restart button
        restart_btn = ctk.CTkButton(
            button_frame,
            text="🔄 最初から開始",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="orange",
            hover_color="darkorange",
            command=lambda: self._on_choice("restart")
        )
        restart_btn.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ キャンセル",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="gray",
            hover_color="darkgray",
            command=lambda: self._on_choice("cancel")
        )
        cancel_btn.pack(side="right", padx=(5, 10), pady=5, fill="x", expand=True)
    
    def _setup_tk_content(self):
        """Setup standard Tkinter dialog content"""
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="前回の処理が見つかりました",
            font=("TkDefaultFont", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Session info
        info_frame = ttk.LabelFrame(main_frame, text="セッション情報", padding=10)
        info_frame.pack(fill="x", pady=(0, 15))
        
        self._add_tk_session_info(info_frame)
        
        # Progress info
        progress_frame = ttk.LabelFrame(main_frame, text="処理進行状況", padding=10)
        progress_frame.pack(fill="x", pady=(0, 15))
        
        self._add_tk_progress_info(progress_frame)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="途中から再開",
            command=lambda: self._on_choice("resume")
        ).pack(side="left", padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="最初から開始",
            command=lambda: self._on_choice("restart")
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame,
            text="キャンセル",
            command=lambda: self._on_choice("cancel")
        ).pack(side="right")
    
    def _add_ctk_session_info(self, parent):
        """Add session information to CustomTkinter frame"""
        info_container = ctk.CTkFrame(parent)
        info_container.pack(fill="x", padx=10, pady=10)
        
        # Video name
        video_name = Path(self.session_data.get('video_file', '')).name
        self._add_ctk_info_row(info_container, "動画ファイル:", video_name)
        
        # Last updated
        last_updated = self.session_data.get('last_updated', '')
        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                self._add_ctk_info_row(info_container, "最終更新:", formatted_time)
            except:
                self._add_ctk_info_row(info_container, "最終更新:", last_updated)
        
        # Settings
        settings = self.session_data.get('settings', {})
        scale_factor = settings.get('scale_factor', 'Unknown')
        quality = settings.get('quality', 'Unknown')
        self._add_ctk_info_row(info_container, "拡大率:", f"{scale_factor}x")
        self._add_ctk_info_row(info_container, "品質:", quality)
    
    def _add_tk_session_info(self, parent):
        """Add session information to Tkinter frame"""
        # Video name
        video_name = Path(self.session_data.get('video_file', '')).name
        self._add_tk_info_row(parent, "動画ファイル:", video_name)
        
        # Last updated
        last_updated = self.session_data.get('last_updated', '')
        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                self._add_tk_info_row(parent, "最終更新:", formatted_time)
            except:
                self._add_tk_info_row(parent, "最終更新:", last_updated)
        
        # Settings
        settings = self.session_data.get('settings', {})
        scale_factor = settings.get('scale_factor', 'Unknown')
        quality = settings.get('quality', 'Unknown')
        self._add_tk_info_row(parent, "拡大率:", f"{scale_factor}x")
        self._add_tk_info_row(parent, "品質:", quality)
    
    def _add_ctk_progress_info(self, parent):
        """Add progress information to CustomTkinter frame"""
        progress_container = ctk.CTkFrame(parent)
        progress_container.pack(fill="x", padx=10, pady=10)
        
        steps = self.session_data.get('steps', {})
        step_names = {
            'validate': '動画検証',
            'extract': 'フレーム抽出',
            'upscale': 'AI処理',
            'combine': '動画結合'
        }
        
        for step_id, step_name in step_names.items():
            step_data = steps.get(step_id, {})
            status = step_data.get('status', 'pending')
            progress = step_data.get('progress', 0)
            
            # Step row
            step_frame = ctk.CTkFrame(progress_container)
            step_frame.pack(fill="x", padx=5, pady=2)
            
            # Status emoji
            status_emoji = self._get_status_emoji(status)
            
            # Step label
            step_label = ctk.CTkLabel(
                step_frame,
                text=f"{status_emoji} {step_name}",
                font=ctk.CTkFont(size=12),
                anchor="w"
            )
            step_label.pack(side="left", padx=(10, 5), pady=5)
            
            # Progress bar
            progress_bar = ctk.CTkProgressBar(step_frame, width=150)
            progress_bar.pack(side="right", padx=(5, 10), pady=5)
            progress_bar.set(progress / 100 if progress > 0 else 0)
            
            # Progress text
            progress_text = ctk.CTkLabel(
                step_frame,
                text=f"{progress:.1f}%",
                font=ctk.CTkFont(size=10)
            )
            progress_text.pack(side="right", padx=(5, 5), pady=5)
    
    def _add_tk_progress_info(self, parent):
        """Add progress information to Tkinter frame"""
        steps = self.session_data.get('steps', {})
        step_names = {
            'validate': '動画検証',
            'extract': 'フレーム抽出',
            'upscale': 'AI処理',
            'combine': '動画結合'
        }
        
        for i, (step_id, step_name) in enumerate(step_names.items()):
            step_data = steps.get(step_id, {})
            status = step_data.get('status', 'pending')
            progress = step_data.get('progress', 0)
            
            # Status
            status_text = self._get_status_text(status)
            
            # Row
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill="x", pady=2)
            
            ttk.Label(
                row_frame,
                text=f"{step_name}:",
                width=12
            ).pack(side="left")
            
            ttk.Label(
                row_frame,
                text=status_text
            ).pack(side="left", padx=(5, 10))
            
            ttk.Label(
                row_frame,
                text=f"{progress:.1f}%"
            ).pack(side="right")
    
    def _add_ctk_info_row(self, parent, label: str, value: str):
        """Add information row to CustomTkinter frame"""
        row_frame = ctk.CTkFrame(parent)
        row_frame.pack(fill="x", padx=5, pady=2)
        
        label_widget = ctk.CTkLabel(
            row_frame,
            text=label,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=100,
            anchor="w"
        )
        label_widget.pack(side="left", padx=(10, 5), pady=5)
        
        value_widget = ctk.CTkLabel(
            row_frame,
            text=str(value),
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        value_widget.pack(side="left", padx=(5, 10), pady=5, fill="x", expand=True)
    
    def _add_tk_info_row(self, parent, label: str, value: str):
        """Add information row to Tkinter frame"""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=2)
        
        ttk.Label(
            row_frame,
            text=label,
            font=("TkDefaultFont", 9, "bold"),
            width=12
        ).pack(side="left")
        
        ttk.Label(
            row_frame,
            text=str(value)
        ).pack(side="left", padx=(5, 0), fill="x", expand=True)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for step status"""
        emoji_map = {
            'pending': '⏳',
            'in_progress': '🔄',
            'completed': '✅',
            'failed': '❌'
        }
        return emoji_map.get(status, '❓')
    
    def _get_status_text(self, status: str) -> str:
        """Get text for step status"""
        status_map = {
            'pending': '待機中',
            'in_progress': '実行中',
            'completed': '完了',
            'failed': '失敗'
        }
        return status_map.get(status, '不明')
    
    def _on_choice(self, choice: str):
        """Handle user choice"""
        self.result = choice
        if self.dialog:
            self.dialog.destroy()

class SessionSelectionDialog:
    """Dialog for selecting from multiple resumable sessions"""
    
    def __init__(self, parent, sessions: List[Dict[str, Any]]):
        self.parent = parent
        self.sessions = sessions
        self.result = None
        self.dialog = None
    
    def show(self) -> Optional[Dict[str, Any]]:
        """Show session selection dialog"""
        if not self.sessions:
            return None
        
        if len(self.sessions) == 1:
            # Single session - show resume dialog directly
            resume_dialog = ResumeDialog(self.parent, self.sessions[0])
            choice = resume_dialog.show()
            if choice == "resume":
                return self.sessions[0]
            return None
        
        # Multiple sessions - show selection dialog
        if GUI_AVAILABLE and ctk:
            return self._show_ctk_selection()
        else:
            return self._show_tk_selection()
    
    def _show_ctk_selection(self) -> Optional[Dict[str, Any]]:
        """Show CustomTkinter session selection dialog"""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title("セッション選択")
        self.dialog.geometry("700x400")
        
        # Make modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.parent.update_idletasks()
        x = (self.parent.winfo_x() + self.parent.winfo_width() // 2) - 350
        y = (self.parent.winfo_y() + self.parent.winfo_height() // 2) - 200
        self.dialog.geometry(f"+{x}+{y}")
        
        # Title
        title_label = ctk.CTkLabel(
            self.dialog,
            text="複数の未完了セッションが見つかりました",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Sessions list (simplified implementation)
        # In a full implementation, you'd show a scrollable list
        # For now, just select the most recent session
        most_recent = max(self.sessions, key=lambda s: s.get('last_updated', ''))
        
        # Show resume dialog for most recent
        self.dialog.destroy()
        resume_dialog = ResumeDialog(self.parent, most_recent)
        choice = resume_dialog.show()
        if choice == "resume":
            return most_recent
        return None
    
    def _show_tk_selection(self) -> Optional[Dict[str, Any]]:
        """Show Tkinter session selection dialog"""
        # Simplified implementation - select most recent
        most_recent = max(self.sessions, key=lambda s: s.get('last_updated', ''))
        resume_dialog = ResumeDialog(self.parent, most_recent)
        choice = resume_dialog.show()
        if choice == "resume":
            return most_recent
        return None