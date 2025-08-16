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
        options_frame.pack(fill="x", padx=10, pady=(0, 10))\n        
        options_title = ctk.CTkLabel(\n            options_frame,\n            text=\"⚙️ 選択してください\",\n            font=ctk.CTkFont(size=14, weight=\"bold\")\n        )\n        options_title.pack(pady=(10, 10))\n        \n        # Button frame\n        button_frame = ctk.CTkFrame(options_frame)\n        button_frame.pack(fill=\"x\", padx=10, pady=(0, 15))\n        \n        # Resume button\n        resume_btn = ctk.CTkButton(\n            button_frame,\n            text=\"🚀 途中から再開\",\n            font=ctk.CTkFont(size=14, weight=\"bold\"),\n            height=40,\n            command=lambda: self._on_choice(\"resume\")\n        )\n        resume_btn.pack(side=\"left\", padx=(10, 5), pady=5, fill=\"x\", expand=True)\n        \n        # Restart button\n        restart_btn = ctk.CTkButton(\n            button_frame,\n            text=\"🔄 最初から開始\",\n            font=ctk.CTkFont(size=14),\n            height=40,\n            fg_color=\"orange\",\n            hover_color=\"darkorange\",\n            command=lambda: self._on_choice(\"restart\")\n        )\n        restart_btn.pack(side=\"left\", padx=5, pady=5, fill=\"x\", expand=True)\n        \n        # Cancel button\n        cancel_btn = ctk.CTkButton(\n            button_frame,\n            text=\"❌ キャンセル\",\n            font=ctk.CTkFont(size=14),\n            height=40,\n            fg_color=\"gray\",\n            hover_color=\"darkgray\",\n            command=lambda: self._on_choice(\"cancel\")\n        )\n        cancel_btn.pack(side=\"right\", padx=(5, 10), pady=5, fill=\"x\", expand=True)\n    \n    def _setup_tk_content(self):\n        \"\"\"Setup standard Tkinter dialog content\"\"\"\n        # Main frame\n        main_frame = ttk.Frame(self.dialog)\n        main_frame.pack(fill=\"both\", expand=True, padx=20, pady=20)\n        \n        # Title\n        title_label = ttk.Label(\n            main_frame,\n            text=\"前回の処理が見つかりました\",\n            font=(\"TkDefaultFont\", 14, \"bold\")\n        )\n        title_label.pack(pady=(0, 20))\n        \n        # Session info\n        info_frame = ttk.LabelFrame(main_frame, text=\"セッション情報\", padding=10)\n        info_frame.pack(fill=\"x\", pady=(0, 15))\n        \n        self._add_tk_session_info(info_frame)\n        \n        # Progress info\n        progress_frame = ttk.LabelFrame(main_frame, text=\"処理進行状況\", padding=10)\n        progress_frame.pack(fill=\"x\", pady=(0, 15))\n        \n        self._add_tk_progress_info(progress_frame)\n        \n        # Buttons\n        button_frame = ttk.Frame(main_frame)\n        button_frame.pack(fill=\"x\", pady=(10, 0))\n        \n        ttk.Button(\n            button_frame,\n            text=\"途中から再開\",\n            command=lambda: self._on_choice(\"resume\")\n        ).pack(side=\"left\", padx=(0, 5))\n        \n        ttk.Button(\n            button_frame,\n            text=\"最初から開始\",\n            command=lambda: self._on_choice(\"restart\")\n        ).pack(side=\"left\", padx=5)\n        \n        ttk.Button(\n            button_frame,\n            text=\"キャンセル\",\n            command=lambda: self._on_choice(\"cancel\")\n        ).pack(side=\"right\")\n    \n    def _add_ctk_session_info(self, parent):\n        \"\"\"Add session information to CustomTkinter frame\"\"\"\n        info_container = ctk.CTkFrame(parent)\n        info_container.pack(fill=\"x\", padx=10, pady=10)\n        \n        # Video name\n        video_name = Path(self.session_data.get('video_file', '')).name\n        self._add_ctk_info_row(info_container, \"動画ファイル:\", video_name)\n        \n        # Last updated\n        last_updated = self.session_data.get('last_updated', '')\n        if last_updated:\n            try:\n                dt = datetime.fromisoformat(last_updated)\n                formatted_time = dt.strftime(\"%Y-%m-%d %H:%M:%S\")\n                self._add_ctk_info_row(info_container, \"最終更新:\", formatted_time)\n            except:\n                self._add_ctk_info_row(info_container, \"最終更新:\", last_updated)\n        \n        # Settings\n        settings = self.session_data.get('settings', {})\n        scale_factor = settings.get('scale_factor', 'Unknown')\n        quality = settings.get('quality', 'Unknown')\n        self._add_ctk_info_row(info_container, \"拡大率:\", f\"{scale_factor}x\")\n        self._add_ctk_info_row(info_container, \"品質:\", quality)\n    \n    def _add_tk_session_info(self, parent):\n        \"\"\"Add session information to Tkinter frame\"\"\"\n        # Video name\n        video_name = Path(self.session_data.get('video_file', '')).name\n        self._add_tk_info_row(parent, \"動画ファイル:\", video_name)\n        \n        # Last updated\n        last_updated = self.session_data.get('last_updated', '')\n        if last_updated:\n            try:\n                dt = datetime.fromisoformat(last_updated)\n                formatted_time = dt.strftime(\"%Y-%m-%d %H:%M:%S\")\n                self._add_tk_info_row(parent, \"最終更新:\", formatted_time)\n            except:\n                self._add_tk_info_row(parent, \"最終更新:\", last_updated)\n        \n        # Settings\n        settings = self.session_data.get('settings', {})\n        scale_factor = settings.get('scale_factor', 'Unknown')\n        quality = settings.get('quality', 'Unknown')\n        self._add_tk_info_row(parent, \"拡大率:\", f\"{scale_factor}x\")\n        self._add_tk_info_row(parent, \"品質:\", quality)\n    \n    def _add_ctk_progress_info(self, parent):\n        \"\"\"Add progress information to CustomTkinter frame\"\"\"\n        progress_container = ctk.CTkFrame(parent)\n        progress_container.pack(fill=\"x\", padx=10, pady=10)\n        \n        steps = self.session_data.get('steps', {})\n        step_names = {\n            'validate': '動画検証',\n            'extract': 'フレーム抽出',\n            'upscale': 'AI処理',\n            'combine': '動画結合'\n        }\n        \n        for step_id, step_name in step_names.items():\n            step_data = steps.get(step_id, {})\n            status = step_data.get('status', 'pending')\n            progress = step_data.get('progress', 0)\n            \n            # Step row\n            step_frame = ctk.CTkFrame(progress_container)\n            step_frame.pack(fill=\"x\", padx=5, pady=2)\n            \n            # Status emoji\n            status_emoji = self._get_status_emoji(status)\n            \n            # Step label\n            step_label = ctk.CTkLabel(\n                step_frame,\n                text=f\"{status_emoji} {step_name}\",\n                font=ctk.CTkFont(size=12),\n                anchor=\"w\"\n            )\n            step_label.pack(side=\"left\", padx=(10, 5), pady=5)\n            \n            # Progress bar\n            progress_bar = ctk.CTkProgressBar(step_frame, width=150)\n            progress_bar.pack(side=\"right\", padx=(5, 10), pady=5)\n            progress_bar.set(progress / 100 if progress > 0 else 0)\n            \n            # Progress text\n            progress_text = ctk.CTkLabel(\n                step_frame,\n                text=f\"{progress:.1f}%\",\n                font=ctk.CTkFont(size=10)\n            )\n            progress_text.pack(side=\"right\", padx=(5, 5), pady=5)\n    \n    def _add_tk_progress_info(self, parent):\n        \"\"\"Add progress information to Tkinter frame\"\"\"\n        steps = self.session_data.get('steps', {})\n        step_names = {\n            'validate': '動画検証',\n            'extract': 'フレーム抽出',\n            'upscale': 'AI処理',\n            'combine': '動画結合'\n        }\n        \n        for i, (step_id, step_name) in enumerate(step_names.items()):\n            step_data = steps.get(step_id, {})\n            status = step_data.get('status', 'pending')\n            progress = step_data.get('progress', 0)\n            \n            # Status\n            status_text = self._get_status_text(status)\n            \n            # Row\n            row_frame = ttk.Frame(parent)\n            row_frame.pack(fill=\"x\", pady=2)\n            \n            ttk.Label(\n                row_frame,\n                text=f\"{step_name}:\",\n                width=12\n            ).pack(side=\"left\")\n            \n            ttk.Label(\n                row_frame,\n                text=status_text\n            ).pack(side=\"left\", padx=(5, 10))\n            \n            ttk.Label(\n                row_frame,\n                text=f\"{progress:.1f}%\"\n            ).pack(side=\"right\")\n    \n    def _add_ctk_info_row(self, parent, label: str, value: str):\n        \"\"\"Add information row to CustomTkinter frame\"\"\"\n        row_frame = ctk.CTkFrame(parent)\n        row_frame.pack(fill=\"x\", padx=5, pady=2)\n        \n        label_widget = ctk.CTkLabel(\n            row_frame,\n            text=label,\n            font=ctk.CTkFont(size=12, weight=\"bold\"),\n            width=100,\n            anchor=\"w\"\n        )\n        label_widget.pack(side=\"left\", padx=(10, 5), pady=5)\n        \n        value_widget = ctk.CTkLabel(\n            row_frame,\n            text=str(value),\n            font=ctk.CTkFont(size=12),\n            anchor=\"w\"\n        )\n        value_widget.pack(side=\"left\", padx=(5, 10), pady=5, fill=\"x\", expand=True)\n    \n    def _add_tk_info_row(self, parent, label: str, value: str):\n        \"\"\"Add information row to Tkinter frame\"\"\"\n        row_frame = ttk.Frame(parent)\n        row_frame.pack(fill=\"x\", pady=2)\n        \n        ttk.Label(\n            row_frame,\n            text=label,\n            font=(\"TkDefaultFont\", 9, \"bold\"),\n            width=12\n        ).pack(side=\"left\")\n        \n        ttk.Label(\n            row_frame,\n            text=str(value)\n        ).pack(side=\"left\", padx=(5, 0), fill=\"x\", expand=True)\n    \n    def _get_status_emoji(self, status: str) -> str:\n        \"\"\"Get emoji for step status\"\"\"\n        emoji_map = {\n            'pending': '⏳',\n            'in_progress': '🔄',\n            'completed': '✅',\n            'failed': '❌'\n        }\n        return emoji_map.get(status, '❓')\n    \n    def _get_status_text(self, status: str) -> str:\n        \"\"\"Get text for step status\"\"\"\n        status_map = {\n            'pending': '待機中',\n            'in_progress': '実行中',\n            'completed': '完了',\n            'failed': '失敗'\n        }\n        return status_map.get(status, '不明')\n    \n    def _on_choice(self, choice: str):\n        \"\"\"Handle user choice\"\"\"\n        self.result = choice\n        if self.dialog:\n            self.dialog.destroy()\n\nclass SessionSelectionDialog:\n    \"\"\"Dialog for selecting from multiple resumable sessions\"\"\"\n    \n    def __init__(self, parent, sessions: List[Dict[str, Any]]):\n        self.parent = parent\n        self.sessions = sessions\n        self.result = None\n        self.dialog = None\n    \n    def show(self) -> Optional[Dict[str, Any]]:\n        \"\"\"Show session selection dialog\"\"\"\n        if not self.sessions:\n            return None\n        \n        if len(self.sessions) == 1:\n            # Single session - show resume dialog directly\n            resume_dialog = ResumeDialog(self.parent, self.sessions[0])\n            choice = resume_dialog.show()\n            if choice == \"resume\":\n                return self.sessions[0]\n            return None\n        \n        # Multiple sessions - show selection dialog\n        if GUI_AVAILABLE and ctk:\n            return self._show_ctk_selection()\n        else:\n            return self._show_tk_selection()\n    \n    def _show_ctk_selection(self) -> Optional[Dict[str, Any]]:\n        \"\"\"Show CustomTkinter session selection dialog\"\"\"\n        self.dialog = ctk.CTkToplevel(self.parent)\n        self.dialog.title(\"セッション選択\")\n        self.dialog.geometry(\"700x400\")\n        \n        # Make modal\n        self.dialog.transient(self.parent)\n        self.dialog.grab_set()\n        \n        # Center on parent\n        self.parent.update_idletasks()\n        x = (self.parent.winfo_x() + self.parent.winfo_width() // 2) - 350\n        y = (self.parent.winfo_y() + self.parent.winfo_height() // 2) - 200\n        self.dialog.geometry(f\"+{x}+{y}\")\n        \n        # Title\n        title_label = ctk.CTkLabel(\n            self.dialog,\n            text=\"複数の未完了セッションが見つかりました\",\n            font=ctk.CTkFont(size=16, weight=\"bold\")\n        )\n        title_label.pack(pady=20)\n        \n        # Sessions list (simplified implementation)\n        # In a full implementation, you'd show a scrollable list\n        # For now, just select the most recent session\n        most_recent = max(self.sessions, key=lambda s: s.get('last_updated', ''))\n        \n        # Show resume dialog for most recent\n        self.dialog.destroy()\n        resume_dialog = ResumeDialog(self.parent, most_recent)\n        choice = resume_dialog.show()\n        if choice == \"resume\":\n            return most_recent\n        return None\n    \n    def _show_tk_selection(self) -> Optional[Dict[str, Any]]:\n        \"\"\"Show Tkinter session selection dialog\"\"\"\n        # Simplified implementation - select most recent\n        most_recent = max(self.sessions, key=lambda s: s.get('last_updated', ''))\n        resume_dialog = ResumeDialog(self.parent, most_recent)\n        choice = resume_dialog.show()\n        if choice == \"resume\":\n            return most_recent\n        return None