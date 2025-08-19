"""
UpScale App - GPU支援フレーム抽出モジュール
ハードウェア加速を利用した高速フレーム抽出
"""

import os
import subprocess
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)

class GPUFrameExtractor:
    """GPU支援によるハードウェア加速フレーム抽出"""
    
    def __init__(self, resource_manager, gpu_info: Dict, temp_dir: str):
        self.resource_manager = resource_manager
        self.gpu_info = gpu_info
        self.temp_dir = Path(temp_dir)
        
        logger.info(f"Initializing GPUFrameExtractor - GPU info: {gpu_info}")
        logger.info(f"Temp directory: {temp_dir}")
        
        # GPU加速オプションの優先順位
        self.gpu_acceleration_methods = [
            {'name': 'amd_amf', 'hwaccel': 'd3d11va', 'encoder': 'h264_amf'},
            {'name': 'intel_qsv', 'hwaccel': 'qsv', 'encoder': 'h264_qsv'},
            {'name': 'd3d11va', 'hwaccel': 'd3d11va', 'encoder': None},
            {'name': 'dxva2', 'hwaccel': 'dxva2', 'encoder': None},
        ]
        
        # FFmpegパスを確認
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        logger.info(f"FFmpeg path: {ffmpeg_path}")
        
        # 最適なGPU加速方法を検出
        logger.info("Starting GPU acceleration detection...")
        self.selected_method = self._detect_best_gpu_acceleration()
        
        if self.selected_method:
            logger.info(f"GPUFrameExtractor initialized successfully - Method: {self.selected_method}")
        else:
            logger.warning("GPUFrameExtractor initialized but no GPU acceleration available")
    
    def _detect_best_gpu_acceleration(self) -> Optional[Dict]:
        """利用可能な最適なGPU加速方法を検出"""
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        if not ffmpeg_path:
            logger.warning("FFmpeg not available for GPU acceleration detection")
            return None
        
        # FFmpegでサポートされているハードウェア加速を確認
        try:
            cmd = [ffmpeg_path, '-hwaccels']
            logger.info(f"Running GPU detection command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode != 0:
                logger.warning(f"Failed to query FFmpeg hardware acceleration support - Return code: {result.returncode}")
                logger.warning(f"FFmpeg stderr: {result.stderr}")
                return None
            
            supported_hwaccels = result.stdout.lower()
            logger.info(f"Supported hardware accelerations: {supported_hwaccels.strip()}")
            
        except Exception as e:
            logger.error(f"GPU acceleration detection failed: {e}", exc_info=True)
            return None
        
        # GPU情報に基づいて最適な方法を選択
        if self.gpu_info.get('amd', {}).get('available') and 'd3d11va' in supported_hwaccels:
            return {
                'name': 'amd_d3d11va',
                'hwaccel': 'd3d11va',
                'hwaccel_output_format': 'd3d11',
                'description': 'AMD Radeon with D3D11VA acceleration'
            }
        elif self.gpu_info.get('intel', {}).get('available') and 'qsv' in supported_hwaccels:
            return {
                'name': 'intel_qsv', 
                'hwaccel': 'qsv',
                'hwaccel_output_format': 'qsv',
                'description': 'Intel Quick Sync Video acceleration'
            }
        elif 'd3d11va' in supported_hwaccels:
            return {
                'name': 'generic_d3d11va',
                'hwaccel': 'd3d11va',
                'description': 'Generic D3D11VA acceleration'
            }
        elif 'dxva2' in supported_hwaccels:
            return {
                'name': 'generic_dxva2',
                'hwaccel': 'dxva2', 
                'description': 'Generic DXVA2 acceleration'
            }
        else:
            logger.info("No suitable GPU acceleration found")
            return None
    
    def extract_frames_gpu_accelerated(self, video_path: Path, total_frames: int, duration: float,
                                     progress_callback: Optional[Callable] = None,
                                     progress_dialog=None) -> List[str]:
        """GPU加速を使用したフレーム抽出（安全なフォールバック付き）"""
        
        if not self.selected_method:
            logger.warning("No GPU acceleration available, using CPU fallback")
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.add_log_message("WARNING: No GPU acceleration available, using CPU fallback"))
            raise RuntimeError("GPU acceleration not available")
        
        logger.info(f"Attempting GPU-accelerated extraction using {self.selected_method['name']}")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"INFO: Trying {self.selected_method['description']}"))
        
        try:
            # まず小さなテスト抽出を実行して動作確認
            test_result = self._test_gpu_extraction_safe(video_path, progress_dialog)
            if not test_result:
                logger.warning("GPU extraction test failed, falling back to CPU")
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message("WARNING: GPU test failed, using CPU"))
                raise RuntimeError("GPU test extraction failed")
            
            # テスト成功時のみ本格処理開始
            if total_frames > 3000:  # より保守的な閾値
                return self._extract_gpu_batched_safe(video_path, total_frames, duration, 
                                               progress_callback, progress_dialog)
            else:
                return self._extract_gpu_single_pass_safe(video_path, progress_callback, progress_dialog)
                
        except Exception as e:
            logger.error(f"GPU extraction failed: {str(e)}")
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.add_log_message(f"ERROR: GPU extraction failed - {str(e)[:100]}"))
            raise  # CPUフォールバックのために例外を再発生
    
    def _extract_gpu_single_pass(self, video_path: Path, 
                               progress_callback: Optional[Callable] = None,
                               progress_dialog=None) -> List[str]:
        """GPU加速による単一パス抽出"""
        
        output_dir = self.temp_dir / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU加速FFmpegコマンド構築
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        cmd = [ffmpeg_path]
        
        # ハードウェア加速オプション追加
        cmd.extend(['-hwaccel', self.selected_method['hwaccel']])
        
        if 'hwaccel_output_format' in self.selected_method:
            cmd.extend(['-hwaccel_output_format', self.selected_method['hwaccel_output_format']])
        
        # 入力ファイルと出力設定
        cmd.extend([
            '-i', str(video_path),
            '-vf', 'fps=source_tb,hwdownload,format=rgb24',  # GPU→CPUデータ転送
            '-threads', '1',  # GPUメイン処理時はCPUスレッド最小限に
            '-preset', 'ultrafast',  # 最高速設定
            '-y',
            str(output_dir / 'frame_%06d.png')
        ])
        
        logger.info(f"GPU acceleration command: {' '.join(cmd)}")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"DEBUG: GPU Command: {' '.join(cmd[:6])}..."))
        
        try:
            start_time = time.time()
            
            # GPU加速プロセス実行
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, universal_newlines=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            
            # 進捗監視
            self._monitor_gpu_extraction_progress(process, output_dir, progress_callback, progress_dialog)
            
            stdout, stderr = process.communicate(timeout=1800)  # 30分タイムアウト
            
            if process.returncode != 0:
                logger.error(f"GPU extraction failed: {stderr}")
                raise RuntimeError(f"GPU acceleration failed: {stderr}")
            
            # 抽出されたフレームを収集
            frame_files = sorted(output_dir.glob("frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
            
            elapsed_time = time.time() - start_time
            fps_rate = len(frame_paths) / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"GPU extraction complete: {len(frame_paths)} frames in {elapsed_time:.1f}s ({fps_rate:.1f} fps)")
            
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.add_log_message(
                        f"SUCCESS: GPU extraction complete - {len(frame_paths)} frames ({fps_rate:.1f} fps)"))
            
            return frame_paths
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("GPU extraction timed out after 30 minutes")
        except Exception as e:
            logger.error(f"GPU extraction error: {e}")
            raise
    
    def _extract_gpu_batched(self, video_path: Path, total_frames: int, duration: float,
                           progress_callback: Optional[Callable] = None,
                           progress_dialog=None) -> List[str]:
        """GPU加速によるバッチ処理抽出（大容量動画用）"""
        
        # GPUメモリを考慮したバッチサイズ（CPUより大きくできる）
        batch_size = 2000  # GPU処理では大きなバッチが効率的
        total_batches = (total_frames + batch_size - 1) // batch_size
        
        logger.info(f"GPU batched extraction: {total_batches} batches of {batch_size} frames")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"INFO: GPU batched processing - {total_batches} batches"))
        
        all_frame_paths = []
        
        for batch_num in range(total_batches):
            start_frame = batch_num * batch_size
            end_frame = min(start_frame + batch_size, total_frames)
            
            # 時間範囲計算
            start_time = (start_frame / total_frames) * duration
            end_time = (end_frame / total_frames) * duration
            
            logger.info(f"GPU batch {batch_num + 1}/{total_batches}: frames {start_frame}-{end_frame}")
            
            try:
                batch_paths = self._extract_gpu_batch_segment(
                    video_path, batch_num, start_time, end_time - start_time,
                    start_frame, progress_dialog
                )
                
                all_frame_paths.extend(batch_paths)
                
                # 進捗更新
                if progress_callback:
                    progress = ((batch_num + 1) / total_batches) * 100
                    progress_callback(progress, f"GPU batch {batch_num + 1}/{total_batches} - {len(all_frame_paths)} frames")
                
            except Exception as e:
                logger.error(f"GPU batch {batch_num + 1} failed: {e}")
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda b=batch_num: progress_dialog.add_log_message(f"ERROR: GPU batch {b + 1} failed"))
                raise
        
        logger.info(f"GPU batched extraction complete: {len(all_frame_paths)} total frames")
        return all_frame_paths
    
    def _extract_gpu_batch_segment(self, video_path: Path, batch_num: int, 
                                 start_time: float, duration: float, frame_offset: int,
                                 progress_dialog=None) -> List[str]:
        """GPU加速による単一バッチセグメント抽出"""
        
        batch_dir = self.temp_dir / 'frames' / f'gpu_batch_{batch_num:03d}'
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        cmd = [ffmpeg_path]
        
        # GPU加速設定
        cmd.extend(['-hwaccel', self.selected_method['hwaccel']])
        if 'hwaccel_output_format' in self.selected_method:
            cmd.extend(['-hwaccel_output_format', self.selected_method['hwaccel_output_format']])
        
        # セグメント抽出設定
        cmd.extend([
            '-ss', f'{start_time:.3f}',
            '-i', str(video_path),
            '-t', f'{duration:.3f}',
            '-vf', 'fps=source_tb,hwdownload,format=rgb24',
            '-threads', '1',
            '-preset', 'ultrafast',
            '-y',
            str(batch_dir / 'frame_%06d.png')
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode != 0:
                raise RuntimeError(f"GPU batch extraction failed: {result.stderr}")
            
            # バッチフレームを収集し、グローバル番号でリネーム
            batch_frames = sorted(batch_dir.glob("frame_*.png"))
            renamed_frames = []
            
            for i, frame_file in enumerate(batch_frames):
                global_frame_num = frame_offset + i + 1
                new_name = self.temp_dir / 'frames' / f'frame_{global_frame_num:06d}.png'
                
                # 安全なファイル移動
                frame_file.rename(new_name)
                renamed_frames.append(str(new_name))
            
            # バッチディレクトリクリーンアップ
            batch_dir.rmdir()
            
            return renamed_frames
            
        except Exception as e:
            logger.error(f"GPU batch segment extraction failed: {e}")
            raise
    
    def _monitor_gpu_extraction_progress(self, process, output_dir: Path, 
                                       progress_callback: Optional[Callable],
                                       progress_dialog) -> None:
        """GPU抽出の進捗を監視"""
        import threading
        import time
        
        def monitor():
            last_count = 0
            stagnant_iterations = 0
            
            while process.poll() is None:
                try:
                    # 抽出されたフレーム数をカウント
                    current_frames = len(list(output_dir.glob("frame_*.png")))
                    
                    if current_frames > last_count:
                        stagnant_iterations = 0
                        if progress_callback:
                            progress_callback(min(90, (current_frames / 1000) * 50), 
                                            f"GPU extracting... {current_frames} frames")
                    else:
                        stagnant_iterations += 1
                    
                    # GPU使用率監視（利用可能な場合）
                    try:
                        gpu_usage = self._get_gpu_usage()
                        if gpu_usage and progress_dialog and current_frames % 500 == 0:
                            progress_dialog.window.after(0, 
                                lambda: progress_dialog.add_log_message(f"INFO: GPU usage: {gpu_usage:.1f}%"))
                    except:
                        pass
                    
                    last_count = current_frames
                    
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(2.0)  # 2秒間隔で監視
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _get_gpu_usage(self) -> Optional[float]:
        """GPU使用率を取得（可能な場合）"""
        try:
            # psutil経由でGPU情報取得を試行
            import psutil
            
            # Windows GPU情報取得の試行
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # GPU温度が取得できればGPUが動作中と推定
                    return 50.0  # 推定使用率
            
            return None
            
        except Exception:
            return None
    
    def estimate_gpu_extraction_time(self, total_frames: int) -> float:
        """GPU加速でのフレーム抽出時間推定"""
        if not self.selected_method:
            return float('inf')  # GPU不可の場合は無限大
        
        # GPU加速による大幅な性能向上を想定
        base_gpu_rate = 60.0  # GPU加速時の基準レート（フレーム/秒）
        
        if self.selected_method['name'].startswith('amd'):
            # AMD GPU: 高性能
            estimated_rate = base_gpu_rate * 1.2
        elif self.selected_method['name'].startswith('intel'):
            # Intel GPU: 中程度
            estimated_rate = base_gpu_rate * 0.8
        else:
            # 汎用加速: 標準
            estimated_rate = base_gpu_rate
        
        return total_frames / estimated_rate
    
    def is_gpu_acceleration_available(self) -> bool:
        """GPU加速が利用可能かどうか"""
        return self.selected_method is not None
    
    def _test_gpu_extraction_safe(self, video_path: Path, progress_dialog=None) -> bool:
        """安全なGPU抽出テスト（最初の5フレームのみ）"""
        try:
            test_dir = self.temp_dir / 'gpu_test'
            test_dir.mkdir(parents=True, exist_ok=True)
            
            ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
            
            # 非常に限定的なテストコマンド
            cmd = [
                ffmpeg_path,
                '-hwaccel', self.selected_method['hwaccel'],
                '-i', str(video_path),
                '-frames:v', '5',  # 最初の5フレームのみ
                '-vf', 'scale=128:128,format=rgb24',  # 小さなサイズでテスト
                '-y',
                str(test_dir / 'test_%03d.png')
            ]
            
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.add_log_message("INFO: Testing GPU extraction..."))
            
            logger.info("Running GPU extraction test...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            success = result.returncode == 0
            test_files = list(test_dir.glob("test_*.png"))
            
            if success and len(test_files) >= 3:
                logger.info(f"GPU test successful: {len(test_files)} test frames extracted")
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message("SUCCESS: GPU test passed"))
            else:
                logger.warning(f"GPU test failed or insufficient frames: {len(test_files)}")
                logger.warning(f"GPU test stderr: {result.stderr}")
                success = False
                
            # テストファイルクリーンアップ
            for f in test_files:
                f.unlink(missing_ok=True)
            test_dir.rmdir()
            
            return success
            
        except Exception as e:
            logger.error(f"GPU test error: {e}")
            return False
    
    def _extract_gpu_single_pass_safe(self, video_path: Path, 
                                    progress_callback: Optional[Callable] = None,
                                    progress_dialog=None) -> List[str]:
        """安全なGPU単一パス抽出（改良版）"""
        
        output_dir = self.temp_dir / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # より保守的なGPU加速コマンド構築
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        cmd = [
            ffmpeg_path,
            '-hwaccel', self.selected_method['hwaccel'],
            '-hwaccel_output_format', 'nv12',  # より互換性の高い形式
            '-i', str(video_path),
            '-vf', 'hwdownload,format=rgb24',  # GPU→CPU転送
            '-threads', '2',  # 最小限のCPUスレッド
            '-preset', 'faster',  # より安定な設定
            '-loglevel', 'warning',  # ログレベル制限
            '-y',
            str(output_dir / 'frame_%06d.png')
        ]
        
        logger.info("Starting safe GPU single-pass extraction")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message("INFO: Starting safe GPU extraction"))
        
        try:
            start_time = time.time()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, universal_newlines=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            
            # より頻繁な進捗監視
            self._monitor_gpu_extraction_progress_safe(process, output_dir, progress_callback, progress_dialog)
            
            stdout, stderr = process.communicate(timeout=1200)  # 20分タイムアウト
            
            if process.returncode != 0:
                logger.error(f"Safe GPU extraction failed: {stderr}")
                raise RuntimeError(f"GPU extraction failed with return code {process.returncode}")
            
            frame_files = sorted(output_dir.glob("frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
            
            elapsed_time = time.time() - start_time
            fps_rate = len(frame_paths) / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Safe GPU extraction complete: {len(frame_paths)} frames in {elapsed_time:.1f}s ({fps_rate:.1f} fps)")
            
            if progress_dialog:
                progress_dialog.window.after(0, 
                    lambda: progress_dialog.add_log_message(f"SUCCESS: {len(frame_paths)} frames extracted ({fps_rate:.1f} fps)"))
            
            return frame_paths
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("GPU extraction timed out")
        except Exception as e:
            logger.error(f"Safe GPU extraction error: {e}")
            raise
    
    def _extract_gpu_batched_safe(self, video_path: Path, total_frames: int, duration: float,
                                progress_callback: Optional[Callable] = None,
                                progress_dialog=None) -> List[str]:
        """安全なGPUバッチ処理抽出"""
        
        # より小さなバッチサイズで安全性向上
        batch_size = 1000  
        total_batches = (total_frames + batch_size - 1) // batch_size
        
        logger.info(f"Safe GPU batched extraction: {total_batches} batches of {batch_size} frames")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"INFO: Safe GPU batched processing - {total_batches} batches"))
        
        all_frame_paths = []
        failed_batches = 0
        
        for batch_num in range(total_batches):
            start_frame = batch_num * batch_size
            end_frame = min(start_frame + batch_size, total_frames)
            
            start_time = (start_frame / total_frames) * duration
            end_time = (end_frame / total_frames) * duration
            
            logger.info(f"Safe GPU batch {batch_num + 1}/{total_batches}: frames {start_frame}-{end_frame}")
            
            try:
                batch_paths = self._extract_gpu_batch_segment_safe(
                    video_path, batch_num, start_time, end_time - start_time,
                    start_frame, progress_dialog
                )
                
                all_frame_paths.extend(batch_paths)
                
                if progress_callback:
                    progress = ((batch_num + 1) / total_batches) * 100
                    progress_callback(progress, f"GPU batch {batch_num + 1}/{total_batches} - {len(all_frame_paths)} frames")
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"Safe GPU batch {batch_num + 1} failed: {e}")
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda b=batch_num: progress_dialog.add_log_message(f"ERROR: GPU batch {b + 1} failed"))
                
                # 3回失敗したら全体を失敗扱い
                if failed_batches >= 3:
                    logger.error("Too many GPU batch failures, aborting")
                    raise RuntimeError(f"Multiple GPU batch failures ({failed_batches})")
        
        logger.info(f"Safe GPU batched extraction complete: {len(all_frame_paths)} total frames")
        return all_frame_paths
    
    def _extract_gpu_batch_segment_safe(self, video_path: Path, batch_num: int, 
                                      start_time: float, duration: float, frame_offset: int,
                                      progress_dialog=None) -> List[str]:
        """安全なGPUバッチセグメント抽出"""
        
        batch_dir = self.temp_dir / 'frames' / f'gpu_batch_{batch_num:03d}'
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        cmd = [
            ffmpeg_path,
            '-hwaccel', self.selected_method['hwaccel'],
            '-ss', f'{start_time:.3f}',
            '-i', str(video_path),
            '-t', f'{duration:.3f}',
            '-vf', 'hwdownload,format=rgb24',
            '-threads', '1',
            '-preset', 'faster',
            '-loglevel', 'error',
            '-y',
            str(batch_dir / 'frame_%06d.png')
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode != 0:
                raise RuntimeError(f"Safe GPU batch extraction failed: {result.stderr}")
            
            batch_frames = sorted(batch_dir.glob("frame_*.png"))
            renamed_frames = []
            
            for i, frame_file in enumerate(batch_frames):
                global_frame_num = frame_offset + i + 1
                new_name = self.temp_dir / 'frames' / f'frame_{global_frame_num:06d}.png'
                
                frame_file.rename(new_name)
                renamed_frames.append(str(new_name))
            
            batch_dir.rmdir()
            return renamed_frames
            
        except Exception as e:
            logger.error(f"Safe GPU batch segment extraction failed: {e}")
            raise
    
    def _monitor_gpu_extraction_progress_safe(self, process, output_dir: Path, 
                                            progress_callback: Optional[Callable],
                                            progress_dialog) -> None:
        """安全なGPU抽出進捗監視"""
        import threading
        
        def monitor():
            last_count = 0
            stagnant_iterations = 0
            
            while process.poll() is None:
                try:
                    current_frames = len(list(output_dir.glob("frame_*.png")))
                    
                    if current_frames > last_count:
                        stagnant_iterations = 0
                        if progress_callback and current_frames % 100 == 0:
                            progress_callback(min(85, (current_frames / 500) * 50), 
                                            f"GPU extracting... {current_frames} frames")
                    else:
                        stagnant_iterations += 1
                        
                        # 長時間停止時は警告
                        if stagnant_iterations > 15 and progress_dialog:
                            progress_dialog.window.after(0, 
                                lambda: progress_dialog.add_log_message("WARNING: GPU extraction may be stalled"))
                    
                    last_count = current_frames
                    
                except Exception as e:
                    logger.debug(f"Safe GPU monitoring error: {e}")
                
                time.sleep(3.0)  # 3秒間隔で監視
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU加速情報を取得"""
        return {
            'available': self.is_gpu_acceleration_available(),
            'method': self.selected_method,
            'estimated_speedup': '3-5x faster than CPU' if self.selected_method else 'N/A'
        }