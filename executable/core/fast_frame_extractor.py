"""
UpScale App - 高速フレーム抽出モジュール
並列FFmpeg処理とマルチスレッド最適化による大幅な性能向上
"""

import os
import subprocess
import concurrent.futures
import threading
import time
import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import psutil
import multiprocessing
from .gpu_frame_extractor import GPUFrameExtractor

logger = logging.getLogger(__name__)

class FastFrameExtractor:
    """高速並列フレーム抽出クラス"""
    
    def __init__(self, resource_manager, temp_dir: str, gpu_info: Dict = None):
        self.resource_manager = resource_manager
        self.temp_dir = Path(temp_dir)
        
        # GPU支援フレーム抽出の初期化
        self.gpu_info = gpu_info or {}
        try:
            self.gpu_extractor = GPUFrameExtractor(resource_manager, self.gpu_info, temp_dir)
        except Exception as e:
            logger.warning(f"GPU extractor initialization failed: {e}")
            self.gpu_extractor = None
        
        # 並列処理設定 - CPU負荷を抑制
        cpu_count = multiprocessing.cpu_count()
        # CPU使用率が高い場合は並列度を抑制 (最大2ワーカー)
        self.max_workers = min(cpu_count // 2, 2)  # CPU負荷を50%に制限
        self.optimal_batch_size = 300  # バッチサイズを削減してメモリ使用量を抑制
        
        # CPU使用率監視とダイナミック調整
        self.cpu_threshold = 85.0  # CPU使用率閾値
        self.adaptive_processing = True
        
        logger.info(f"FastFrameExtractor initialized - Workers: {self.max_workers}/{cpu_count}, Batch size: {self.optimal_batch_size}")
        logger.info(f"CPU-conscious mode enabled - Max CPU target: {100 - (100/cpu_count * (cpu_count - self.max_workers)):.0f}%")
        
        # GPU加速の可用性をログに記録
        if self.gpu_extractor and self.gpu_extractor.is_gpu_acceleration_available():
            logger.info(f"GPU acceleration available: {self.gpu_extractor.selected_method['description']}")
        else:
            logger.info("GPU acceleration not available, using CPU-only processing")
    
    def extract_frames_parallel(self, video_path: Path, total_frames: int, duration: float,
                              progress_callback: Optional[Callable] = None,
                              progress_dialog=None) -> List[str]:
        """GPU優先の高速フレーム抽出"""
        
        # 詳細ログで処理開始を記録
        logger.info(f"Frame extraction started - Video: {video_path}, Frames: {total_frames}, Duration: {duration:.1f}s")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"INFO: Starting extraction - {total_frames} frames"))
        
        # GPU加速が利用可能な場合は優先使用
        if self.gpu_extractor and self.gpu_extractor.is_gpu_acceleration_available():
            try:
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message("INFO: Attempting GPU-accelerated frame extraction"))
                
                logger.info(f"Using GPU acceleration: {self.gpu_extractor.selected_method}")
                
                # GPU加速による抽出を実行
                frame_paths = self.gpu_extractor.extract_frames_gpu_accelerated(
                    video_path, total_frames, duration, progress_callback, progress_dialog
                )
                
                # GPU処理成功時
                logger.info(f"GPU extraction successful: {len(frame_paths)} frames extracted")
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message(f"SUCCESS: GPU extraction completed - {len(frame_paths)} frames"))
                
                return frame_paths
                
            except Exception as e:
                error_msg = str(e)[:200]  # エラーメッセージを制限
                logger.error(f"GPU extraction failed: {error_msg}")
                logger.info("Detailed GPU error for debugging:", exc_info=True)
                
                if progress_dialog:
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message(f"ERROR: GPU extraction failed"))
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message(f"ERROR DETAIL: {error_msg}"))
                    progress_dialog.window.after(0, 
                        lambda: progress_dialog.add_log_message("INFO: Switching to CPU fallback processing"))
        
        # CPU処理へのフォールバック（最適化戦略の選択）
        logger.info("Using CPU-only processing for frame extraction")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message("INFO: Using CPU-only processing"))
        
        if total_frames > 10000:
            return self._extract_large_video_optimized(
                video_path, total_frames, duration, progress_callback, progress_dialog
            )
        elif total_frames > 1000:
            return self._extract_medium_video_parallel(
                video_path, total_frames, duration, progress_callback, progress_dialog
            )
        else:
            return self._extract_small_video_fast(
                video_path, total_frames, duration, progress_callback, progress_dialog
            )
    
    def _extract_large_video_optimized(self, video_path: Path, total_frames: int, duration: float,
                                     progress_callback: Optional[Callable] = None,
                                     progress_dialog=None) -> List[str]:
        """大容量動画用の最適化並列処理"""
        
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message("INFO: Using optimized large video extraction"))
        
        logger.info(f"Starting optimized large video extraction: {total_frames} frames")
        
        # 動的バッチサイズ計算
        optimal_batch_size = max(250, min(1000, total_frames // (self.max_workers * 4)))
        total_batches = (total_frames + optimal_batch_size - 1) // optimal_batch_size
        
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(
                    f"INFO: Optimized strategy - {total_batches} batches of {optimal_batch_size} frames, {self.max_workers} parallel workers"
                ))
        
        logger.info(f"Optimized batch strategy: {total_batches} batches, {optimal_batch_size} frames each, {self.max_workers} workers")
        
        # 並列バッチ処理実行
        return self._execute_parallel_batches(
            video_path, total_frames, duration, optimal_batch_size, 
            progress_callback, progress_dialog
        )
    
    def _extract_medium_video_parallel(self, video_path: Path, total_frames: int, duration: float,
                                     progress_callback: Optional[Callable] = None,
                                     progress_dialog=None) -> List[str]:
        """中容量動画用の並列処理"""
        
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message("INFO: Using medium video parallel extraction"))
        
        logger.info(f"Starting medium video parallel extraction: {total_frames} frames")
        
        # 中サイズ動画に最適化されたバッチサイズ
        batch_size = 300
        total_batches = (total_frames + batch_size - 1) // batch_size
        
        return self._execute_parallel_batches(
            video_path, total_frames, duration, batch_size, 
            progress_callback, progress_dialog
        )
    
    def _extract_small_video_fast(self, video_path: Path, total_frames: int, duration: float,
                                progress_callback: Optional[Callable] = None,
                                progress_dialog=None) -> List[str]:
        """小容量動画用の高速単一処理"""
        
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message("INFO: Using fast single-pass extraction"))
        
        logger.info(f"Starting fast single-pass extraction: {total_frames} frames")
        
        return self._execute_single_pass_optimized(
            video_path, progress_callback, progress_dialog
        )
    
    def _execute_parallel_batches(self, video_path: Path, total_frames: int, duration: float,
                                batch_size: int, progress_callback: Optional[Callable] = None,
                                progress_dialog=None) -> List[str]:
        """並列バッチ実行エンジン"""
        
        total_batches = (total_frames + batch_size - 1) // batch_size
        all_frame_paths = []
        completed_batches = 0
        lock = threading.Lock()
        
        # 並列実行用のバッチリスト作成
        batch_tasks = []
        for batch_num in range(total_batches):
            start_frame = batch_num * batch_size
            end_frame = min(start_frame + batch_size - 1, total_frames - 1)
            
            start_time = (start_frame / total_frames) * duration
            end_time = (end_frame / total_frames) * duration
            
            batch_tasks.append({
                'batch_num': batch_num,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'expected_frames': end_frame - start_frame + 1
            })
        
        def process_batch(batch_info):
            """単一バッチの並列処理"""
            batch_num = batch_info['batch_num']
            start_time = batch_info['start_time']
            end_time = batch_info['end_time']
            expected_frames = batch_info['expected_frames']
            
            try:
                # バッチ専用ディレクトリ
                batch_dir = self.temp_dir / 'frames' / f'batch_{batch_num:03d}'
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                # CPU負荷を抑制したFFmpegコマンド
                ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
                
                # CPU使用率に基づいてスレッド数を動的調整
                cpu_usage = psutil.cpu_percent(interval=0.1)
                if cpu_usage > 90:
                    thread_count = '1'  # CPU使用率が非常に高い場合
                elif cpu_usage > 70:
                    thread_count = '2'  # CPU使用率が高い場合
                else:
                    thread_count = '3'  # 通常時
                
                cmd = [
                    ffmpeg_path,
                    '-ss', f'{start_time:.3f}',  # 開始時間（高精度）
                    '-i', str(video_path),
                    '-t', f'{end_time - start_time:.3f}',  # 処理時間
                    # フレームレート維持（自動検出）
                    '-threads', thread_count,  # 動的スレッド数制限
                    '-preset', 'fast',  # CPU負荷軽減
                    '-y',  # 上書き許可
                    str(batch_dir / 'frame_%06d.png')
                ]
                
                # CPU負荷軽減のための処理前の小休憩
                if cpu_usage > 85:
                    time.sleep(0.2)  # 高CPU使用率時は200ms休憩
                elif cpu_usage > 70:
                    time.sleep(0.1)  # 中程度の場合は100ms休憩
                
                # FFmpeg実行（短時間タイムアウト）
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    shell=False,
                    encoding='utf-8'
                )
                
                if result.returncode != 0:
                    logger.error(f"Batch {batch_num + 1} failed: {result.stderr}")
                    return []
                
                # 抽出フレーム収集
                batch_frames = sorted(batch_dir.glob("frame_*.png"))
                frame_paths = [str(f) for f in batch_frames]
                
                # 進捗更新（スレッドセーフ）
                nonlocal completed_batches
                with lock:
                    completed_batches += 1
                    if progress_callback:
                        progress = (completed_batches / total_batches) * 100
                        progress_callback(progress, 
                            f"Parallel batch {completed_batches}/{total_batches} - {len(frame_paths)} frames")
                
                logger.info(f"Batch {batch_num + 1} complete: {len(frame_paths)} frames")
                return frame_paths
                
            except Exception as e:
                logger.error(f"Batch {batch_num + 1} error: {e}")
                return []
        
        # CPU使用率に基づく動的ワーカー調整
        current_workers = self._get_adaptive_worker_count()
        
        # 並列バッチ実行
        logger.info(f"Starting {total_batches} parallel batches with {current_workers} workers (adaptive)")
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(f"INFO: CPU-adaptive processing - Using {current_workers} workers"))
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
            # 全バッチを並列実行
            future_to_batch = {
                executor.submit(process_batch, batch_info): batch_info 
                for batch_info in batch_tasks
            }
            
            # CPU監視開始
            if self.adaptive_processing:
                self._monitor_cpu_during_processing(executor, future_to_batch, progress_dialog)
            
            # 結果を順序通りに収集
            batch_results = [None] * total_batches
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_info = future_to_batch[future]
                batch_num = batch_info['batch_num']
                
                try:
                    batch_frames = future.result()
                    batch_results[batch_num] = batch_frames
                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} exception: {e}")
                    batch_results[batch_num] = []
        
        # フレームパスを順序通りに結合
        for batch_frames in batch_results:
            if batch_frames:
                all_frame_paths.extend(batch_frames)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel extraction complete: {len(all_frame_paths)} frames in {elapsed_time:.1f}s")
        
        if progress_dialog:
            progress_dialog.window.after(0, 
                lambda: progress_dialog.add_log_message(
                    f"SUCCESS: Parallel extraction complete - {len(all_frame_paths)} frames in {elapsed_time:.1f}s"
                ))
        
        return all_frame_paths
    
    def _execute_single_pass_optimized(self, video_path: Path, 
                                     progress_callback: Optional[Callable] = None,
                                     progress_dialog=None) -> List[str]:
        """最適化された単一パス抽出"""
        
        output_dir = self.temp_dir / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = self.resource_manager.get_binary_path('ffmpeg')
        
        # 高速単一パス抽出コマンド
        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            # フレームレート自動検出
            '-threads', str(min(4, multiprocessing.cpu_count())),  # 最大スレッド活用
            '-preset', 'ultrafast',  # 最高速設定
            '-y',
            str(output_dir / 'frame_%06d.png')
        ]
        
        logger.info(f"Starting optimized single-pass extraction")
        start_time = time.time()
        
        # プロセス実行
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW,
            shell=False,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Single-pass extraction failed: {result.stderr}")
        
        # フレーム収集
        frame_files = sorted(output_dir.glob("frame_*.png"))
        frame_paths = [str(f) for f in frame_files]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Single-pass extraction complete: {len(frame_paths)} frames in {elapsed_time:.1f}s")
        
        if progress_callback:
            progress_callback(100, f"Extracted {len(frame_paths)} frames")
        
        return frame_paths
    
    def _get_adaptive_worker_count(self) -> int:
        """動的にワーカー数を調整してCPU負荷を制御"""
        try:
            # 現在のCPU使用率をチェック
            current_cpu = psutil.cpu_percent(interval=1.0)
            
            if current_cpu > self.cpu_threshold:
                # CPU使用率が高い場合は1ワーカーに制限
                adaptive_workers = 1
                logger.info(f"High CPU usage detected ({current_cpu:.1f}%) - Reducing to 1 worker")
            elif current_cpu > (self.cpu_threshold - 15):
                # 中程度のCPU使用率の場合は少し制限
                adaptive_workers = min(2, self.max_workers)
                logger.info(f"Moderate CPU usage ({current_cpu:.1f}%) - Using {adaptive_workers} workers")
            else:
                # CPU使用率が低い場合は最大ワーカー数を使用
                adaptive_workers = self.max_workers
                logger.info(f"Low CPU usage ({current_cpu:.1f}%) - Using {adaptive_workers} workers")
            
            return adaptive_workers
            
        except Exception as e:
            logger.warning(f"CPU monitoring failed, using default workers: {e}")
            return self.max_workers
    
    def _monitor_cpu_during_processing(self, executor, future_to_batch, progress_dialog):
        """処理中のCPU監視とワーカー調整"""
        try:
            import threading
            
            def cpu_monitor():
                while not all(future.done() for future in future_to_batch.keys()):
                    try:
                        cpu_usage = psutil.cpu_percent(interval=2.0)
                        
                        if cpu_usage > self.cpu_threshold and progress_dialog:
                            progress_dialog.window.after(0, 
                                lambda: progress_dialog.add_log_message(f"WARNING: High CPU usage detected: {cpu_usage:.1f}%"))
                        
                        # CPU使用率が非常に高い場合は小さな休憩を入れる
                        if cpu_usage > 95.0:
                            time.sleep(0.5)  # 短い休憩
                            
                    except Exception as e:
                        logger.debug(f"CPU monitoring error: {e}")
                    
                    time.sleep(3.0)  # 3秒間隔で監視
            
            # CPU監視を別スレッドで実行
            monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            logger.warning(f"CPU monitoring setup failed: {e}")
    
    def estimate_extraction_time(self, total_frames: int) -> float:
        """GPU/CPU統合での抽出時間推定"""
        
        # GPU加速が利用可能な場合は大幅に短縮
        if self.gpu_extractor and self.gpu_extractor.is_gpu_acceleration_available():
            gpu_time = self.gpu_extractor.estimate_gpu_extraction_time(total_frames)
            logger.info(f"GPU extraction estimated: {gpu_time/60:.1f} minutes ({gpu_time:.1f} seconds)")
            return gpu_time
        
        # CPU処理の場合（従来の推定）
        base_rate = 8.0  # フレーム/秒（CPU制約下での現実的なレート）
        
        if total_frames > 10000:
            # 大容量: 並列処理効果があるがCPU制約あり
            estimated_rate = base_rate * min(2, self.max_workers)  # 並列効果
            cpu_time = total_frames / estimated_rate
        elif total_frames > 1000:
            # 中容量: 限定的な並列効果
            estimated_rate = base_rate * 1.5
            cpu_time = total_frames / estimated_rate
        else:
            # 小容量: 単一処理
            cpu_time = total_frames / base_rate
        
        logger.info(f"CPU extraction estimated: {cpu_time/60:.1f} minutes ({cpu_time:.1f} seconds)")
        return cpu_time