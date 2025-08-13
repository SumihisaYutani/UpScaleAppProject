#!/usr/bin/env python3
"""
Enhanced UpScale App CLI - AI Video Upscaling Tool
Enhanced Command Line Interface with improved error handling and performance monitoring
"""

import click
import logging
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from enhanced_upscale_app import EnhancedUpScaleApp, ProcessingError
from modules.performance_monitor import PerformanceMonitor
from config.settings import VIDEO_SETTINGS, AI_SETTINGS


class EnhancedProgressBar:
    """Enhanced progress bar with system monitoring"""
    
    def __init__(self, show_system_stats: bool = False):
        self.pbar = None
        self.last_message = ""
        self.show_system_stats = show_system_stats
        self.monitor = PerformanceMonitor() if show_system_stats else None
        
        if self.monitor:
            self.monitor.start_monitoring()
    
    def update(self, progress: float, message: str):
        """Update progress bar with optional system stats"""
        if self.pbar is None:
            total = 100
            if self.show_system_stats:
                self.pbar = tqdm(total=total, desc=message, unit="%", 
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total} [{elapsed}<{remaining}, {rate_fmt}]')
            else:
                self.pbar = tqdm(total=total, desc=message, unit="%")
        
        # Update progress
        current_progress = int(progress)
        if hasattr(self.pbar, 'n'):
            delta = current_progress - self.pbar.n
            if delta > 0:
                self.pbar.update(delta)
        
        # Update description
        if message != self.last_message:
            desc = message
            if self.show_system_stats and self.monitor:
                try:
                    stats = self.monitor.get_current_stats()
                    system_info = stats.get("system", {})
                    desc += f" [CPU: {system_info.get('cpu_percent', 0):.1f}% RAM: {system_info.get('memory_percent', 0):.1f}%]"
                except:
                    pass  # Ignore monitoring errors
            
            self.pbar.set_description(desc)
            self.last_message = message
        
        # Close if complete
        if progress >= 100:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
                if self.monitor:
                    self.monitor.stop_monitoring()


@click.group()
@click.version_option(version="0.2.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(verbose, debug):
    """🎬 Enhanced UpScale App - AI Video Upscaling Tool with Advanced Features"""
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.getLogger().setLevel(log_level)
    
    click.echo("🎬 Enhanced UpScale App - AI Video Upscaling Tool v0.2.0")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--scale', '-s', default=1.5, type=float, help='Scale factor (default: 1.5)')
@click.option('--no-ai', is_flag=True, help='Use simple upscaling instead of AI')
@click.option('--no-enhanced-ai', is_flag=True, help='Use basic AI instead of enhanced AI')
@click.option('--no-cleanup', is_flag=True, help='Keep temporary files')
@click.option('--no-monitoring', is_flag=True, help='Disable performance monitoring')
@click.option('--show-system-stats', is_flag=True, help='Show system stats during processing')
@click.option('--quality-preset', type=click.Choice(['fast', 'balanced', 'quality']), 
              default='balanced', help='Quality preset for AI processing')
@click.option('--max-retries', default=3, type=int, help='Maximum retry attempts on errors')
def upscale(input_file, output, scale, no_ai, no_enhanced_ai, no_cleanup, 
           no_monitoring, show_system_stats, quality_preset, max_retries):
    """Enhanced video upscaling with comprehensive error handling"""
    
    click.echo(f"📹 Processing: {input_file}")
    click.echo(f"🔍 Scale factor: {scale}x")
    click.echo(f"⚙️  Quality preset: {quality_preset}")
    
    # Quality settings based on preset
    quality_settings = {
        'fast': {'strength': 0.2, 'num_inference_steps': 10},
        'balanced': {'strength': 0.3, 'num_inference_steps': 20},
        'quality': {'strength': 0.4, 'num_inference_steps': 30}
    }
    
    # Initialize app with enhanced features
    try:
        app = EnhancedUpScaleApp(
            use_ai=not no_ai, 
            use_enhanced_ai=not no_enhanced_ai,
            temp_cleanup=not no_cleanup,
            enable_monitoring=not no_monitoring
        )
        app.max_recovery_attempts = max_retries
        
    except Exception as e:
        click.echo(f"❌ Failed to initialize application: {e}", err=True)
        sys.exit(1)
    
    # Show system info
    system_info = app.get_system_info_enhanced()
    if system_info.get("cuda_available", False):
        gpu_name = system_info.get('cuda_device_name', 'Unknown GPU')
        gpu_memory = system_info.get('cuda_memory_gb', 0)
        click.echo(f"🚀 GPU acceleration available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        click.echo("⚠️  Using CPU processing (will be slower)")
    
    if system_info.get("enhanced_ai_enabled", False):
        click.echo("🧠 Enhanced AI processing enabled")
    
    # Setup enhanced progress tracking
    progress_bar = EnhancedProgressBar(show_system_stats=show_system_stats)
    
    try:
        # Process video with enhanced error handling
        result = app.process_video_enhanced(
            input_file, 
            output, 
            scale, 
            progress_callback=progress_bar.update,
            quality_settings=quality_settings.get(quality_preset, quality_settings['balanced'])
        )
        
        if result["success"]:
            click.echo(f"✅ Success! Output saved to: {result['output_path']}")
            
            # Show enhanced statistics
            stats = result["stats"]
            click.echo("📊 Processing Statistics:")
            click.echo(f"  • Frames processed: {stats.get('processed_frame_count', 'N/A')}/{stats.get('frame_count', 'N/A')}")
            click.echo(f"  • Original size: {stats.get('original_size', 0) / (1024*1024):.1f} MB")
            click.echo(f"  • Output size: {stats.get('output_size', 0) / (1024*1024):.1f} MB")
            click.echo(f"  • Resolution: {stats.get('original_resolution')} → {stats.get('target_resolution')}")
            click.echo(f"  • Session ID: {result.get('session_id', 'N/A')}")
            
            # Performance statistics
            if "performance" in result and result["performance"]:
                perf = result["performance"]
                if perf.get("tasks_analyzed", 0) > 0:
                    click.echo("⚡ Performance Stats:")
                    click.echo(f"  • Average FPS: {perf.get('average_fps', 0):.2f}")
                    click.echo(f"  • Success rate: {perf.get('average_success_rate', 0):.1f}%")
                    click.echo(f"  • Total processing time: {perf.get('total_processing_time', 0):.1f}s")
            
            # Processing stages
            if "processing_stages" in result:
                click.echo("🔄 Processing Stages:")
                for stage in result["processing_stages"]:
                    click.echo(f"  • {stage['stage']}: {stage['status']}")
            
            # Warnings
            if stats.get("resource_warning"):
                click.echo("⚠️  Resource warning: Processing completed despite resource constraints")
            
        else:
            error_code = result.get("error_code", "UNKNOWN")
            error_msg = result.get("error", "Unknown error")
            
            click.echo(f"❌ Error ({error_code}): {error_msg}", err=True)
            
            # Show error details if available
            if "error_details" in result and result["error_details"]:
                details = result["error_details"]
                click.echo("🔍 Error Details:")
                click.echo(f"  • Operation: {details.get('operation', 'N/A')}")
                click.echo(f"  • Error Type: {details.get('error_type', 'N/A')}")
                click.echo(f"  • Processing Time: {details.get('processing_time', 0):.1f}s")
            
            # Show recovery information
            if system_info.get("recovery_attempts", 0) > 0:
                click.echo(f"🔄 Recovery attempts made: {system_info['recovery_attempts']}")
            
            # Save error log
            if "session_id" in result:
                click.echo(f"📝 Detailed logs saved for session: {result['session_id']}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        click.echo("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"💥 Unexpected error: {e}", err=True)
        sys.exit(1)
    finally:
        # Cleanup
        try:
            app.cleanup()
        except:
            pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output preview file path')
@click.option('--frames', default=30, type=int, help='Number of frames for preview')
def preview(input_file, output, frames):
    """Create an enhanced preview with system monitoring"""
    
    click.echo(f"🎬 Creating preview for: {input_file} ({frames} frames)")
    
    try:
        app = EnhancedUpScaleApp(use_ai=False, enable_monitoring=True)
        
        # Create preview (implementation would need to be added to enhanced app)
        click.echo("⚠️  Preview functionality available in basic mode only")
        click.echo("💡 Use: python main.py preview for basic preview functionality")
        
    except Exception as e:
        click.echo(f"💥 Error creating preview: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def info(input_file):
    """Show comprehensive information about a video file"""
    
    click.echo(f"📹 Video Information: {input_file}")
    
    try:
        app = EnhancedUpScaleApp(use_ai=False, enable_monitoring=False)
        validation = app.video_processor.validate_video_file(input_file)
        
        if validation["valid"]:
            info = validation["info"]
            click.echo("✅ Valid video file")
            click.echo(f"  • Duration: {info['duration']:.2f} seconds")
            click.echo(f"  • Resolution: {info['width']}x{info['height']}")
            click.echo(f"  • Codec: {info['codec_name']}")
            click.echo(f"  • Frame rate: {info['frame_rate']:.2f} fps")
            click.echo(f"  • Frame count: {info['frame_count']}")
            click.echo(f"  • File size: {info['size'] / (1024*1024):.1f} MB")
            
            # Enhanced estimates
            memory_estimates = app._estimate_memory_requirements(info, 1.5)
            click.echo("💾 Memory Estimates (for 1.5x upscale):")
            click.echo(f"  • Temp storage needed: {memory_estimates['temp_storage_gb']:.2f} GB")
            click.echo(f"  • Recommended RAM: {memory_estimates['recommended_ram_gb']:.1f} GB")
            click.echo(f"  • Peak memory usage: {memory_estimates['peak_memory_gb']:.2f} GB")
            
            # Resource availability check
            resource_ok = app._check_resource_availability(memory_estimates)
            status = "✅ Sufficient" if resource_ok else "⚠️  Limited"
            click.echo(f"  • Resource availability: {status}")
            
            # Processing time estimates
            time_est = app.video_processor.estimate_processing_time(info)
            click.echo("⏱️  Processing time estimates:")
            click.echo(f"  • With GPU: {time_est['estimated_gpu_minutes']:.1f} minutes")
            click.echo(f"  • With CPU: {time_est['estimated_cpu_minutes']:.1f} minutes")
            
        else:
            click.echo(f"❌ Invalid video file: {validation['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"💥 Error analyzing file: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--save-report', is_flag=True, help='Save system report to file')
def system(save_report):
    """Show comprehensive system information"""
    
    click.echo("🖥️  Enhanced System Information:")
    
    try:
        app = EnhancedUpScaleApp(use_ai=True, enable_monitoring=True)
        info = app.get_system_info_enhanced()
        
        click.echo(f"  • Platform: {info.get('platform', 'Unknown')}")
        click.echo(f"  • Python: {info.get('python_version', 'Unknown')}")
        click.echo(f"  • Session ID: {info.get('session_id', 'N/A')}")
        click.echo(f"  • Uptime: {info.get('session_uptime', 0):.1f}s")
        
        # CUDA information
        cuda_available = info.get('cuda_available', False)
        click.echo(f"  • CUDA available: {'Yes' if cuda_available else 'No'}")
        
        if cuda_available:
            click.echo(f"  • GPU: {info.get('cuda_device_name', 'Unknown')}")
            click.echo(f"  • VRAM: {info.get('cuda_memory_gb', 0):.1f} GB")
        
        # Enhanced features
        click.echo(f"  • Enhanced AI: {'Enabled' if info.get('enhanced_ai_enabled') else 'Disabled'}")
        click.echo(f"  • Monitoring: {'Enabled' if info.get('monitoring_enabled') else 'Disabled'}")
        
        # Performance stats (if available)
        if "performance_stats" in info:
            perf_stats = info["performance_stats"]
            system_perf = perf_stats.get("system", {})
            click.echo("📈 Current Performance:")
            click.echo(f"  • CPU: {system_perf.get('cpu_percent', 0):.1f}%")
            click.echo(f"  • Memory: {system_perf.get('memory_percent', 0):.1f}% ({system_perf.get('memory_used_gb', 0):.1f}GB)")
            if system_perf.get('gpu_memory_used_gb', 0) > 0:
                click.echo(f"  • GPU Memory: {system_perf.get('gpu_memory_used_gb', 0):.1f}GB")
        
        # Error history
        error_count = info.get('error_count', 0)
        if error_count > 0:
            click.echo(f"  • Error count: {error_count}")
            click.echo(f"  • Recovery attempts: {info.get('recovery_attempts', 0)}")
        
        # Directory information
        click.echo(f"  • Temp directory: {info.get('temp_dir', 'N/A')}")
        click.echo(f"  • Output directory: {info.get('output_dir', 'N/A')}")
        
        # Save report if requested
        if save_report:
            report_file = f"system_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            click.echo(f"📄 System report saved to: {report_file}")
        
    except Exception as e:
        click.echo(f"💥 Error getting system information: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--show-ai', is_flag=True, help='Show AI model settings')
def config(show_ai):
    """Show enhanced configuration"""
    
    click.echo("⚙️  Enhanced Configuration:")
    
    click.echo("📹 Video Settings:")
    for key, value in VIDEO_SETTINGS.items():
        click.echo(f"  • {key}: {value}")
    
    if show_ai:
        click.echo("🤖 AI Settings:")
        for key, value in AI_SETTINGS.items():
            click.echo(f"  • {key}: {value}")
    
    # Show available quality presets
    click.echo("🎨 Quality Presets:")
    presets = {
        'fast': 'Fast processing, lower quality',
        'balanced': 'Balanced speed and quality (default)',
        'quality': 'Slower processing, higher quality'
    }
    for preset, desc in presets.items():
        click.echo(f"  • {preset}: {desc}")


@cli.command()
@click.option('--session-id', help='Specific session ID to analyze')
@click.option('--last-n', default=5, type=int, help='Analyze last N sessions')
def logs(session_id, last_n):
    """Analyze processing logs and performance data"""
    
    click.echo("📝 Log Analysis:")
    
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            click.echo("No logs directory found")
            return
        
        # Find log files
        if session_id:
            log_files = list(logs_dir.glob(f"processing_log_{session_id}.json"))
        else:
            log_files = sorted(logs_dir.glob("processing_log_*.json"))[-last_n:]
        
        if not log_files:
            click.echo("No log files found")
            return
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                result = log_data.get("result", {})
                session = log_data.get("session_id", "Unknown")
                
                click.echo(f"\n📊 Session: {session}")
                click.echo(f"  • Success: {'Yes' if result.get('success') else 'No'}")
                
                if result.get("success"):
                    stats = result.get("stats", {})
                    click.echo(f"  • Frames: {stats.get('processed_frame_count', 0)}")
                    click.echo(f"  • Scale: {stats.get('scale_factor', 'N/A')}x")
                else:
                    click.echo(f"  • Error: {result.get('error', 'Unknown')}")
                
                # Show processing stages
                stages = result.get("processing_stages", [])
                if stages:
                    click.echo(f"  • Completed stages: {len(stages)}")
                
            except Exception as e:
                click.echo(f"  ⚠️  Error reading {log_file}: {e}")
        
    except Exception as e:
        click.echo(f"💥 Error analyzing logs: {e}", err=True)


if __name__ == '__main__':
    cli()