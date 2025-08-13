#!/usr/bin/env python3
"""
UpScale App - AI Video Upscaling Tool
Command Line Interface
"""

import click
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from upscale_app import UpScaleApp
from config.settings import VIDEO_SETTINGS, AI_SETTINGS


class ProgressBar:
    """Progress bar handler"""
    
    def __init__(self):
        self.pbar = None
        self.last_message = ""
    
    def update(self, progress: float, message: str):
        """Update progress bar"""
        if self.pbar is None:
            self.pbar = tqdm(total=100, desc=message, unit="%")
        
        # Update progress
        current_progress = int(progress)
        if hasattr(self.pbar, 'n'):
            delta = current_progress - self.pbar.n
            if delta > 0:
                self.pbar.update(delta)
        
        # Update description if message changed
        if message != self.last_message:
            self.pbar.set_description(message)
            self.last_message = message
        
        # Close if complete
        if progress >= 100:
            if self.pbar:
                self.pbar.close()
                self.pbar = None


@click.group()
@click.version_option(version="0.1.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """UpScale App - AI Video Upscaling Tool"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("🎬 UpScale App - AI Video Upscaling Tool")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--scale', '-s', default=1.5, type=float, help='Scale factor (default: 1.5)')
@click.option('--no-ai', is_flag=True, help='Use simple upscaling instead of AI')
@click.option('--no-cleanup', is_flag=True, help='Keep temporary files')
def upscale(input_file, output, scale, no_ai, no_cleanup):
    """Upscale a video file using AI"""
    
    click.echo(f"📹 Processing: {input_file}")
    click.echo(f"🔍 Scale factor: {scale}x")
    
    # Initialize app
    app = UpScaleApp(use_ai=not no_ai, temp_cleanup=not no_cleanup)
    
    # Show system info
    system_info = app.get_system_info()
    if system_info["cuda_available"]:
        click.echo(f"🚀 GPU acceleration available: {system_info['cuda_device_name']}")
    else:
        click.echo("⚠️  Using CPU processing (will be slower)")
    
    # Setup progress tracking
    progress_bar = ProgressBar()
    
    try:
        # Process video
        result = app.process_video(
            input_file, 
            output, 
            scale, 
            progress_callback=progress_bar.update
        )
        
        if result["success"]:
            click.echo(f"✅ Success! Output saved to: {result['output_path']}")
            
            # Show statistics
            stats = result["stats"]
            click.echo("📊 Statistics:")
            click.echo(f"  • Frames processed: {stats.get('processed_frames', 'N/A')}")
            click.echo(f"  • Original size: {stats.get('original_size', 0) / (1024*1024):.1f} MB")
            click.echo(f"  • Output size: {stats.get('output_size', 0) / (1024*1024):.1f} MB")
            click.echo(f"  • Resolution: {stats.get('original_resolution')} → {stats.get('output_resolution')}")
            
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)
            sys.exit(1)
            
    except KeyboardInterrupt:
        click.echo("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"💥 Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output preview file path')
def preview(input_file, output):
    """Create a preview of the upscaling process"""
    
    click.echo(f"🎬 Creating preview for: {input_file}")
    
    app = UpScaleApp(use_ai=False)  # Use simple upscaling for preview
    
    try:
        preview_path = app.create_preview(input_file, output)
        
        if preview_path:
            click.echo(f"✅ Preview created: {preview_path}")
        else:
            click.echo("❌ Failed to create preview", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"💥 Error creating preview: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def info(input_file):
    """Show information about a video file"""
    
    click.echo(f"📹 Video Information: {input_file}")
    
    app = UpScaleApp()
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
        
        # Show upscaling estimates
        time_est = app.video_processor.estimate_processing_time(info)
        click.echo("⏱️  Processing time estimates:")
        click.echo(f"  • With GPU: {time_est['estimated_gpu_minutes']:.1f} minutes")
        click.echo(f"  • With CPU: {time_est['estimated_cpu_minutes']:.1f} minutes")
        
    else:
        click.echo(f"❌ Invalid video file: {validation['error']}", err=True)
        sys.exit(1)


@cli.command()
def system():
    """Show system information"""
    
    click.echo("🖥️  System Information:")
    
    app = UpScaleApp()
    info = app.get_system_info()
    
    click.echo(f"  • Platform: {info['platform']}")
    click.echo(f"  • Python: {info['python_version']}")
    click.echo(f"  • CUDA available: {'Yes' if info['cuda_available'] else 'No'}")
    
    if info['cuda_available']:
        click.echo(f"  • GPU: {info['cuda_device_name']}")
        click.echo(f"  • VRAM: {info['cuda_memory_gb']:.1f} GB")
    
    click.echo(f"  • Temp directory: {info['temp_dir']}")
    click.echo(f"  • Output directory: {info['output_dir']}")
    click.echo(f"  • Max memory: {info['max_memory_gb']} GB")


@cli.command()
def config():
    """Show current configuration"""
    
    click.echo("⚙️  Configuration:")
    
    click.echo("📹 Video Settings:")
    for key, value in VIDEO_SETTINGS.items():
        click.echo(f"  • {key}: {value}")
    
    click.echo("🤖 AI Settings:")
    for key, value in AI_SETTINGS.items():
        click.echo(f"  • {key}: {value}")


if __name__ == '__main__':
    cli()