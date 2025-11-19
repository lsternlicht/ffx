#!/usr/bin/env python3
"""
ffx - FFmpeg, but improved
A beautiful, intuitive wrapper for ffmpeg with smart defaults and helpful guidance.

Installation:
    pip install rich click humanize
    Save this file as 'ffx'
    chmod +x ffx
    Move to /usr/local/bin/ or add to PATH

Requirements:
    - ffmpeg installed on system
    - Python 3.7+
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

try:
    import click
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.syntax import Syntax
    from rich import print as rprint
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    import humanize
except ImportError:
    print("⚠️  Please install required packages:")
    print("   pip install rich click humanize")
    sys.exit(1)

console = Console()

# Version
__version__ = "1.0.0"

# Presets configuration
PRESETS = {
    "social": {
        "name": "Social Media",
        "desc": "Optimized for Twitter, Instagram, TikTok",
        "video_codec": "libx264",
        "video_bitrate": "2M",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
        "extra": ["-preset", "fast", "-movflags", "+faststart"],
        "max_size_mb": 100,
    },
    "streaming": {
        "name": "Web Streaming",
        "desc": "Optimized for web playback (YouTube, Vimeo)",
        "video_codec": "libx264",
        "video_bitrate": "5M",
        "audio_codec": "aac", 
        "audio_bitrate": "192k",
        "extra": ["-preset", "medium", "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    },
    "mobile": {
        "name": "Mobile Devices",
        "desc": "Smaller files for mobile storage/sharing",
        "video_codec": "libx265",
        "video_bitrate": "1M",
        "audio_codec": "aac",
        "audio_bitrate": "96k",
        "extra": ["-preset", "medium"],
    },
    "archive": {
        "name": "Archive Quality",
        "desc": "High quality for long-term storage",
        "video_codec": "libx265",
        "video_bitrate": "10M",
        "audio_codec": "flac",
        "extra": ["-preset", "slow", "-crf", "18"],
    },
    "gif": {
        "name": "Animated GIF",
        "desc": "Optimized GIF creation",
        "extra": ["-vf", "fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"],
    }
}

@dataclass
class MediaInfo:
    """Information about a media file"""
    duration: float
    size: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    bitrate: int
    has_audio: bool
    
    @property
    def resolution_name(self) -> str:
        if self.height >= 2160:
            return "4K"
        elif self.height >= 1440:
            return "2K"
        elif self.height >= 1080:
            return "1080p"
        elif self.height >= 720:
            return "720p"
        elif self.height >= 480:
            return "480p"
        return "SD"

class FFmpegWrapper:
    """Core ffmpeg wrapper functionality"""
    
    def __init__(self):
        self.check_ffmpeg()
    
    def check_ffmpeg(self):
        """Check if ffmpeg is installed"""
        if not shutil.which("ffmpeg"):
            console.print("[red]❌ ffmpeg is not installed![/red]")
            console.print("\n[yellow]To install ffmpeg:[/yellow]")
            console.print("  • macOS:    brew install ffmpeg")
            console.print("  • Ubuntu:   sudo apt install ffmpeg")
            console.print("  • Windows:  winget install ffmpeg")
            sys.exit(1)
    
    def get_media_info(self, file_path: Path) -> Optional[MediaInfo]:
        """Extract media information using ffprobe"""
        if not shutil.which("ffprobe"):
            return None
        
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            # Extract info
            format_info = data.get("format", {})
            video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
            audio_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), None)
            
            if not video_stream and not audio_stream:
                return None
            
            return MediaInfo(
                duration=float(format_info.get("duration", 0)),
                size=int(format_info.get("size", 0)),
                width=int(video_stream.get("width", 0)) if video_stream else 0,
                height=int(video_stream.get("height", 0)) if video_stream else 0,
                fps=eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
                video_codec=video_stream.get("codec_name", "") if video_stream else "",
                audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
                bitrate=int(format_info.get("bit_rate", 0)),
                has_audio=audio_stream is not None
            )
        except Exception as e:
            return None
    
    def estimate_size(self, duration: float, video_bitrate: str, audio_bitrate: str = "128k") -> int:
        """Estimate output file size"""
        # Parse bitrates
        vbr = self._parse_bitrate(video_bitrate)
        abr = self._parse_bitrate(audio_bitrate)
        total_bitrate = vbr + abr
        # Calculate size (bitrate * duration / 8)
        return int((total_bitrate * duration) / 8)
    
    def _parse_bitrate(self, bitrate_str: str) -> int:
        """Parse bitrate string to bits per second"""
        if not bitrate_str:
            return 0
        bitrate_str = bitrate_str.lower()
        multiplier = 1
        if bitrate_str.endswith('k'):
            multiplier = 1000
            bitrate_str = bitrate_str[:-1]
        elif bitrate_str.endswith('m'):
            multiplier = 1000000
            bitrate_str = bitrate_str[:-1]
        try:
            return int(float(bitrate_str) * multiplier)
        except:
            return 0
    
    def build_command(self, input_file: Path, output_file: Path, 
                     preset: Optional[str] = None, **kwargs) -> List[str]:
        """Build ffmpeg command with smart defaults"""
        cmd = ["ffmpeg", "-i", str(input_file), "-y"]
        
        if preset and preset in PRESETS:
            p = PRESETS[preset]
            if "video_codec" in p:
                cmd.extend(["-c:v", p["video_codec"]])
            if "video_bitrate" in p:
                cmd.extend(["-b:v", p["video_bitrate"]])
            if "audio_codec" in p:
                cmd.extend(["-c:a", p["audio_codec"]])
            if "audio_bitrate" in p:
                cmd.extend(["-b:a", p["audio_bitrate"]])
            if "extra" in p:
                cmd.extend(p["extra"])
        
        # Add any additional arguments
        for key, value in kwargs.items():
            if value is not None:
                if key == "start_time":
                    cmd.extend(["-ss", str(value)])
                elif key == "end_time":
                    cmd.extend(["-to", str(value)])
                elif key == "scale":
                    cmd.extend(["-vf", f"scale={value}"])
                
        cmd.append(str(output_file))
        return cmd
    
    def run_with_progress(self, cmd: List[str], duration: Optional[float] = None):
        """Run ffmpeg command with progress bar"""
        console.print(f"\n[cyan]🎬 Processing...[/cyan]")
        
        # Show command in debug mode
        if os.environ.get("FFX_DEBUG"):
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Converting...", total=100)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Parse ffmpeg output for progress
            for line in process.stderr:
                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        parts = time_str.split(":")
                        current_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        if duration and duration > 0:
                            percentage = (current_time / duration) * 100
                            progress.update(task, completed=min(percentage, 100))
                    except:
                        pass
            
            process.wait()
            progress.update(task, completed=100)
            
            if process.returncode != 0:
                console.print(f"\n[red]❌ Error during conversion[/red]")
                if os.environ.get("FFX_DEBUG"):
                    stderr = process.stderr.read() if process.stderr else ""
                    console.print(f"[dim]{stderr}[/dim]")
                return False
        
        return True

# Initialize wrapper
ffmpeg = FFmpegWrapper()

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    return humanize.naturalsize(size_bytes, binary=True)

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def show_file_info(file_path: Path, info: MediaInfo):
    """Display file information in a nice panel"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("📁 File", file_path.name)
    table.add_row("💾 Size", format_size(info.size))
    table.add_row("⏱️  Duration", format_duration(info.duration))
    
    if info.width and info.height:
        table.add_row("📐 Resolution", f"{info.width}×{info.height} ({info.resolution_name})")
        table.add_row("🎞️  FPS", f"{info.fps:.2f}")
        table.add_row("🎥 Video Codec", info.video_codec)
    
    if info.has_audio:
        table.add_row("🔊 Audio Codec", info.audio_codec)
    
    table.add_row("📊 Bitrate", f"{info.bitrate // 1000} kbps")
    
    console.print(Panel(table, title="[bold]Input File Info[/bold]", border_style="blue"))

def interactive_mode(input_file: Path):
    """Interactive mode with smart suggestions"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🎯 FFX Interactive Mode[/bold cyan]\n"
        "Smart video processing with personalized recommendations",
        border_style="cyan"
    ))
    
    # Analyze input file
    info = ffmpeg.get_media_info(input_file)
    if not info:
        console.print("[red]Could not analyze input file[/red]")
        return
    
    show_file_info(input_file, info)
    
    # Ask what user wants to do
    console.print("\n[bold]What would you like to do?[/bold]\n")
    options = [
        "1. Compress for smaller size",
        "2. Convert to different format", 
        "3. Extract clip/trim",
        "4. Create animated GIF",
        "5. Extract audio only",
        "6. Optimize for social media",
        "7. Custom settings"
    ]
    
    for opt in options:
        console.print(f"  {opt}")
    
    choice = IntPrompt.ask("\n[cyan]Select option[/cyan]", choices=["1","2","3","4","5","6","7"])
    
    if choice == 1:  # Compress
        console.print("\n[bold]Compression Options:[/bold]\n")
        
        # Calculate estimates
        sizes = {
            "High Quality": (info.size * 0.5, "libx264", "3M", "Excellent quality, ~50% smaller"),
            "Balanced": (info.size * 0.3, "libx264", "2M", "Good quality, ~70% smaller"),
            "Maximum": (info.size * 0.15, "libx265", "1M", "OK quality, ~85% smaller"),
        }
        
        table = Table(show_header=True)
        table.add_column("Option", style="cyan")
        table.add_column("Est. Size", style="yellow")
        table.add_column("Details", style="dim")
        
        for name, (size, codec, bitrate, desc) in sizes.items():
            table.add_row(name, format_size(int(size)), desc)
        
        console.print(table)
        
        quality = Prompt.ask(
            "\n[cyan]Choose quality[/cyan]",
            choices=["high", "balanced", "maximum"],
            default="balanced"
        )
        
        output_file = input_file.with_suffix(".compressed.mp4")
        
        # Build and run command
        if quality == "high":
            cmd = ffmpeg.build_command(input_file, output_file, video_bitrate="3M")
        elif quality == "maximum":
            cmd = ffmpeg.build_command(input_file, output_file, video_bitrate="1M", video_codec="libx265")
        else:
            cmd = ffmpeg.build_command(input_file, output_file, preset="mobile")
        
        if ffmpeg.run_with_progress(cmd, info.duration):
            new_size = output_file.stat().st_size
            console.print(f"\n✨ [green]Success![/green] Compressed from {format_size(info.size)} to {format_size(new_size)}")
            console.print(f"   Saved: {format_size(info.size - new_size)} ({100 - (new_size/info.size)*100:.1f}% reduction)")
            console.print(f"   Output: [cyan]{output_file}[/cyan]")
    
    elif choice == 3:  # Trim
        console.print(f"\n[dim]Video duration: {format_duration(info.duration)}[/dim]")
        
        start_time = Prompt.ask("\n[cyan]Start time[/cyan] (e.g., 00:30 or 30)", default="0")
        end_time = Prompt.ask("[cyan]End time[/cyan] (e.g., 01:45 or 105)")
        
        output_file = input_file.with_stem(f"{input_file.stem}_clip")
        cmd = ffmpeg.build_command(input_file, output_file, start_time=start_time, end_time=end_time)
        
        if ffmpeg.run_with_progress(cmd):
            console.print(f"\n✨ [green]Success![/green] Created clip: [cyan]{output_file}[/cyan]")
    
    elif choice == 4:  # GIF
        console.print("\n[bold]GIF Settings:[/bold]")
        
        width = IntPrompt.ask("[cyan]Width in pixels[/cyan]", default=480)
        fps = IntPrompt.ask("[cyan]Frames per second[/cyan]", default=15)
        
        output_file = input_file.with_suffix(".gif")
        
        # Two-pass GIF creation for better quality
        cmd = [
            "ffmpeg", "-i", str(input_file), "-y",
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_file)
        ]
        
        if ffmpeg.run_with_progress(cmd, info.duration):
            size = output_file.stat().st_size
            console.print(f"\n✨ [green]Success![/green] Created GIF: [cyan]{output_file}[/cyan] ({format_size(size)})")
    
    elif choice == 5:  # Extract audio
        if not info.has_audio:
            console.print("[red]This file has no audio track![/red]")
            return
        
        format_choice = Prompt.ask(
            "\n[cyan]Output format[/cyan]",
            choices=["mp3", "aac", "wav", "flac"],
            default="mp3"
        )
        
        output_file = input_file.with_suffix(f".{format_choice}")
        
        codecs = {"mp3": "libmp3lame", "aac": "aac", "wav": "pcm_s16le", "flac": "flac"}
        cmd = ["ffmpeg", "-i", str(input_file), "-vn", "-c:a", codecs[format_choice], str(output_file)]
        
        if ffmpeg.run_with_progress(cmd, info.duration):
            size = output_file.stat().st_size
            console.print(f"\n✨ [green]Success![/green] Extracted audio: [cyan]{output_file}[/cyan] ({format_size(size)})")
    
    elif choice == 6:  # Social media
        console.print("\n[bold]Social Media Platforms:[/bold]\n")
        
        platforms = {
            "twitter": ("Twitter/X", "Max 512MB, 2:20 duration", "social"),
            "instagram": ("Instagram Reel", "Max 100MB, 90 seconds", "social"),
            "tiktok": ("TikTok", "Max 287MB, 10 minutes", "social"),
            "youtube": ("YouTube", "Optimized streaming", "streaming"),
        }
        
        for key, (name, desc, preset) in platforms.items():
            console.print(f"  • [cyan]{name}[/cyan] - {desc}")
        
        platform = Prompt.ask("\n[cyan]Choose platform[/cyan]", choices=list(platforms.keys()))
        preset = platforms[platform][2]
        
        output_file = input_file.with_stem(f"{input_file.stem}_{platform}")
        cmd = ffmpeg.build_command(input_file, output_file, preset=preset)
        
        if ffmpeg.run_with_progress(cmd, info.duration):
            size = output_file.stat().st_size
            console.print(f"\n✨ [green]Success![/green] Optimized for {platforms[platform][0]}")
            console.print(f"   Output: [cyan]{output_file}[/cyan] ({format_size(size)})")

@click.group(invoke_without_command=True, context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(__version__, '-v', '--version', prog_name='ffx')
@click.pass_context
def cli(ctx):
    """
    🎬 ffx - FFmpeg, but improved
    
    Beautiful, intuitive video/audio processing with smart defaults.
    
    Examples:
        ffx compress video.mp4
        ffx convert video.avi output.mp4
        ffx trim video.mp4 00:30 01:45
        ffx interactive video.mp4
    """
    if ctx.invoked_subcommand is None:
        # Show beautiful help
        console.print(Panel.fit(
            "[bold cyan]🎬 FFX - FFmpeg, but improved[/bold cyan]\n"
            f"Version {__version__}\n\n"
            "[yellow]Quick Start:[/yellow]\n"
            "  ffx compress video.mp4      - Smart compression\n"
            "  ffx interactive video.mp4   - Interactive mode\n"
            "  ffx --help                  - Show all commands",
            border_style="cyan"
        ))

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output filename')
@click.option('--quality', '-q', type=click.Choice(['high', 'medium', 'low']), default='medium')
@click.option('--preset', '-p', type=click.Choice(list(PRESETS.keys())), help='Use preset')
def compress(input_file, output, quality, preset):
    """Intelligently compress video files"""
    input_path = Path(input_file)
    output_path = Path(output) if output else input_path.with_stem(f"{input_path.stem}_compressed")
    
    info = ffmpeg.get_media_info(input_path)
    if info:
        show_file_info(input_path, info)
    
    # Quality to bitrate mapping
    bitrates = {
        'high': '5M',
        'medium': '2M', 
        'low': '1M'
    }
    
    console.print(f"\n[cyan]Compressing with {quality} quality...[/cyan]")
    
    cmd = ffmpeg.build_command(
        input_path, 
        output_path,
        preset=preset,
        video_bitrate=bitrates[quality]
    )
    
    if ffmpeg.run_with_progress(cmd, info.duration if info else None):
        if info:
            new_size = output_path.stat().st_size
            console.print(f"\n✨ [green]Success![/green]")
            console.print(f"   Original: {format_size(info.size)}")
            console.print(f"   New size: {format_size(new_size)}")
            console.print(f"   Saved: {format_size(info.size - new_size)} ({100 - (new_size/info.size)*100:.1f}% reduction)")
        console.print(f"   Output: [cyan]{output_path}[/cyan]")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--video-codec', '-vc', help='Video codec')
@click.option('--audio-codec', '-ac', help='Audio codec')
@click.option('--preset', '-p', type=click.Choice(list(PRESETS.keys())), help='Use preset')
def convert(input_file, output_file, video_codec, audio_codec, preset):
    """Convert between formats with optimal settings"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    info = ffmpeg.get_media_info(input_path)
    if info:
        show_file_info(input_path, info)
    
    console.print(f"\n[cyan]Converting to {output_path.suffix}...[/cyan]")
    
    kwargs = {}
    if video_codec:
        kwargs['video_codec'] = video_codec
    if audio_codec:
        kwargs['audio_codec'] = audio_codec
    
    cmd = ffmpeg.build_command(input_path, output_path, preset=preset, **kwargs)
    
    if ffmpeg.run_with_progress(cmd, info.duration if info else None):
        console.print(f"\n✨ [green]Success![/green] Converted to: [cyan]{output_path}[/cyan]")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('start_time')
@click.argument('end_time')
@click.option('--output', '-o', type=click.Path(), help='Output filename')
def trim(input_file, start_time, end_time, output):
    """Extract a clip from video"""
    input_path = Path(input_file)
    output_path = Path(output) if output else input_path.with_stem(f"{input_path.stem}_clip")
    
    info = ffmpeg.get_media_info(input_path)
    
    console.print(f"[cyan]✂️  Trimming from {start_time} to {end_time}...[/cyan]")
    
    cmd = ffmpeg.build_command(
        input_path,
        output_path,
        start_time=start_time,
        end_time=end_time
    )
    
    if ffmpeg.run_with_progress(cmd, info.duration if info else None):
        console.print(f"\n✨ [green]Success![/green] Created clip: [cyan]{output_path}[/cyan]")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--width', '-w', type=int, default=480, help='GIF width in pixels')
@click.option('--fps', '-f', type=int, default=15, help='Frames per second')
@click.option('--output', '-o', type=click.Path(), help='Output filename')
def gif(input_file, width, fps, output):
    """Create optimized animated GIFs"""
    input_path = Path(input_file)
    output_path = Path(output) if output else input_path.with_suffix('.gif')
    
    info = ffmpeg.get_media_info(input_path)
    
    console.print(f"[cyan]🎞️  Creating GIF ({width}px wide, {fps} fps)...[/cyan]")
    
    # Two-pass GIF creation for quality
    cmd = [
        "ffmpeg", "-i", str(input_path), "-y",
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(output_path)
    ]
    
    if ffmpeg.run_with_progress(cmd, info.duration if info else None):
        size = output_path.stat().st_size
        console.print(f"\n✨ [green]Success![/green] Created GIF: [cyan]{output_path}[/cyan] ({format_size(size)})")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['mp3', 'aac', 'wav', 'flac']), default='mp3')
@click.option('--output', '-o', type=click.Path(), help='Output filename')
def audio(input_file, format, output):
    """Extract audio from video files"""
    input_path = Path(input_file)
    output_path = Path(output) if output else input_path.with_suffix(f'.{format}')
    
    info = ffmpeg.get_media_info(input_path)
    
    if info and not info.has_audio:
        console.print("[red]❌ This file has no audio track![/red]")
        return
    
    console.print(f"[cyan]🔊 Extracting audio as {format}...[/cyan]")
    
    codecs = {"mp3": "libmp3lame", "aac": "aac", "wav": "pcm_s16le", "flac": "flac"}
    cmd = ["ffmpeg", "-i", str(input_path), "-vn", "-c:a", codecs[format], str(output_path)]
    
    if ffmpeg.run_with_progress(cmd, info.duration if info else None):
        size = output_path.stat().st_size
        console.print(f"\n✨ [green]Success![/green] Extracted audio: [cyan]{output_path}[/cyan] ({format_size(size)})")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def interactive(input_file):
    """Interactive mode with smart suggestions"""
    interactive_mode(Path(input_file))

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def info(input_file):
    """Display detailed file information"""
    input_path = Path(input_file)
    info = ffmpeg.get_media_info(input_path)
    
    if not info:
        console.print("[red]Could not analyze file[/red]")
        return
    
    show_file_info(input_path, info)
    
    # Show additional technical details
    console.print("\n[bold]Preset Recommendations:[/bold]")
    
    recommendations = []
    if info.size > 100 * 1024 * 1024:  # > 100MB
        recommendations.append("• Consider using [cyan]compress[/cyan] to reduce file size")
    
    if info.resolution_name in ["4K", "2K"]:
        recommendations.append("• File is high-res - [cyan]mobile[/cyan] preset recommended for sharing")
    
    if info.video_codec not in ["h264", "libx264", "h265", "libx265"]:
        recommendations.append("• Non-standard codec - consider [cyan]convert[/cyan] for better compatibility")
    
    if not recommendations:
        recommendations.append("• File is well-optimized!")
    
    for rec in recommendations:
        console.print(rec)

@cli.command()
def presets():
    """Show all available presets"""
    console.print(Panel.fit("[bold]📋 Available Presets[/bold]", border_style="cyan"))
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Preset", style="yellow")
    table.add_column("Description")
    table.add_column("Best For", style="dim")
    
    use_cases = {
        "social": "Twitter, Instagram, TikTok uploads",
        "streaming": "YouTube, Vimeo, web playback",
        "mobile": "WhatsApp, saving phone storage",
        "archive": "Long-term storage, backup",
        "gif": "Memes, reactions, demos"
    }
    
    for key, preset in PRESETS.items():
        table.add_row(
            key,
            preset['desc'],
            use_cases.get(key, "")
        )
    
    console.print(table)
    console.print("\n[dim]Use with: ffx compress -p <preset> video.mp4[/dim]")

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        if os.environ.get("FFX_DEBUG"):
            console.print_exception()
        else:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Set FFX_DEBUG=1 for detailed errors[/dim]")
        sys.exit(1)
