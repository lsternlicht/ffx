<p align="center">
  <img src="ffx-image.png" alt="ffx logo" width="400" />
</p>

# ffx - FFmpeg, but improved

A beautiful, intuitive wrapper for ffmpeg with smart defaults and helpful guidance.

## Installation

You can install `ffx` directly from the source. This will automatically install dependencies and the `ffx` command line tool.

```bash
# Install from current directory
pip install .

# Or for development (editable mode)
pip install -e .
```

## Requirements

- ffmpeg installed on system
- Python 3.7+

## Quickstart

Run the interactive mode for guided optimization:

```bash
ffx interactive video.mp4
```

Compress a video to reduce size significantly:

```bash
ffx compress input.mp4
```

Trim a video clip (start at 30s, end at 1m 45s):

```bash
ffx trim input.mp4 00:30 01:45
```

Convert video to a GIF:

```bash
ffx gif input.mp4 --width 480 --fps 15
```

Extract audio from a video:

```bash
ffx audio input.mp4 --format mp3
```

Check file information:

```bash
ffx info input.mp4
```

For a full list of commands:

```bash
ffx --help
```
