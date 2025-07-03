# Meeting Transcription and Processing

A comprehensive Python notebook for transcribing audio meetings with speaker identification and intelligent summarization.

## Features

- **Audio Format Conversion**: Convert M4A/MP4 files to WAV format
- **Speaker Diarization**: Identify who speaks when using pyannote-audio
- **Speech Recognition**: Transcribe speech segments using Google Speech Recognition
- **AI-Powered Summarization**: Generate meeting summaries using Ollama
- **Performance Optimization**: GPU acceleration and audio preprocessing options
- **Flexible Processing**: Choose between speaker diarization or time-based chunking

## Requirements

### System Dependencies

- Python 3.8+
- ffmpeg (for audio conversion)
- Ollama (for summary generation)

### Python Packages

Install required packages using:

```bash
pip install -r requirements.txt
```

### Additional Setup for Speaker Diarization

1. **Accept HuggingFace Model License**

   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the user conditions

2. **Get HuggingFace Token**

   - Get a token from: https://huggingface.co/settings/tokens
   - Copy `.env.sample` to `.env`
   - Set the `USE_AUTH_TOKEN` variable with your token

3. **GPU Support (Optional but Recommended)**
   For faster processing, install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### 1. Audio Conversion

The notebook automatically converts MP4/M4A files to WAV format:

```python
file_path = "meetings\\your-meeting-file"
input_file = file_path + ".mp4"  # or .m4a
output_wav_file = file_path + ".wav"
```

### 2. Processing Modes

#### Speaker Diarization Mode (Recommended)

- **Pros**: Identifies who is speaking, accurate speaker boundaries
- **Cons**: Requires diarization step (can be slow)
- **Best for**: Multi-speaker meetings

#### Time-based Chunking Mode

- **Pros**: Fast, no diarization required
- **Cons**: No speaker identification
- **Best for**: Single speaker recordings or quick transcription

### 3. Performance Expectations

| Processing Mode     | Hardware | Speed            |
| ------------------- | -------- | ---------------- |
| Speaker Diarization | GPU      | ~2-4x real-time  |
| Speaker Diarization | CPU      | ~5-15x real-time |
| Time-based Chunking | Any      | ~1-2x real-time  |

### 4. Optimization Options

- **Audio Preprocessing**: Convert to mono, downsample to 16kHz
- **Segment Filtering**: Skip segments shorter than specified duration
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Batch Processing**: Process multiple segments efficiently

## Workflow

1. **Audio Conversion**: Convert source audio to WAV format
2. **Speaker Diarization** (optional): Identify speaker segments
3. **Speech Recognition**: Transcribe each segment
4. **Summary Generation**: Create intelligent meeting summaries using AI

## Output Files

- `transcription.txt`: Timestamped transcription with speaker labels
- `summary.md`: AI-generated meeting summary
- `meeting_report.md`: Comprehensive report with action items

## Summary Generation

The notebook uses Ollama for intelligent summarization with options for:

- **Basic Summary**: Key questions, suggestions, main topics
- **Detailed Analysis**: Participant contributions, action items, decisions
- **Custom Prompts**: Use your own analysis templates

### Supported Models

- gemma2 (default)
- llama3
- mistral
- qwen2
- And other Ollama-compatible models

## Configuration

Key settings can be adjusted in the notebook:

```python
# Transcription mode
USE_DIARIZATION = True  # False for time-based chunking

# Performance settings
CHUNK_LENGTH_MS = 60000  # 1-minute chunks for time-based mode
min_duration = 3.0       # Skip segments shorter than 3 seconds

# Summary settings
OLLAMA_MODEL = "gemma2"
SUMMARY_TYPE = "detailed"  # "basic", "detailed", or "custom"
```

## Troubleshooting

### Common Issues

1. **Slow Processing**:

   - Enable GPU support
   - Use audio preprocessing
   - Reduce audio quality/length

2. **Diarization Errors**:

   - Check HuggingFace token
   - Verify model license acceptance
   - Try time-based chunking mode

3. **Recognition Failures**:

   - Check internet connection (Google Speech API)
   - Improve audio quality
   - Adjust energy thresholds

4. **Ollama Issues**:
   - Ensure Ollama is running (`ollama serve`)
   - Install required model (`ollama pull gemma2`)
   - Check available models (`ollama list`)

## File Structure

```
meetings_processor/
├── transcriber.ipynb          # Main processing notebook
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create from .env.sample)
├── meetings/                 # Audio files directory
│   ├── *.mp4                # Source audio files
│   └── *.wav                # Converted audio files
├── transcription.txt         # Generated transcription
├── summary.md               # AI-generated summary
└── meeting_report.md        # Comprehensive report
```

## Performance Tips

1. **Use GPU**: Significantly faster diarization processing
2. **Preprocess Audio**: Mono conversion and downsampling reduce processing time
3. **Filter Short Segments**: Skip very brief segments that are likely noise
4. **Batch Processing**: Process multiple files in sequence
5. **Cloud Alternatives**: Consider Azure/Google APIs for very long recordings

## License

This project uses the following key dependencies:

- pyannote-audio (MIT License)
- speech_recognition (BSD License)
- pydub (MIT License)
- ollama (MIT License)

Make sure to comply with HuggingFace model licenses for speaker diarization models.
