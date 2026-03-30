# EdgeRVC - Text-to-Speech and Voice Conversion Web Interface

<div align="center">
  <img src="https://raw.githubusercontent.com/5SRT7/EdgeRVC/main/logo.png" alt="EdgeRVC Logo" width="400" height="200">
  <p>A retrieval-based voice conversion web interface that supports text-to-speech, SRT file import, and voice conversion</p>
</div>

## 🌐 Language / 语言

- [English](README.md) | [中文](README-zh.md)

## 📋 Project Introduction

EdgeRVC is a powerful voice processing tool that combines Edge TTS's text-to-speech capability with RVC (Retrieval-based Voice Conversion) technology, allowing users to easily convert text or SRT subtitle files to speech and perform voice conversion.

## 💡 Project Motivation

The motivation behind this project is quite simple.

I often work on videos or small projects that require TTS (Text-to-Speech).

However, I found that while there are many **free TTS** options available, most of them have average quality and still sound like machine-generated speech. On the other hand, **high-quality TTS services** usually require subscriptions or pay-per-use, which can be quite expensive.

As a student, if I'm just working on personal projects or videos, it's a bit too costly for me.

So I wondered if there was a way to **create high-quality voice generation using free tools**.

That's how I created this project called EdgeRVC.

Simply put, **EdgeRVC is a tool that combines EdgeTTS and RVC**.

## ✨ Key Features

- **🎤 Text-to-Speech**: Generate natural and fluent speech using Edge TTS
- **📄 SRT File Import**: Support importing text from subtitle files
- **🎭 Voice Conversion**: High-quality voice conversion using RVC models
- **🗣️ Multiple Voice Options**: Provide multiple Chinese voice options
- **⚙️ Adjustable Parameters**: Support adjusting speech speed, pitch, pitch extraction algorithm, etc.
- **📁 Multiple Output Formats**: Support wav, flac, mp3, m4a, etc.
- **🌐 Intuitive Web Interface**: User-friendly interface built on Gradio

## 🛠️ Tech Stack

- **Backend**: Python 3.8+
- **Web Framework**: Gradio
- **Speech Synthesis**: Edge TTS
- **Voice Conversion**: RVC (Retrieval-based Voice Conversion)
- **Deep Learning**: PyTorch
- **Feature Retrieval**: FAISS

## 📦 System Requirements

- Python 3.8 or higher
- At least 4GB RAM
- Supported operating systems: Windows, macOS, Linux
- (Optional) NVIDIA GPU for better performance

## 🚀 Installation Guide

### 1. Clone the Project

```bash
git clone https://github.com/5SRT7/EdgeRVC.git
cd EdgeRVC-main
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment using venv
python3 -m venv .venv

# Activate the virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variable Configuration

The project already includes a `.env` file, you can modify the configuration as needed:

```env
# Model weight directory
weight_root=./assets/weights

# Index file directory
index_root=./assets/indices

# External index file directory
outside_index_root=./assets/outside_indices
```

### 5. Prepare Model Files

1. Place RVC model files (.pth format) in the `./assets/weights` directory
2. Place feature index files (.index format) in the `./assets/indices` directory

## 📚 Pre-models and Dependencies

RVC requires some other pre-models for inference and training.

You can download these models from `https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main`.

### 1. Download assets

Here is a list of all pre-models and other files required for RVC:

- `./assets/hubert/hubert_base.pt`
- `./assets/pretrained`

To use v2 version models, you need to additionally download:

- `./assets/pretrained_v2`

### 2. Install ffmpeg

Skip if ffmpeg and ffprobe are already installed.

**Ubuntu/Debian Users**
```bash
sudo apt install ffmpeg
```

**MacOS Users**
```bash
brew install ffmpeg
```

**Windows Users**
Download and place in the root directory:
- Download ffmpeg.exe
- Download ffprobe.exe

### 3. Download files required for RMVPE pitch extraction algorithm

If you want to use the latest RMVPE pitch extraction algorithm, you need to download the pitch extraction model parameters and place them in the RVC root directory:

- Download rmvpe.pt
- Download rmvpe's DML environment (optional, for AMD/Intel GPU users)
  - Download rmvpe.onnx

## 🎯 Usage Instructions

### Start the Application

```bash
python infer-web.py
```

The application will start at `http://localhost:7865`.

### Usage Steps

1. **Select Input Method**:
   - Text Input: Directly enter the text to be converted in the text box
   - SRT File Import: Upload an SRT subtitle file

2. **Select Voice**: Choose an Edge TTS voice from the dropdown menu

3. **Adjust Speech Speed**: Use the slider to adjust speech speed

4. **Select Voice Conversion Model**: Choose an RVC model from the dropdown menu

5. **Adjust Voice Conversion Parameters**:
   - Speaker ID: Select the speaker in the model
   - Pitch Shift: Adjust the pitch (number of semitones)
   - Pitch Extraction Algorithm: Choose a suitable algorithm
   - Feature Retrieval Library: Select the feature index file
   - Retrieval Feature Ratio: Adjust the weight of feature retrieval
   - Post-processing Resampling: Set the final sampling rate
   - Export File Format: Select the output audio format
   - Save Directory: Set the output file save location

6. **Generate and Convert Voice**: Click the "Generate Speech and Convert Voice" button

7. **View Results**: View the generated audio and information in the output area

8. **Delete Audio**: Click the "Delete Audio" button to delete the generated file

## ⚙️ Command Line Parameters

- `--port`: Set the listening port (default: 7865)
- `--pycmd`: Set the Python command (default: current Python)
- `--colab`: Launch in Colab
- `--noparallel`: Disable parallel processing
- `--noautoopen`: Disable automatic opening in browser
- `--dml`: Use DirectML

## 📁 Project Structure

```
Retrieval-based-Voice-Conversion-WebUI/
├── assets/              # Resource directory
│   ├── weights/         # Model weight files
│   └── indices/         # Feature index files
├── configs/             # Configuration files directory
│   ├── config.py        # Main configuration file
│   └── inuse/           # Runtime configuration
├── infer/               # Inference module
│   └── modules/         # Core modules
├── logs/                # Log directory
├── output/              # Output directory
├── .env                 # Environment variable configuration
├── infer-web.py         # Main application file
└── requirements.txt     # Dependencies
```

## ⚠️ Notes

1. **Network Connection**: Using Edge TTS requires a network connection
2. **Model Files**: RVC model files and feature index files need to be prepared in advance
3. **Performance**: Running on CPU may be slow, GPU is recommended for better performance
4. **Copyright**: Please ensure compliance with relevant audio and model copyright regulations

## 📝 Disclaimer

The author of this software has no control over the software. Users of the software and those who distribute voices exported by the software are solely responsible. If you do not agree to this clause, you cannot use or reference any code or files in the software package.

## 🤝 Contribution

Welcome to submit Issues and Pull Requests to improve this project!

## 📞 Contact

If you have any questions or suggestions, please contact us through GitHub Issues.

## 🙏 Acknowledgements

This project uses code and ideas from the following open-source projects:

- **Edge TTS**
  `https://github.com/rany2/edge-tts`

- **Retrieval-based Voice Conversion (RVC)**
  `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`

Sincere thanks to the developers and contributors of these projects.

---

<div align="center">
  <p>✨ Thank you for using EdgeRVC ✨</p>
</div>