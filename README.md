# EdgeRVC - 文本转语音与语音转换Web界面

<div align="center">
  <img src="https://raw.githubusercontent.com/5SRT7/EdgeRVC/main/logo.png" alt="EdgeRVC Logo" width="400" height="200">
  <p>一个基于检索的语音转换Web界面，支持文本转语音、SRT文件导入和语音变声功能</p>
</div>

## 📋 项目简介

EdgeRVC是一个功能强大的语音处理工具，结合了Edge TTS的文本转语音能力和RVC（Retrieval-based Voice Conversion）的语音转换技术，让用户可以轻松地将文本或SRT字幕文件转换为语音并进行变声处理。

## 💡 项目初衷

做这个项目的初衷其实很简单。

我平时会做一些视频或者小项目，需要用到 TTS，也就是文本转语音。

但是我发现，现在市面上的 **免费 TTS** 虽然很多，但大多数效果都比较一般，听起来还是比较像机器读出来的。而那些 **效果比较好的 TTS 服务**，基本上都需要订阅或者按量付费，价格其实不算便宜。

对于像我这样的学生来说，如果只是做一些个人项目或者视频，其实还是有点负担不起。

所以我就在想，有没有办法 **用免费的工具，做出效果比较好的语音生成**。

于是我就做了这个项目，叫 EdgeRVC。

简单来说，**EdgeRVC 是一个把 EdgeTTS 和 RVC 结合起来的工具**。

## ✨ 主要功能

- **🎤 文本转语音**：使用Edge TTS生成自然流畅的语音
- **📄 SRT文件导入**：支持从字幕文件导入文本
- **🎭 语音变声**：使用RVC模型进行高质量的语音转换
- **🗣️ 多种语音选择**：提供多种中文语音选项
- **⚙️ 参数可调**：支持调整语音速度、变调、音高提取算法等参数
- **📁 多种输出格式**：支持wav、flac、mp3、m4a等格式
- **🌐 直观的Web界面**：基于Gradio构建的用户友好界面

## 🛠️ 技术栈

- **后端**：Python 3.8+
- **Web框架**：Gradio
- **语音合成**：Edge TTS
- **语音转换**：RVC (Retrieval-based Voice Conversion)
- **深度学习**：PyTorch
- **特征检索**：FAISS

## 📦 系统要求

- Python 3.8 或更高版本
- 至少 4GB RAM
- 支持的操作系统：Windows、macOS、Linux
- （可选）NVIDIA GPU 以获得更好的性能

## 🚀 安装指南

### 1. 克隆项目

```bash
git clone https://github.com/5SRT7/EdgeRVC.git
cd EdgeRVC-main
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用venv创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 环境变量配置

项目已包含 `.env` 文件，你可以根据需要修改其中的配置：

```env
# 模型权重目录
weight_root=./assets/weights

# 索引文件目录
index_root=./assets/indices

# 外部索引文件目录
outside_index_root=./assets/outside_indices
```

### 5. 准备模型文件

1. 在 `./assets/weights` 目录中放置RVC模型文件（.pth格式）
2. 在 `./assets/indices` 目录中放置特征索引文件（.index格式）

## 📚 预模型和依赖项

RVC需要其他一些预模型来推理和训练。

你可以从 `https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main` 下载到这些模型。

### 1. 下载 assets

以下是一份清单，包括了所有RVC所需的预模型和其他文件的名称：

- `./assets/hubert/hubert_base.pt`
- `./assets/pretrained`

想使用v2版本模型的话，需要额外下载：

- `./assets/pretrained_v2`

### 2. 安装 ffmpeg

若ffmpeg和ffprobe已安装则跳过。

**Ubuntu/Debian 用户**
```bash
sudo apt install ffmpeg
```

**MacOS 用户**
```bash
brew install ffmpeg
```

**Windows 用户**
下载后放置在根目录：
- 下载ffmpeg.exe
- 下载ffprobe.exe

### 3. 下载 rmvpe 人声音高提取算法所需文件

如果你想使用最新的RMVPE人声音高提取算法，则你需要下载音高提取模型参数并放置于RVC根目录：

- 下载rmvpe.pt
- 下载 rmvpe 的 dml 环境(可选, A卡/I卡用户)
  - 下载rmvpe.onnx

## 🎯 使用说明

### 启动应用

```bash
python infer-web.py
```

应用将在 `http://localhost:7865` 启动。

### 使用步骤

1. **选择输入方式**：
   - 文本输入：直接在文本框中输入要转换的文本
   - SRT文件导入：上传SRT字幕文件

2. **选择语音**：从下拉菜单中选择Edge TTS语音

3. **调整语音速度**：使用滑块调整语音速度

4. **选择变声音色**：从下拉菜单中选择RVC模型

5. **调整变声参数**：
   - 说话人ID：选择模型中的说话人
   - 变调：调整音高（半音数量）
   - 音高提取算法：选择适合的算法
   - 特征检索库：选择特征索引文件
   - 检索特征占比：调整特征检索的权重
   - 后处理重采样：设置最终采样率
   - 导出文件格式：选择输出音频格式
   - 保存目录：设置输出文件保存位置

6. **生成并变声**：点击"生成语音并变声"按钮

7. **查看结果**：在输出区域查看生成的音频和信息

8. **删除音频**：点击"删除音频"按钮删除生成的文件

## ⚙️ 命令行参数

- `--port`：设置监听端口（默认：7865）
- `--pycmd`：设置Python命令（默认：当前Python）
- `--colab`：在Colab中启动
- `--noparallel`：禁用并行处理
- `--noautoopen`：禁止自动在浏览器中打开
- `--dml`：使用DirectML

## 📁 项目结构

```
Retrieval-based-Voice-Conversion-WebUI/
├── assets/              # 资源目录
│   ├── weights/         # 模型权重文件
│   └── indices/         # 特征索引文件
├── configs/             # 配置文件目录
│   ├── config.py        # 主配置文件
│   └── inuse/           # 运行时配置
├── infer/               # 推理模块
│   └── modules/         # 核心模块
├── logs/                # 日志目录
├── output/              # 输出目录
├── .env                 # 环境变量配置
├── infer-web.py         # 主应用文件
└── requirements.txt     # 依赖项
```

## ⚠️ 注意事项

1. **网络连接**：使用Edge TTS需要网络连接
2. **模型文件**：需要提前准备RVC模型文件和特征索引文件
3. **性能**：在CPU上运行可能较慢，建议使用GPU以获得更好的性能
4. **版权**：请确保遵守相关音频和模型的版权规定

## 📝 免责声明

本软件作者不对软件具备任何控制力，使用软件者、传播软件导出的声音者自负全责。如不认可该条款，则不能使用或引用软件包内任何代码和文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系我们。

## 🙏 致谢

本项目使用了以下开源项目的代码和思想：

- **Edge TTS**
  `https://github.com/rany2/edge-tts`

- **检索式语音转换 (RVC)**
  `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`

向这些项目的开发者和贡献者致以诚挚的感谢。

---

<div align="center">
  <p>✨ 感谢使用 EdgeRVC ✨</p>
</div>