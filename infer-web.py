import os
import sys
import asyncio
import time

# 强制使用 CPU 以避免 MPS 兼容性问题
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 禁用 CUDA 以确保使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

# 导入 torch 并设置设备为 CPU
import torch
torch.set_default_device("cpu")

from infer.modules.vc.modules import VC
from configs.config import Config
import platform
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
import warnings
import traceback
import shutil
import logging

# 导入 edge-tts
import edge_tts
import aiohttp


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "output"), exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml



class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []


def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}








# 语言映射
language_dict = {
    "中文": {
        "title": "EdgeRVC - 文本转语音与语音转换",
        "header": "## EdgeRVC",
        "disclaimer": "本软件作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件.",
        "section_title": "## 文本转语音 & 变声",
        "section_desc": "输入文本，使用 Edge TTS 生成语音，然后通过 RVC 进行变声处理",
        "input_mode": "输入方式",
        "text_input": "文本输入",
        "srt_input": "SRT文件导入",
        "text_placeholder": "请输入要转换为语音的文本",
        "srt_label": "上传SRT字幕文件",
        "srt_placeholder": "拖放文件至此处或点击上传",
        "voice_label": "选择语音",
        "speed_label": "语音速度",
        "refresh_button": "刷新音色列表",
        "voice_model_label": "变声音色",
        "speaker_id_label": "请选择说话人id",
        "pitch_label": "变调(整数, 半音数量, 升八度12降八度-12)",
        "f0_method_label": "选择音高提取算法",
        "index_label": "特征检索库文件路径(自动检测)",
        "index_rate_label": "检索特征占比",
        "resample_label": "后处理重采样至最终采样率，0为不进行重采样",
        "format_label": "导出文件格式",
        "save_dir_label": "保存目录",
        "save_dir_placeholder": "请输入保存目录路径",
        "protect_label": "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果",
        "generate_button": "生成语音并变声",
        "output_audio_label": "输出音频",
        "output_info_label": "输出信息",
        "delete_button": "删除音频",
        "error_no_text": "错误：请输入文本或上传SRT文件",
        "error_no_srt_text": "错误：SRT文件中没有找到文本",
        "error_voice_unavailable": "错误：语音 '{voice}' 不可用，请尝试其他语音",
        "error_connection": "错误：无法连接到语音合成服务，请检查网络连接或稍后重试",
        "error_tts_failed": "错误：语音合成失败 - {error}",
        "error_no_audio": "错误：无法生成语音文件",
        "save_success": "音频已保留在: {path}",
        "save_no_file": "没有可保留的音频文件",
        "delete_success": "音频已删除: {path}",
        "delete_failed": "删除失败: {error}",
        "delete_no_file": "没有可删除的音频文件"
    },
    "English": {
        "title": "EdgeRVC - Text-to-Speech and Voice Conversion",
        "header": "## EdgeRVC",
        "disclaimer": "The author of this software has no control over the software. Users of the software and those who distribute voices exported by the software are solely responsible. If you do not agree to this clause, you cannot use or reference any code or files in the software package.",
        "section_title": "## Text-to-Speech & Voice Conversion",
        "section_desc": "Enter text, generate speech using Edge TTS, then convert voice using RVC",
        "input_mode": "Input Mode",
        "text_input": "Text Input",
        "srt_input": "SRT File Import",
        "text_placeholder": "Please enter text to convert to speech",
        "srt_label": "Upload SRT Subtitle File",
        "srt_placeholder": "Drag and drop files here or click to upload",
        "voice_label": "Select Voice",
        "speed_label": "Speech Speed",
        "refresh_button": "Refresh Voice List",
        "voice_model_label": "Voice Model",
        "speaker_id_label": "Please select speaker ID",
        "pitch_label": "Pitch Shift (integer, number of semitones, +12 for octave up, -12 for octave down)",
        "f0_method_label": "Select Pitch Extraction Algorithm",
        "index_label": "Feature Index File Path (auto-detected)",
        "index_rate_label": "Feature Retrieval Ratio",
        "resample_label": "Post-processing resampling to final sample rate, 0 for no resampling",
        "format_label": "Export File Format",
        "save_dir_label": "Save Directory",
        "save_dir_placeholder": "Please enter save directory path",
        "protect_label": "Protect consonants and breathing sounds, prevent artifacts like electric tearing, 0.5 means disabled, lower values increase protection but may reduce index effect",
        "generate_button": "Generate Speech and Convert Voice",
        "output_audio_label": "Output Audio",
        "output_info_label": "Output Information",
        "delete_button": "Delete Audio",
        "error_no_text": "Error: Please enter text or upload SRT file",
        "error_no_srt_text": "Error: No text found in SRT file",
        "error_voice_unavailable": "Error: Voice '{voice}' is unavailable, please try another voice",
        "error_connection": "Error: Unable to connect to speech synthesis service, please check network connection or try again later",
        "error_tts_failed": "Error: Speech synthesis failed - {error}",
        "error_no_audio": "Error: Unable to generate audio file",
        "save_success": "Audio has been saved at: {path}",
        "save_no_file": "No audio file to save",
        "delete_success": "Audio has been deleted: {path}",
        "delete_failed": "Deletion failed: {error}",
        "delete_no_file": "No audio file to delete"
    }
}

# 语言切换函数
def update_language(language):
    lang = language_dict[language]
    return (
        gr.update(value=lang["header"]),
        gr.update(value=lang["disclaimer"]),
        gr.update(value=lang["section_title"]),
        gr.update(value=lang["section_desc"]),
        gr.update(label=lang["input_mode"], choices=[lang["text_input"], lang["srt_input"]], value=lang["text_input"]),
        gr.update(label=lang["text_input"], placeholder=lang["text_placeholder"]),
        gr.update(label=lang["srt_label"]),
        gr.update(label=lang["voice_label"]),
        gr.update(label=lang["speed_label"]),
        gr.update(value=lang["refresh_button"]),
        gr.update(label=lang["voice_model_label"]),
        gr.update(label=lang["speaker_id_label"]),
        gr.update(label=lang["pitch_label"]),
        gr.update(label=lang["f0_method_label"]),
        gr.update(label=lang["index_label"]),
        gr.update(label=lang["index_rate_label"]),
        gr.update(label=lang["resample_label"]),
        gr.update(label=lang["format_label"]),
        gr.update(label=lang["save_dir_label"], placeholder=lang["save_dir_placeholder"]),
        gr.update(label=lang["protect_label"]),
        gr.update(value=lang["generate_button"]),
        gr.update(label=lang["output_audio_label"]),
        gr.update(label=lang["output_info_label"]),
        gr.update(value=lang["delete_button"])
    )

with gr.Blocks(title="EdgeRVC") as app:
    # 语言选择
    with gr.Row():
        language = gr.Dropdown(
            label="Language / 语言",
            choices=["中文", "English"],
            value="中文",
            interactive=True
        )
    
    # UI组件
    header = gr.Markdown("## EdgeRVC")
    disclaimer = gr.Markdown(
        value="本软件作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件."
    )
    
    # 只保留文本转语音界面，不使用Tabs
    section_title = gr.Markdown("## 文本转语音 & 变声")
    section_desc = gr.Markdown("输入文本，使用 Edge TTS 生成语音，然后通过 RVC 进行变声处理")
    
    # 输入方式选择
    input_mode = gr.Radio(
        label="输入方式",
        choices=["文本输入", "SRT文件导入"],
        value="文本输入",
        interactive=True,
    )
    
    # 文本输入区域
    text_input = gr.Textbox(
        label="输入文本",
        placeholder="请输入要转换为语音的文本",
        lines=3,
        interactive=True,
    )
    
    # SRT文件上传区域
    srt_file = gr.File(
        label="上传SRT字幕文件",
        file_types=[".srt"],
        visible=False,
        interactive=True
    )
    
    with gr.Row():
        # 语音选择
        with gr.Column():
            voice = gr.Dropdown(
                label="选择语音",
                choices=[
                    "zh-CN-XiaoxiaoNeural",
                    "zh-CN-XiaoyiNeural",
                    "zh-CN-YunjianNeural",
                    "zh-CN-YunxiNeural",
                    "zh-CN-YunxiaNeural",
                    "zh-CN-YunyangNeural",
                    "zh-CN-liaoning-XiaobeiNeural",
                    "zh-CN-shaanxi-XiaoniNeural",
                    "zh-HK-HiuGaaiNeural",
                    "zh-HK-HiuMaanNeural",
                    "zh-HK-WanLungNeural",
                    "zh-TW-HsiaoChenNeural",
                    "zh-TW-HsiaoYuNeural",
                    "zh-TW-YunJheNeural",
                    "en-US-AnaNeural",
                    "en-US-AndrewMultilingualNeural",
                    "en-US-AndrewNeural",
                    "en-US-AriaNeural",
                    "en-US-AvaMultilingualNeural",
                    "en-US-AvaNeural",
                    "en-US-BrianMultilingualNeural",
                    "en-US-BrianNeural",
                    "en-US-ChristopherNeural",
                    "en-US-EmmaMultilingualNeural",
                    "en-US-EmmaNeural",
                    "en-US-EricNeural",
                    "en-US-GuyNeural",
                    "en-US-JennyNeural",
                    "en-US-MichelleNeural",
                    "en-US-RogerNeural",
                    "en-US-SteffanNeural",
                ],
                value="zh-CN-XiaoxiaoNeural",
                interactive=True,
            )
            
            # 语音速度调整
            speed = gr.Slider(
                label="语音速度",
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                interactive=True,
            )
    
    with gr.Row():
        refresh_button = gr.Button("刷新音色列表", variant="primary")
    
    with gr.Row():
        # 变声参数
        with gr.Column():
            tts_sid = gr.Dropdown(label="变声音色", choices=sorted(names))
            tts_spk_item = gr.Slider(
                minimum=0,
                maximum=2333,
                step=1,
                label="请选择说话人id",
                value=0,
                visible=True,
                interactive=True,
            )
            tts_vc_transform = gr.Number(
                label="变调(整数, 半音数量, 升八度12降八度-12)",
                value=0,
            )
            tts_f0method = gr.Radio(
                label="选择音高提取算法",
                choices=(
                    ["pm", "harvest", "crepe", "rmvpe"]
                    if config.dml == False
                    else ["pm", "harvest", "rmvpe"]
                ),
                value="rmvpe",
                interactive=True,
            )
        
        with gr.Column():
            tts_file_index2 = gr.Dropdown(
                label="特征检索库文件路径(自动检测)",
                choices=sorted(index_paths),
                interactive=True,
            )
            tts_index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label="检索特征占比",
                value=0.75,
                interactive=True,
            )
            tts_resample_sr = gr.Slider(
                minimum=0,
                maximum=48000,
                label="后处理重采样至最终采样率，0为不进行重采样",
                value=0,
                step=1,
                interactive=True,
            )
            tts_format = gr.Radio(
                label="导出文件格式",
                choices=["wav", "flac", "mp3", "m4a"],
                value="wav",
                interactive=True,
            )
            tts_save_dir = gr.Textbox(
                label="保存目录",
                value="./output",
                placeholder="请输入保存目录路径",
                interactive=True,
            )
    
    # 保护参数
    protect = gr.Slider(
        minimum=0,
        maximum=0.5,
        label="保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果",
        value=0.33,
        step=0.01,
        interactive=True,
        visible=False
    )
    
    # 生成并变声按钮
    with gr.Row():
        tts_button = gr.Button("生成语音并变声", variant="primary")
    
    # 输出区域
    with gr.Row():
        tts_output_audio = gr.Audio(label="输出音频", type="filepath", interactive=False, scale=2)
    with gr.Row():
        tts_output_info = gr.Textbox(label="输出信息", lines=4, interactive=False)
    with gr.Row():
        delete_button = gr.Button("删除音频", variant="stop")
    
    # 绑定语言切换事件
    language.change(
        fn=update_language,
        inputs=[language],
        outputs=[
            header,
            disclaimer,
            section_title,
            section_desc,
            input_mode,
            text_input,
            srt_file,
            voice,
            speed,
            refresh_button,
            tts_sid,
            tts_spk_item,
            tts_vc_transform,
            tts_f0method,
            tts_file_index2,
            tts_index_rate,
            tts_resample_sr,
            tts_format,
            tts_save_dir,
            protect,
            tts_button,
            tts_output_audio,
            tts_output_info,
            delete_button
        ]
    )
    
    # 绑定音色选择事件
    tts_sid.change(
        fn=vc.get_vc,
        inputs=[tts_sid, protect, protect],
        outputs=[tts_spk_item, protect, protect, tts_file_index2, tts_file_index2],
    )
    
    
    # 保存目录变量，用于保存和删除操作
    current_save_dir = "./output"
    current_output_file = ""
    
    # 解析SRT文件的函数
    def parse_srt(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # 分割字幕条目
        entries = content.strip().split('\n\n')
        text_parts = []
        
        for entry in entries:
            lines = entry.strip().split('\n')
            # 跳过序号和时间戳，只提取文本
            for line in lines[2:]:  # 从第三行开始是文本
                if line.strip():
                    text_parts.append(line.strip())
        
        # 将所有文本合并成一个长文本
        return ' '.join(text_parts)
    
    # 文本转语音并变声的函数
    async def text_to_speech_and_convert(input_mode, text, srt_file, voice, speed, spk_item, f0_up_key, f0_method, file_index2, index_rate, resample_sr, format1, save_dir, language):
        global current_save_dir, current_output_file
        try:
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 保存当前保存目录
            current_save_dir = save_dir

            # 根据输入方式获取文本
            lang = language_dict[language]
            if input_mode == lang["srt_input"] and srt_file:
                # 解析SRT文件
                text = parse_srt(srt_file.name)
                if not text:
                    return lang["error_no_srt_text"], None
            elif not text:
                return lang["error_no_text"], None
            
            
            # 生成临时文件路径
            temp_tts_file = os.path.join(save_dir, f"temp_tts_{int(time.time())}.wav")
            
            # 使用 Edge TTS 生成语音，设置速度
            # 将speed转换为百分比形式，1.0对应正常速度，0.5对应-50%，2.0对应+100%
            # Edge TTS的rate参数需要带有正负号，例如+10%或-50%
            rate_value = (speed - 1.0) * 100
            if rate_value >= 0:
                rate = f"+{rate_value:.0f}%"
            else:
                rate = f"{rate_value:.0f}%"
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            try:
                with open(temp_tts_file, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
            except edge_tts.exceptions.NoAudioReceived:
                return f"错误：语音 '{voice}' 不可用，请尝试其他语音", None
            except aiohttp.ClientConnectorDNSError:
                return "错误：无法连接到语音合成服务，请检查网络连接或稍后重试", None
            except Exception as e:
                return f"错误：语音合成失败 - {str(e)}", None
            
            # 检查生成的文件是否存在
            if not os.path.exists(temp_tts_file):
                return "错误：无法生成语音文件", None
            
            # 使用 RVC 进行变声
            info, audio_path = vc.vc_single(
                spk_item,
                temp_tts_file,
                f0_up_key,
                None,
                f0_method,
                "",
                file_index2,
                index_rate,
                3,  # filter_radius
                resample_sr,
                0.25,  # rms_mix_rate
                0.33,  # protect
                save_dir,
                format1,
            )
            
            # 检查返回值格式
            if isinstance(audio_path, tuple):
                # 如果是元组，说明出错了
                return info, None
            
            # 保存当前输出文件路径
            current_output_file = audio_path
            
            # 清理临时文件
            if os.path.exists(temp_tts_file):
                os.remove(temp_tts_file)
            
            return info, audio_path
        except Exception as e:
            logger.error(f"Error in text_to_speech_and_convert: {e}")
            import traceback
            error_info = traceback.format_exc()
            logger.error(error_info)
            return f"错误：{str(e)}", None
    
    # 保留音频按钮功能
    def save_audio(audio_path):
        global current_output_file
        if current_output_file and os.path.exists(current_output_file):
            # 音频已经保存，只需要显示成功信息
            return f"音频已保留在: {current_output_file}"
        return "没有可保留的音频文件"
    
    # 删除音频按钮功能
    def delete_audio(audio_path):
        global current_output_file
        if current_output_file and os.path.exists(current_output_file):
            try:
                # 保存当前输出文件路径用于返回消息
                file_to_delete = current_output_file
                os.remove(file_to_delete)
                # 清空当前输出文件路径
                current_output_file = ""
                return f"音频已删除: {file_to_delete}"
            except Exception as e:
                return f"删除失败: {str(e)}"
        return "没有可删除的音频文件"
    
    # 绑定生成按钮事件
    tts_button.click(
        fn=text_to_speech_and_convert,
        inputs=[
            input_mode,
            text_input,
            srt_file,
            voice,
            speed,
            tts_spk_item,
            tts_vc_transform,
            tts_f0method,
            tts_file_index2,
            tts_index_rate,
            tts_resample_sr,
            tts_format,
            tts_save_dir,
            language,
        ],
        outputs=[tts_output_info, tts_output_audio],
    )
    
    # 刷新音色列表函数
    def refresh_voice_list():
        # 重新读取weights目录下的.pth文件
        names = []
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)
        
        # 重新读取索引路径
        index_paths = []
        def lookup_indices(index_root):
            nonlocal index_paths
            for root, dirs, files in os.walk(index_root, topdown=False):
                for name in files:
                    if name.endswith(".index") and "trained" not in name:
                        index_paths.append("%s/%s" % (root, name))
        
        lookup_indices(index_root)
        lookup_indices(outside_index_root)
        
        return {
            "choices": sorted(names),
            "__type__": "update"
        }, {
            "choices": sorted(index_paths),
            "__type__": "update"
        }
    
    # 绑定刷新按钮事件
    refresh_button.click(
        fn=refresh_voice_list,
        inputs=[],
        outputs=[tts_sid, tts_file_index2]
    )
    
    # 输入方式切换函数
    def update_input_visibility(mode, language):
        lang = language_dict[language]
        if mode == lang["text_input"]:
            return (
                gr.update(visible=True),
                gr.update(visible=False)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True)
            )
    
    # 绑定输入方式切换事件
    input_mode.change(
        fn=update_input_visibility,
        inputs=[input_mode, language],
        outputs=[text_input, srt_file]
    )

    # 绑定删除按钮事件
    delete_button.click(
        fn=delete_audio,
        inputs=[tts_output_audio],
        outputs=[tts_output_info]
    )

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
