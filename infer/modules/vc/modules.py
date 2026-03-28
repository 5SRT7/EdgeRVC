import traceback
import logging
import os

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO
import tempfile

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        save_dir=None,
        format1="wav",
    ):
        if input_audio_path is None:
            return "You need to upload an audio", (None, None)
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                try:
                    logger.info("Loading Hubert model in vc_single...")
                    self.hubert_model = load_hubert(self.config)
                    logger.info("Hubert model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading Hubert model in vc_single: {e}")
                    logger.error(traceback.format_exc())
                    raise

            # 处理索引文件
            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉
            
            logger.info(f"Using index file: {file_index}")

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            # 正确处理采样率
            if resample_sr >= 16000 and resample_sr != self.tgt_sr:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            
            # 确保采样率是有效的
            if not isinstance(tgt_sr, (int, float)) or tgt_sr <= 0:
                logger.error(f"Invalid sample rate: {tgt_sr}")
                tgt_sr = self.tgt_sr
            
            logger.info(f"Using sample rate: {tgt_sr}")
            index_info = (
                "Index:\n%s." % file_index
                if file_index and os.path.exists(file_index)
                else "Index not used."
            )
            # 确保音频数据不为空
            if audio_opt.size == 0:
                logger.error("Generated audio is empty")
                return "Error: Generated audio is empty", (None, None)
            
            # 确保音频数据范围在 -1 到 1 之间
            audio_min = np.min(audio_opt)
            audio_max = np.max(audio_opt)
            if audio_min < -1 or audio_max > 1:
                logger.warning(f"Audio data out of range: min={audio_min}, max={audio_max}")
                # 归一化音频数据
                if audio_max - audio_min > 0:
                    audio_opt = 2 * (audio_opt - audio_min) / (audio_max - audio_min) - 1
            
            # 确保音频数据形状为一维以兼容 Gradio
            if len(audio_opt.shape) > 1:
                audio_opt = audio_opt.flatten()
                logger.info(f"Flattened audio to: shape={audio_opt.shape}")
            
            # 确保音频数据类型为 float32
            if audio_opt.dtype != np.float32:
                audio_opt = audio_opt.astype(np.float32)
                logger.info(f"Converted audio to float32: dtype={audio_opt.dtype}")
            
            # 确保音频数据范围在 -1 到 1 之间
            audio_min = np.min(audio_opt)
            audio_max = np.max(audio_opt)
            if audio_min < -1 or audio_max > 1:
                logger.warning(f"Audio data out of range: min={audio_min}, max={audio_max}")
                # 归一化音频数据
                if audio_max - audio_min > 0:
                    audio_opt = 2 * (audio_opt - audio_min) / (audio_max - audio_min) - 1
            
            logger.info(f"Returning audio: shape={audio_opt.shape}, dtype={audio_opt.dtype}, min={np.min(audio_opt)}, max={np.max(audio_opt)}")
            
            # 创建唯一的文件名，使用输入音频的文件名加上时间戳
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 提取输入文件的基本名称（不含扩展名）
            input_basename = os.path.splitext(os.path.basename(input_audio_path))[0]
            # 构建输出文件路径
            output_file_path = ""
            
            # 如果提供了保存目录，则保存音频到文件
            if save_dir:
                logger.info("Starting to save audio to file...")
                # 确保保存目录存在
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Save directory created successfully: {os.path.exists(save_dir)}")
                
                # 创建输出文件名
                output_file_path = os.path.abspath(os.path.join(save_dir, f"{input_basename}_{timestamp}.{format1}"))
                logger.info(f"Created output file: {output_file_path}")
                
                # 保存音频到文件
                logger.info(f"Saving audio to file: {output_file_path}")
                logger.info(f"Audio shape: {audio_opt.shape}, dtype: {audio_opt.dtype}, sample rate: {tgt_sr}")
                if format1 in ["wav", "flac"]:
                    sf.write(output_file_path, audio_opt, tgt_sr)
                else:
                    from io import BytesIO
                    with BytesIO() as wavf:
                        sf.write(wavf, audio_opt, tgt_sr, format="wav")
                        wavf.seek(0, 0)
                        with open(output_file_path, "wb") as outf:
                            wav2(wavf, outf, format1)
                logger.info(f"Saved audio to file: {output_file_path}")
                
                # 验证文件是否存在
                if os.path.exists(output_file_path):
                    file_size = os.path.getsize(output_file_path)
                    logger.info(f"Audio file exists and has size: {file_size} bytes")
                else:
                    logger.error(f"Failed to create audio file: {output_file_path}")
            
            # 添加索引文件使用情况的信息
            index_usage_info = f"Index file: {file_index}\n"
            if not file_index:
                index_usage_info = "Index file: None (no index file selected)\n"
            elif not os.path.exists(file_index):
                index_usage_info = f"Index file: {file_index} (file not found)\n"
            else:
                index_usage_info = f"Index file: {file_index} (loaded, but search disabled due to compatibility issues)\n"
            
            # 返回信息和文件路径
            info = "Success.\n%s%sTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs." % (index_usage_info, index_info, *times)
            logger.info(f"Returning info: {info}")
            logger.info(f"Returning file path: {output_file_path}")
            return info, output_file_path
        except Exception as e:
            info = traceback.format_exc()
            logger.error(f"Error in vc_single: {e}")
            logger.error(f"Traceback: {info}")
            logger.error(f"Input parameters: sid={sid}, input_audio_path={input_audio_path}, f0_up_key={f0_up_key}, f0_method={f0_method}, file_index={file_index}, file_index2={file_index2}, index_rate={index_rate}, filter_radius={filter_radius}, resample_sr={resample_sr}, rms_mix_rate={rms_mix_rate}, protect={protect}, save_dir={save_dir}")
            return info, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            
            # 获取要处理的音频文件列表
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            
            infos = []
            for path in paths:
                try:
                    info, opt = self.vc_single(
                        sid,
                        path,
                        f0_up_key,
                        None,
                        f0_method,
                        file_index,
                        file_index2,
                        index_rate,
                        filter_radius,
                        resample_sr,
                        rms_mix_rate,
                        protect,
                        opt_root,
                        format1,
                    )
                    if "Success" in info:
                        try:
                            tgt_sr, audio_opt = opt
                            if format1 in ["wav", "flac"]:
                                sf.write(
                                    "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                                    audio_opt,
                                    tgt_sr,
                                )
                            else:
                                output_path = "%s/%s.%s" % (
                                    opt_root,
                                    os.path.basename(path),
                                    format1,
                                )
                                with BytesIO() as wavf:
                                    sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                    wavf.seek(0, 0)
                                    with open(output_path, "wb") as outf:
                                        wav2(wavf, outf, format1)
                        except:
                            info += traceback.format_exc()
                    infos.append("%s->%s" % (os.path.basename(path), info))
                    yield "\n".join(infos)
                except Exception as e:
                    logger.error(f"Error processing file {path}: {e}")
                    logger.error(traceback.format_exc())
                    infos.append("%s->Error: %s" % (os.path.basename(path), str(e)))
                    yield "\n".join(infos)
            yield "\n".join(infos)
        except Exception as e:
            logger.error(f"Error in vc_multi: {e}")
            logger.error(traceback.format_exc())
            yield traceback.format_exc()
