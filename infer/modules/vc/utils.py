import os
import torch
import logging

logger = logging.getLogger(__name__)

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    try:
        logger.info("Loading Hubert model...")
        # 确保使用 CPU 加载 Hubert 模型
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Monkey patch torch.load to use weights_only=False by default
        original_torch_load = torch.load
        
        def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
            # 导入 pickle 模块
            import pickle
            # 如果没有指定 pickle_module，则使用默认的 pickle
            if pickle_module is None:
                pickle_module = pickle
            # 如果没有指定 weights_only 参数，则默认设置为 False
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(f, map_location, pickle_module, **kwargs)
        
        # 应用 monkey patch
        torch.load = patched_torch_load
        logger.info("Applied monkey patch to torch.load")
        
        # 加载模型
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
        logger.info("Model loaded successfully")
        
        hubert_model = models[0]
        logger.info("Moving model to device...")
        hubert_model = hubert_model.to(device)
        logger.info("Model moved to device successfully")
        
        if config.is_half:
            logger.info("Converting model to half precision...")
            hubert_model = hubert_model.half()
        else:
            logger.info("Converting model to float precision...")
            hubert_model = hubert_model.float()
        
        logger.info("Model conversion completed")
        return hubert_model.eval()
    except Exception as e:
        logger.error(f"Error loading Hubert model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
