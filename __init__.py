from .nodes import (
    QwenPanoramaPrompt,
    QwenPanoramaLatent,
    QwenPanorama360Viewer,
    QwenPanoramaExport,
    QwenPanoramaUpscale,
    QwenPanoramaModelUpscale,
    QwenPanoramaSeamExtract,
    QwenPanoramaSeamPaste,
    QwenPanoramaRoll,
    QwenPanoramaSeamRouter,
)

NODE_CLASS_MAPPINGS = {
    "QwenPanoramaPrompt": QwenPanoramaPrompt,
    "QwenPanoramaLatent": QwenPanoramaLatent,
    "QwenPanorama360Viewer": QwenPanorama360Viewer,
    "QwenPanoramaExport": QwenPanoramaExport,
    "QwenPanoramaUpscale": QwenPanoramaUpscale,
    "QwenPanoramaModelUpscale": QwenPanoramaModelUpscale,
    "QwenPanoramaSeamExtract": QwenPanoramaSeamExtract,
    "QwenPanoramaSeamPaste": QwenPanoramaSeamPaste,
    "QwenPanoramaRoll": QwenPanoramaRoll,
    "QwenPanoramaSeamRouter": QwenPanoramaSeamRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPanoramaPrompt": "Qwen 360 Panorama Prompt",
    "QwenPanoramaLatent": "Qwen 360 Panorama Latent (2:1)",
    "QwenPanorama360Viewer": "Qwen 360 VR Panorama Viewer",
    "QwenPanoramaExport": "Qwen 360 Panorama Export",
    "QwenPanoramaUpscale": "Qwen 360 Panorama Upscale (2K/4K)",
    "QwenPanoramaModelUpscale": "Qwen 360 Seamless Model Upscale (RealESRGAN)",
    "QwenPanoramaSeamExtract": "Qwen 360 Panorama Seam Extract (Step 1)",
    "QwenPanoramaSeamPaste": "Qwen 360 Panorama Seam Restore (Step 2)",
    "QwenPanoramaRoll": "Qwen 360 Panorama Horizontal Roll",
    "QwenPanoramaSeamRouter": "Qwen 360 Panorama Seam Router (2-Phase)",
}

# Auto-register Viggle Turbo nodes if present for standalone portability
try:
    from .viggle_turbo import ViggleTurboLora, ViggleTurboSigmas
    NODE_CLASS_MAPPINGS["ViggleTurboLora"] = ViggleTurboLora
    NODE_CLASS_MAPPINGS["ViggleTurboSigmas"] = ViggleTurboSigmas
    NODE_DISPLAY_NAME_MAPPINGS["ViggleTurboLora"] = "Viggle Turbo Lora (Bundled)"
    NODE_DISPLAY_NAME_MAPPINGS["ViggleTurboSigmas"] = "Viggle Turbo Sigmas (Bundled)"
except Exception as e:
    pass

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
