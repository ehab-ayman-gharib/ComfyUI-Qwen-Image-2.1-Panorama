# 🌐 ComfyUI-Qwen-Panorama

Transform any single 2D photo into a seamless **360° Equirectangular VR Panorama** (or synthesize completely from scratch via **Text-to-Panorama**) using **Qwen-Image 2.1** and the specialized **360 Panorama Maker LoRA** (`pano360_qwen21_edit_v1.safetensors` by Gogodr / Poly Haven). Includes an in-canvas interactive WebGL 360° VR viewer, Google Photo Sphere XMP metadata injection, native 2K spherical generation, seamless 4K upscaler, and dual execution modes (Turbo ~20s vs. Full Quality 2K).

Tested and optimized for **8GB VRAM GPUs** (RTX 4070 Laptop) — stays strictly under **7.0 GB peak VRAM**.

<p align="center">
  <a href="docs/showcase_16x9.mp4">
    <img src="docs/showcase.gif" alt="Qwen 360 Panorama Spatial Expansion Showcase" width="100%">
  </a>
  <br>
  <em>🌐 Spatial expansion demo (2D portrait seed input &rarr; seamless 360° VR world). <b>Click image to open full 1080p video</b>.</em>
</p>

---

## ✨ Features

- **Dual Modality: Image-to-Pano & Text-to-Pano**:
  - **🖼️ Image-to-Panorama (I2I)**: Supply a single 2D reference photo; the model anchors it at center ($0^\circ$ yaw) and invents the surrounding 360° environment matching exact lighting and atmosphere.
  - **✍️ Pure Text-to-Panorama (T2I)**: Generate rich, photorealistic 360° VR worlds completely from text prompts without any input images (simply bypass `Load Reference Photo` with `Ctrl + M`).
- **360 Panorama LoRA Spherical Backbone**: Powered by Gogodr's `pano360_qwen21_edit_v1.safetensors` (trained on Poly Haven CC0 HDRIs). Injects spherical projection geometry into Qwen-Image 2.1 to generate seamless 360° equirectangular panoramas with continuous horizons, seamless 360° boundaries, and consistent lighting.
- **In-Canvas Interactive 360° VR Viewer**:
  - Hardware-accelerated pure WebGL equirectangular shader (0 KB external libraries).
  - Smooth drag-to-look (yaw/pitch) with zero gimbal lock.
  - Mouse-wheel FOV zoom (40° to 120°).
  - 1-click Fullscreen VR preview.
- **Qwen 360 Panorama Latent (2:1)**: Built-in custom latent generator ensuring exact 2:1 equirectangular spherical aspect ratios and 64-channel initialization for Qwen-Image 2.1 DiT + Wan 2.2 VAE:
  - **Native Best Quality is 2K (`2048×1024`)**: Recommended for production quality, delivering razor-sharp VR skyboxes and fine landscape textures directly in latent space.
  - **Fast Draft Presets**: `1472×736`, `1536×768`, or `1024×512` for ultra-fast iteration and tight 8GB VRAM budgets.
- **Seamless 2K & 4K VR Upscaling**: `QwenPanoramaUpscale` with 5% circular toroidal boundary wrapping (`seamless_boundary: true`) upscales native 2K panoramas to **4K Ultra VR (`4096×2048`)** (or draft renders to 2K) in seconds with zero vertical boundary seams.
- **Dual Pipeline Support**:
  - **⚡ Fast Turbo Mode (~20–30s)**: 9 steps via Viggle Turbo v0.3 LoRA & custom refined sigmas (7 turbo + 2 high-precision detail refinement steps).
  - **💎 Full Quality Mode (< 2 min)**: 25 steps via native Euler sampler for pristine 2K VR production skyboxes.
- **Smart 360 Export**:
  - Standard Equirectangular PNG / WebP.
  - Automatic **Google Photo Sphere XMP metadata injection** into JPEG so Facebook, Google Photos, and VR headsets auto-detect and render it as an interactive 360 sphere.
  - Optional rotating 360° horizontal flythrough MP4 video encoded via FFmpeg.

---

## 📦 Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ehab-ayman-gharib/ComfyUI-Qwen-Image-2.1-Panorama.git
pip install -r ComfyUI-Qwen-Image-2.1-Panorama/requirements.txt
```

2. **Prerequisites & Dependencies**:
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**: Required to load the quantized `Q5_K_M.gguf` DiT model (search `GGUF` in ComfyUI Manager or install via git). *(If you use unquantized bf16 safetensors instead, simply swap `UnetLoaderGGUF` for ComfyUI's native `UNETLoader`)*.
- **FFmpeg** *(Optional)*: Required only if you toggle `export_rotating_video` in `Qwen 360 Panorama Export` to generate MP4 flythrough videos.
- **Viggle Turbo Nodes**: Already **bundled inside this repository** ([viggle_turbo.py](viggle_turbo.py)) — no need to install a separate custom node!

3. Restart ComfyUI.

---

## 📥 Required Models

| Role | File Name | Size | Storage Directory | Download Link |
| :--- | :--- | :--- | :--- | :--- |
| **Diffusion (DiT)** | `qwen_image_2.1_Q5_K_M.gguf` | 5.8 GB | `models/unet/` | [Hugging Face](https://huggingface.co/AlperKTS/Qwen-Image-2.1-GGUF) |
| **360 Pano LoRA (Essential)** | `pano360_qwen21_edit_v1.safetensors` | 152 MB | `models/loras/` | [Hugging Face](https://huggingface.co/Gogodr/qwen-image-2.1-edit-pano360-lora) |
| **Turbo LoRA** *(Optional)* | `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors` | 315 MB | `models/loras/` | [Hugging Face](https://huggingface.co/ViggleAI/Qwen-Image-2.1-viggle-turbo) |
| **Text Encoder** | `qwen3vl_8b_w4a8.safetensors` | 5.8 GB | `models/text_encoders/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| **VAE** | `qwen_image_2.1_vae_bf16.safetensors` | 496 MB | `models/vae/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| **2× Upscaler** *(Optional)* | `RealESRGAN_x2plus.pth` | 64 MB | `models/upscale_models/` | [GitHub](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| **4× Upscaler (RealESRGAN)** *(Optional)* | `RealESRGAN_x4plus.pth` | 64 MB | `models/upscale_models/` | [GitHub](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| **4× Upscaler (UltraSharp)** *(Recommended 8K)* | `4x-UltraSharp.pth` | 67 MB | `models/upscale_models/` | [Hugging Face](https://huggingface.co/lokcx/4x-Ultrasharp) |

---

## 🚀 Workflows Included

All production-ready workflow JSON files are located in `workflows/` and can be loaded directly in ComfyUI:

### 🌟 Primary Generation Workflows (End-to-End 2K Generation &rarr; 4K Bilateral Upscale)
Both main workflows now feature our **winning 4K super-resolution pipeline** built directly into the flow: **Generation &rarr; 5% Toroidal Wrap &rarr; RealESRGAN 2× &rarr; Subtle Bilateral Filter &rarr; Crop &rarr; Final 4K**:
1. **`Qwen_Image_2.1_Viggle_Turbo_Panorama_360.json`** *(Recommended for Fast Production)*:
   - 9-step dual-LoRA stack (Pano360 LoRA + Viggle Turbo v0.3) + custom refined sigmas.
   - Generates a 2K equirectangular world and seamlessly upscales to **4K Ultra VR (`4096×2048`)** with gentle bilateral noise suppression in **~22s total** on 8GB VRAM.
   - Includes interactive **Image Comparer (rgthree)** hover slider to dynamically compare the raw 2K render vs. the final cleaned 4K panorama in real time.
2. **`Qwen_Image_2.1_Full_Quality_Panorama_360.json`** *(Maximum Architectural Precision)*:
   - 25-step native Euler sampler with Pano360 LoRA.
   - Deep structural geometry and texture details, upscaled and cleaned to **4K Ultra VR (`4096×2048`)** in < 2 minutes.
   - Includes interactive **Image Comparer (rgthree)** hover slider to dynamically compare the raw 2K render vs. the final cleaned 4K panorama in real time.

---

### 🔬 Standalone & Experimental Upscalers (in `workflows/Upscaling/`)
For upscaling existing panoramas, A/B benchmarking, or experimenting with extreme 8K resolutions:
1. **`Qwen_Image_2.1_4K_Panorama_Bilateral_Upscaler.json`** *(⭐ Best Overall Result)*:
   - Standalone RealESRGAN 2× + 360-aware Bilateral Filter with side-by-side A/B preview windows.
   - Eliminates neural speckles and dotted noise while keeping sharp architectural edges, horizons, and silhouettes 100% crisp.
2. **`Qwen_Image_2.1_4K_Supersampled_Panorama_Upscaler.json`**:
   - 4K Supersampling Anti-Aliasing pipeline ($2\text{K} \to 8\text{K} \text{ via RealESRGAN 4×} \to 4\text{K}$).
   - Downsamples using selectable `area` (box averaging), `bicubic`, or `lanczos` filters.
3. **`Qwen_Image_2.1_8K_Panorama_UltraSharp_Upscaler.json`**:
   - Extreme 4× high-frequency photographic super-resolution via `4x-UltraSharp.pth` (2K $\to$ **8K Ultra VR / 33.5 MP**).
4. **`Qwen_Image_2.1_8K_Panorama_Upscaler.json`**:
   - Standard 4× neural super-resolution via `RealESRGAN_x4plus.pth` (2K $\to$ **8K Cinema VR / 33.5 MP**).
5. **`Qwen_Image_2.1_4K_Panorama_Upscaler.json`**:
   - Standard 2× neural super-resolution without post-filtering.
6. **`Qwen_Image_2.1_4K_Panorama_Bicubic_Upscaler.json`**:
   - Zero-latency (<0.05s, 0 GB VRAM) mathematical baseline upscaler for direct benchmarking.

Simply drag and drop any JSON file directly into your ComfyUI window!

### 💡 How to Generate from Text Only (Pure Text-to-Panorama)
Both workflows support pure Text-to-Panorama generation out-of-the-box without needing any separate workflow:
1. In ComfyUI, click on the **`Load Reference Photo`** node and press **`Ctrl + M`** to **Bypass / Mute** it (or disconnect the wire running from `Load Reference Photo` into `Text Encode Qwen Image 2.1`).
2. In **`Qwen 360 Panorama Prompt`**, write your desired spherical environment in `scene_description` (e.g. *"A futuristic cyberpunk city skyline under a neon rainstorm, wet asphalt reflections below, flying vehicle streaks above"*).
3. Click **Queue Prompt**: With the image input muted, `Text Encode Qwen Image 2.1` receives 0 reference images and automatically runs in pure text mode, synthesizing a pristine 360° VR world from scratch!

---

## 🧩 Custom Nodes Reference

All custom nodes are categorized under `QwenPanorama`:

| Node Name | Category | Primary Function | Key Parameters |
| :--- | :--- | :--- | :--- |
| **Qwen 360 Panorama Prompt** | `QwenPanorama` | Constructs formatted 360° trigger prompts with lighting and environment modifiers. Defaults: `lighting: default` and `environment_type: custom` for unconstrained scene prompting. | `scene_description`, `lighting` (default: "default"), `environment_type` (default: "custom") |
| **Qwen 360 Panorama Latent (2:1)** | `QwenPanorama` | Generates 64-channel empty latents snapped to exact 2:1 spherical dimensions (multiples of 64). Recommended **Best Quality setting is 2K (`2048×1024`)** for pristine VR fidelity; draft presets (`1472×736`, `1024×512`) also included. | `resolution_preset` (default: 2048×1024 2K), `custom_width`, `batch_size` |
| **Qwen 360 VR Panorama Viewer** | `QwenPanorama` | Static in-canvas WebGL 360° viewer (strictly static unless actively dragging with mouse). Supports smooth yaw/pitch dragging and mouse-wheel FOV zoom. | `images`, `default_fov`, `auto_rotate` (default: false / static) |
| **Qwen 360 Panorama Export** | `QwenPanorama` | Exports with Google Photo Sphere XMP metadata and optional rotating MP4 flythrough video. | `format`, `export_rotating_video`, `video_duration_seconds` |
| **Qwen 360 Panorama Upscale (2K/4K)** | `QwenPanorama` | Seamless 2:1 upscaler with 5% toroidal boundary wrapping to prevent vertical seam artifacts. Upscales production 2K renders to **4K Ultra VR (`4096×2048`)** or draft renders to **2K (`2048×1024`)**. | `target_resolution` (2K / 4K / Custom), `seamless_boundary`, `upscale_method` |
| **Qwen 360 Seamless Model Upscale (RealESRGAN)** | `QwenPanorama` | All-in-one neural super-resolution node (`RealESRGAN_x2plus`, `4x-UltraSharp`). Applies 5% toroidal margin wrapping, runs the model full-frame or tiled, and crops out padding for seamless 4K/8K spherical output. Optionally includes built-in bilateral filtering. | `upscale_model`, `seamless_boundary`, `wrap_percent`, `processing_mode`, `enable_bilateral_filter` |
| **Qwen 360 Panorama Bilateral Filter (Subtle Denoise)** | `QwenPanorama` | Subtle edge-preserving bilateral cleanup filter for 360° panoramas. Eliminates neural upscaler speckles and dotted artifacts while maintaining sharp edges. Features 5% toroidal boundary wrapping for zero seam artifacts. | `d` (default: 5), `sigma_color` (10-20), `sigma_space` (3-5), `blend_opacity`, `seamless_boundary` |
| **Qwen 360 Panorama Downsample (Supersample 4K)** | `QwenPanorama` | Dedicated 360° anti-aliasing reconstruction filter with 5% toroidal seam protection. Takes 8K neural upscaled panoramas and smoothly downsamples to **4K** or **2K**, crushing single-pixel dotted noise and grain while preserving sharp edges. | `downsample_method` (`area` [best for noise removal], `bicubic`, `lanczos`), `target_resolution`, `seamless_boundary` |
| **Qwen 360 Panorama Seam Extract (Step 1)** | `QwenPanorama` | Rolls by 50% to place seam at center and crops narrow vertical strip for fast inpainting. | `image`, `crop_width` (default: 512px) |
| **Qwen 360 Panorama Seam Restore (Step 2)** | `QwenPanorama` | Blends healed crop back using cosine feathered alpha masking and unrolls to original orientation. | `healed_crop`, `seam_context`, `feather_width` |
| **Qwen 360 Panorama Horizontal Roll** | `QwenPanorama` | Horizontally shifts equirectangular image and synced mask along the cylindrical axis. | `mode` (Roll 50% / Unroll / Custom), `mask` |
| **Qwen 360 Panorama Seam Router (2-Phase)** | `QwenPanorama` | Automatically routes between centering/saving to `input/` (Phase 1) and inpainting (Phase 2). | `image`, `mask`, `mode` |

---

## 🤝 Credits & Acknowledgments

- **Gogodr**: Author of [360 Panorama Maker for Qwen-Image 2.1 Edit](https://civitai.com/models/2976898/360-panorama-maker-qwen-image-21-edit) and Poly Haven dataset curation.
- **Qwen Team (Alibaba)**: [Qwen-Image 2.1](https://github.com/QwenLM/Qwen-Image) foundation model.
- **Poly Haven**: CC0 HDRIs and panoramic training environments.

---

## 📄 License

This repository is licensed under the **Apache License 2.0**.
