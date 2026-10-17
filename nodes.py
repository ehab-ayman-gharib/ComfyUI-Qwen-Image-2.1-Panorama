import os
import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths
import subprocess
import comfy.model_management
import comfy.utils

class QwenPanoramaPrompt:
    """
    Constructs the exact equirectangular 360 trigger prompt required by the
    Gogodr/qwen-image-2.1-edit-pano360-lora model for Qwen-Image 2.1.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "A serene alpine lake surrounded by evergreen pine forests and snow-capped granite mountains, crystal clear water reflections, dramatic horizon line.",
                    "tooltip": "Detailed description of the environment, focal subjects, architecture, and horizon to expand into 360°."
                }),
                "lighting": ([
                    "default",
                    "bright midday sun with clear blue sky",
                    "warm golden hour sunset with long shadows",
                    "blue hour twilight just after dusk",
                    "night starry sky with milky way and celestial glow",
                    "dramatic overcast storm clouds with moody lighting",
                    "foggy misty morning with diffused dawn light",
                    "indoor warm architectural ambient lighting",
                    "cyberpunk neon glow with reflections"
                ], {
                    "default": "default",
                    "tooltip": "Atmospheric lighting to enforce across the full 360° spherical horizon."
                }),
                "environment_type": ([
                    "outdoor natural landscape",
                    "indoor architectural interior",
                    "city street / urban cityscape",
                    "sci-fi / fantasy realm",
                    "aerial cliffside vista",
                    "custom"
                ], {
                    "default": "custom",
                    "tooltip": "Broad environment setting for the spherical scene."
                }),
            },
            "optional": {
                "custom_details": ("STRING", {
                    "multiline": False,
                    "default": "high fidelity equirectangular projection, seamless 360 horizon.",
                    "tooltip": "Additional technical or stylistic prompt keywords appended to the positive prompt."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, distorted poles, broken seams, disfigured architecture, seams down the middle.",
                    "tooltip": "Negative prompt to suppress stitching seams, pole distortions, or blurriness."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "display_info")
    OUTPUT_TOOLTIPS = (
        "Positive prompt formatted with 360 equirectangular trigger syntax.",
        "Negative prompt passed downstream to CLIP conditioning.",
        "Summary text containing character count and configuration info."
    )
    FUNCTION = "build_prompt"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Constructs the equirectangular 360° trigger prompt required by the Pano360 LoRA, merging scene description, lighting conditions, and environment type."

    def build_prompt(self, scene_description, lighting, environment_type, custom_details="", negative_prompt=""):
        # Auto clean PyTorch CUDA cache to prevent shared memory paging on 8GB VRAM
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:
            pass

        parts = []
        if scene_description.strip():
            parts.append(scene_description.strip())
        if lighting != "default":
            parts.append(lighting)
        if environment_type != "custom":
            parts.append(environment_type)
        if custom_details.strip():
            parts.append(custom_details.strip())

        full_scene = ", ".join(parts)
        desc_lower = scene_description.strip().lower()
        if desc_lower.startswith("transform") or desc_lower.startswith("an equirectangular") or desc_lower.startswith("equirectangular"):
            full_positive = full_scene
        else:
            full_positive = f"Transform this set of images into an equirectangular 360 panorama. Scene: {full_scene}."

        display_info = f"Prompt generated for Qwen 360 Panorama LoRA.\nLength: {len(full_positive)} chars"
        return (full_positive, negative_prompt, display_info)


class QwenPanoramaLatent:
    """
    Enforces exact 2:1 equirectangular aspect ratios (multiples of 32/64)
    necessary for proper spherical projection and WanVAE decoding.
    Defaults to 2048x1024 (2K VR Production) for crisp non-pixelated spherical detail.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_preset": ([
                    "2048x1024 (2K VR Production 2:1)",
                    "1536x768 (LoRA Native Draft 2:1)",
                    "1472x736 (Low-VRAM Draft 1.08 MP)",
                    "1568x672 (21:9 Ultrawide 1.05 MP)",
                    "1024x512 (Ultra Fast Preview 2:1)",
                    "Custom 2:1"
                ], {
                    "default": "2048x1024 (2K VR Production 2:1)",
                    "tooltip": "Standard 2:1 equirectangular resolution preset. 2048x1024 recommended for production; 1536x768 or 1024x512 for fast drafts on 8GB VRAM."
                }),
                "custom_width": ("INT", {
                    "default": 2048, "min": 512, "max": 4096, "step": 64,
                    "tooltip": "Custom pixel width when 'Custom 2:1' is selected (auto-snapped to multiple of 64)."
                }),
                "custom_height": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 32,
                    "tooltip": "Custom pixel height when 'Custom 2:1' is selected (auto-snapped to multiple of 32)."
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of latent samples to generate."
                })
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    OUTPUT_TOOLTIPS = (
        "Empty 64-channel latent tensor for Qwen-Image 2.1 DiT and Wan 2.2 VAE.",
        "Calculated image width in pixels.",
        "Calculated image height in pixels."
    )
    FUNCTION = "generate_latent"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Generates empty 64-channel latent tensors for Qwen-Image 2.1 DiT enforcing exact 2:1 equirectangular aspect ratios and multiple-of-64 dimensions."

    def generate_latent(self, resolution_preset, custom_width, custom_height, batch_size):
        if "1472x736" in resolution_preset:
            w, h = 1472, 736
        elif "1568x672" in resolution_preset:
            w, h = 1568, 672
        elif "1536x768" in resolution_preset:
            w, h = 1536, 768
        elif "2048x1024" in resolution_preset:
            w, h = 2048, 1024
        elif "1024x512" in resolution_preset:
            w, h = 1024, 512
        else:
            w = (custom_width // 64) * 64
            h = (custom_height // 32) * 32

        # Native 64-channel latent for Qwen-Image-2.1 DiT and Wan 2.2 VAE (spatial downscale 16)
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
        latent = torch.zeros([batch_size, 64, h // 16, w // 16], device=device, dtype=dtype)
        return ({"samples": latent}, w, h)


class QwenPanorama360Viewer:
    """
    Interactive in-canvas WebGL 360 equirectangular sphere viewer.
    Allows dragging to look in full 360 (yaw/pitch), mouse-wheel FOV zoom,
    auto-rotation, and full-window VR preview directly inside ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Equirectangular 360° panorama image(s) to display in the interactive sphere viewer."}),
            },
            "optional": {
                "default_fov": ("INT", {
                    "default": 80, "min": 40, "max": 120, "step": 5,
                    "tooltip": "Initial camera field-of-view in degrees."
                }),
                "auto_rotate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable smooth rotation when idle (disabled by default; viewer remains static)."
                }),
                "auto_rotate_speed": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Turntable rotation speed."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Pass-through image tensor for chaining to saving or downstream nodes.",)
    FUNCTION = "view_panorama"
    OUTPUT_NODE = True
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Interactive in-canvas 360° VR equirectangular sphere viewer. Supports mouse dragging, wheel FOV zoom, and full-screen VR preview inside ComfyUI."

    def view_panorama(self, images, default_fov=80, auto_rotate=False, auto_rotate_speed=0.5):
        temp_dir = folder_paths.get_temp_directory()
        results = []

        for i, image_tensor in enumerate(images):
            # Tensor to numpy uint8
            img_np = 255.0 * image_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.getchannel("A"))
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            filename = f"qwen_pano_preview_{i}_{os.getpid()}.jpg"
            file_path = os.path.join(temp_dir, filename)
            img.save(file_path, quality=92)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp",
                "width": img.width,
                "height": img.height,
                "aspect_ratio": f"{img.width}:{img.height}",
                "default_fov": default_fov,
                "auto_rotate": auto_rotate,
                "auto_rotate_speed": auto_rotate_speed
            })

        return {"ui": {"images": results}, "result": (images,)}


class QwenPanoramaExport:
    """
    Multi-format export node for 360 panoramas:
    - High-resolution equirectangular PNG/JPG
    - Standard Photo Sphere XMP metadata injection (for auto-360 VR playback on Facebook/Google/VR)
    - Optional 360 rotating horizontal flythrough video (H.264 MP4) via FFmpeg
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Equirectangular 360° panorama image(s) to export."}),
                "filename_prefix": ("STRING", {
                    "default": "Qwen_Panorama_360",
                    "tooltip": "Prefix for exported image and video filenames (saved in ComfyUI/output/Panoramas/)."
                }),
                "format": (["png", "jpg_vr360", "webp"], {
                    "default": "png",
                    "tooltip": "Export file format (PNG or JPEG with Google Photo Sphere GPano 360 metadata)."
                }),
            },
            "optional": {
                "export_rotating_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, renders a smooth 360-degree rotating flythrough MP4 video alongside the image using FFmpeg."
                }),
                "video_duration_seconds": ("INT", {
                    "default": 10, "min": 3, "max": 60, "step": 1,
                    "tooltip": "Duration of the rotating flythrough MP4 video in seconds."
                }),
                "video_fps": ("INT", {
                    "default": 30, "min": 15, "max": 60, "step": 5,
                    "tooltip": "Frame rate (frames per second) of the rotating flythrough MP4 video."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    OUTPUT_TOOLTIPS = ("Comma-separated list of saved image and video file paths in output/Panoramas/.",)
    FUNCTION = "export_panorama"
    OUTPUT_NODE = True
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Exports 360° equirectangular panoramas with embedded Google Photo Sphere (GPano) XMP metadata for VR platforms, with optional rotating MP4 video flythrough."

    def export_panorama(self, images, filename_prefix, format="png", export_rotating_video=False, video_duration_seconds=10, video_fps=30):
        output_dir = folder_paths.get_output_directory()
        pano_dir = os.path.join(output_dir, "Panoramas")
        os.makedirs(pano_dir, exist_ok=True)

        saved_paths = []

        # Find unique index
        base_count = 1
        while any(os.path.exists(os.path.join(pano_dir, f"{filename_prefix}_{base_count:05d}.{ext}")) for ext in ["png", "webp", "jpg"]):
            base_count += 1

        for i, img_tensor in enumerate(images):
            img_np = 255.0 * img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            idx = base_count + i
            ext = "jpg" if format == "jpg_vr360" else format
            out_filename = f"{filename_prefix}_{idx:05d}.{ext}"
            out_path = os.path.join(pano_dir, out_filename)

            if format == "jpg_vr360":
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out_path, format="JPEG", quality=95)
                try:
                    self._inject_vr360_xmp(out_path, img.width, img.height)
                except Exception as e:
                    print(f"[QwenPanoramaExport] XMP injection warning: {e}")
            elif format == "webp":
                img.save(out_path, format="WEBP", quality=92)
            else:
                try:
                    from PIL import PngImagePlugin
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("XML:com.adobe.xmp", self._get_xmp_xml(img.width, img.height))
                    img.save(out_path, format="PNG", pnginfo=pnginfo)
                except Exception:
                    img.save(out_path, format="PNG")

            saved_paths.append(out_path)

            # Optional rotating MP4 flythrough
            if export_rotating_video:
                mp4_filename = f"{filename_prefix}_{idx:05d}_flythrough.mp4"
                mp4_path = os.path.join(pano_dir, mp4_filename)
                self._generate_rotating_mp4(out_path, mp4_path, img.width, img.height, video_duration_seconds, video_fps)
                saved_paths.append(mp4_path)

        return {"ui": {"exported_files": saved_paths}, "result": (", ".join(saved_paths),)}

    def _get_xmp_xml(self, width, height):
        return f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
   <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
   <GPano:CaptureSoftware>ComfyUI-Qwen-Panorama</GPano:CaptureSoftware>
   <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
   <GPano:CroppedAreaImageWidthPixels>{width}</GPano:CroppedAreaImageWidthPixels>
   <GPano:CroppedAreaImageHeightPixels>{height}</GPano:CroppedAreaImageHeightPixels>
   <GPano:FullPanoWidthPixels>{width}</GPano:FullPanoWidthPixels>
   <GPano:FullPanoHeightPixels>{height}</GPano:FullPanoHeightPixels>
   <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
   <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

    def _inject_vr360_xmp(self, jpeg_path, width, height):
        xmp_template = f"http://ns.adobe.com/xap/1.0/\x00{self._get_xmp_xml(width, height)}"
        with open(jpeg_path, "rb") as f:
            data = f.read()

        if data[:2] != b'\xff\xd8':
            return

        xmp_bytes = xmp_template.encode("utf-8")
        app1_len = len(xmp_bytes) + 2
        app1_header = b'\xff\xe1' + app1_len.to_bytes(2, byteorder='big')
        app1_packet = app1_header + xmp_bytes

        # Insert APP1 after SOI
        new_data = data[:2] + app1_packet + data[2:]
        with open(jpeg_path, "wb") as f:
            f.write(new_data)

    def _generate_rotating_mp4(self, img_path, out_mp4_path, width, height, duration, fps):
        """
        Uses FFmpeg v360 filter to render a smooth 360-degree horizontal yaw pan video.
        """
        total_frames = max(1, int(duration * fps))
        h_speed = 1.0 / float(total_frames)
        # Scroll equirectangular horizontally, then project to perspective flat view (1280x720)
        filter_expr = f"scroll=h=-{h_speed:.8f},v360=input=e:output=flat:d_fov=90:w=1280:h=720"

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", img_path,
            "-vf", filter_expr,
            "-t", str(duration),
            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart",
            "-crf", "19",
            out_mp4_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[QwenPanoramaExport] Successfully rendered rotating video: {out_mp4_path}")
        except Exception as e:
            print(f"[QwenPanoramaExport] FFmpeg video generation warning: {e}")


class QwenPanoramaUpscale:
    """
    High-fidelity 360° equirectangular upscaler with circular horizontal seam wrapping.
    Prevents vertical seam line artifacts at the 0°/360° wrap boundary and provides instant,
    zero-latency upscaling from draft resolutions (1024x512 or 1536x768) to 2K (2048x1024) or 4K.
    Includes a direct bypass toggle.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Draft equirectangular panorama image to upscale."}),
                "target_resolution": ([
                    "2048x1024 (2K Production 2:1)",
                    "4096x2048 (4K Ultra VR 2:1)",
                    "Custom Scale",
                    "Bypass / Original (No Upscale)"
                ], {
                    "default": "2048x1024 (2K Production 2:1)",
                    "tooltip": "Target 2:1 spherical resolution preset (2048x1024 Production, 4096x2048 Ultra VR, Custom Scale, or Bypass)."
                }),
                "upscale_method": ([
                    "bicubic",
                    "bilinear",
                    "nearest-exact"
                ], {
                    "default": "bicubic",
                    "tooltip": "Interpolation algorithm ('bicubic' with antialiasing recommended for smooth textures)."
                }),
                "custom_scale": ("FLOAT", {
                    "default": 1.333, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Multiplier factor applied when 'Custom Scale' is selected."
                }),
                "seamless_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enables 5% toroidal boundary wrapping so the left and right edges interpolate together, eliminating vertical seam line artifacts."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    OUTPUT_TOOLTIPS = (
        "Upscaled seamless equirectangular 360° panorama image.",
        "Target width in pixels.",
        "Target height in pixels.",
        "Summary string of the upscale operation."
    )
    FUNCTION = "upscale"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Upscales 360° panoramas to 2K (2048x1024) or 4K (4096x2048) with 5% horizontal toroidal boundary wrapping to eliminate vertical seam artifacts."

    def upscale(self, image, target_resolution, upscale_method, custom_scale, seamless_boundary):
        B, H, W, C = image.shape
        if target_resolution == "Bypass / Original (No Upscale)" or (target_resolution == "Custom Scale" and custom_scale == 1.0):
            return (image, W, H, f"Original: {W}x{H} (Bypassed)")

        if "2048x1024" in target_resolution:
            target_w, target_h = 2048, 1024
        elif "4096x2048" in target_resolution:
            target_w, target_h = 4096, 2048
        else:
            target_w = int(round(W * custom_scale / 64)) * 64
            target_h = int(round(H * custom_scale / 32)) * 32

        if target_w == W and target_h == H:
            return (image, W, H, f"No change: {W}x{H}")

        # Circular horizontal seam wrapping
        if seamless_boundary:
            pad_w = max(16, int(W * 0.05))
            left_pad = image[:, :, -pad_w:, :]
            right_pad = image[:, :, :pad_w, :]
            padded = torch.cat([left_pad, image, right_pad], dim=2)
            pad_target = int(target_w * (pad_w / W))
            padded_target_w = target_w + 2 * pad_target
        else:
            padded = image
            padded_target_w = target_w
            pad_target = 0

        x = padded.permute(0, 3, 1, 2)
        antialias = True if upscale_method in ("bicubic", "bilinear") else False

        upscaled = torch.nn.functional.interpolate(
            x,
            size=(target_h, padded_target_w),
            mode=upscale_method,
            align_corners=False if upscale_method in ("bicubic", "bilinear") else None,
            antialias=antialias
        )

        if pad_target > 0:
            cropped = upscaled[:, :, :, pad_target : pad_target + target_w]
        else:
            cropped = upscaled

        out = cropped.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        info_str = f"Seamless 360 Upscaled: {W}x{H} -> {target_w}x{target_h} ({upscale_method})"
        return (out, target_w, target_h, info_str)


class QwenPanoramaModelUpscale:
    """
    All-in-One 360° Equirectangular Neural Model Upscaler with Seam Elimination.
    Pre-pads 5% toroidal boundary wrapping across left and right edges, executes the loaded
    upscale model (RealESRGAN_x2plus, 4x-UltraSharp, etc.) with automatic full-frame or tiled
    VRAM management, crops out the padding, and outputs a mathematically seamless 360° panorama.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL", {
                    "tooltip": "Neural upscale model loaded via 'Load Upscale Model' (e.g. RealESRGAN_x2plus.pth)."
                }),
                "image": ("IMAGE", {
                    "tooltip": "2:1 equirectangular 360° panorama image to upscale."
                }),
            },
            "optional": {
                "seamless_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable 5% toroidal boundary wrapping so the left and right edges are upscaled with continuous context, eliminating vertical stitch line artifacts."
                }),
                "wrap_percent": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 0.15, "step": 0.01,
                    "tooltip": "Percentage of image width cloned to opposite margins (5% recommended)."
                }),
                "processing_mode": ([
                    "Full Frame (Best Quality)",
                    "Tiled (1024)",
                    "Tiled (Low VRAM 512)"
                ], {
                    "default": "Full Frame (Best Quality)",
                    "tooltip": "Processing mode. Full Frame processes the entire panorama at once without tile boundaries; Tiled saves VRAM on lower-end GPUs."
                }),
                "enable_bilateral_filter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Applies a subtle edge-preserving bilateral filter on the padded upscaled panorama before cropping padding, eliminating dotted noise and speckles without blurring sharp boundaries."
                }),
                "bilateral_d": ("INT", {
                    "default": 5, "min": 1, "max": 15, "step": 1,
                    "tooltip": "Diameter of pixel neighborhood (default: 5). Keep small for subtle cleanup."
                }),
                "bilateral_sigma_color": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Filter sigma in color space (mild recommended: 10 to 20). Higher values smooth color variations."
                }),
                "bilateral_sigma_space": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Filter sigma in coordinate space (mild recommended: 3 to 5)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    OUTPUT_TOOLTIPS = (
        "Upscaled seamless equirectangular 360° panorama image.",
        "Target width in pixels.",
        "Target height in pixels.",
        "Summary information of the upscale process."
    )
    FUNCTION = "upscale_seamless"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Seamless 360° equirectangular model upscaler. Uses toroidal boundary wrapping to eliminate vertical stitch lines when upscaling with learned models (RealESRGAN_x2plus, 4x-UltraSharp)."

    def upscale_seamless(self, upscale_model, image, seamless_boundary=True, wrap_percent=0.05, processing_mode="Full Frame (Best Quality)", enable_bilateral_filter=False, bilateral_d=5, bilateral_sigma_color=15.0, bilateral_sigma_space=4.0):
        device = upscale_model.patcher.load_device
        B, H, W, C = image.shape
        scale = getattr(upscale_model, "scale", 2.0)

        # Neural upscale models (ESRGAN, RealESRGAN, UltraSharp) strictly expect 3-channel RGB.
        # Extract alpha channel if present (RGBA) to prevent channel-mismatch errors in Spandrel.
        has_alpha = C > 3
        is_grayscale = C == 1
        if has_alpha:
            alpha = image[:, :, :, 3:]
            work_image = image[:, :, :, :3]
        elif is_grayscale:
            alpha = None
            work_image = image.repeat(1, 1, 1, 3)
        else:
            alpha = None
            work_image = image

        # Step 1: Pre-pad 5% toroidal boundary if enabled
        if seamless_boundary:
            pad_w = max(16, int(round(W * wrap_percent)))
            # Align padding to 8 pixels for neural models
            pad_w = ((pad_w + 7) // 8) * 8
            left_pad = work_image[:, :, -pad_w:, :]
            right_pad = work_image[:, :, :pad_w, :]
            padded = torch.cat([left_pad, work_image, right_pad], dim=2)
            pad_target = int(round(pad_w * scale))
            target_w = int(round(W * scale))
            target_h = int(round(H * scale))
        else:
            padded = work_image
            pad_w = 0
            pad_target = 0
            target_w = int(round(W * scale))
            target_h = int(round(H * scale))

        # Step 2: Manage VRAM & Load Model
        memory_required = (512 * 512 * 3) * padded.element_size() * max(scale, 1.0) * 384.0
        memory_required += padded.nelement() * padded.element_size()
        comfy.model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)

        in_img = padded.movedim(-1, -3).to(device)

        # Tile size determination
        if processing_mode == "Tiled (Low VRAM 512)":
            tile = 512
        elif processing_mode == "Tiled (1024)":
            tile = 1024
        else:
            # Full Frame (Best Quality)
            tile = max(in_img.shape[3], in_img.shape[2], 2048)

        overlap = 32
        output_device = comfy.model_management.intermediate_device()

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a.float()), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=scale, pbar=pbar, output_device=output_device)
                oom = False
            except Exception as e:
                comfy.model_management.raise_non_oom(e)
                tile //= 2
                if tile < 128:
                    raise e

        # Step 3: Format
        upscaled = torch.clamp(s.movedim(-3, -1), min=0.0, max=1.0).to(comfy.model_management.intermediate_dtype())

        # Step 4: Bilateral Filter on padded image before cropping (if enabled)
        if enable_bilateral_filter and bilateral_d > 0:
            out_list = []
            upscaled_cpu = upscaled.cpu()
            for b in range(upscaled.shape[0]):
                img_np = (upscaled_cpu[b].numpy() * 255.0).clip(0, 255).astype(np.uint8)
                if img_np.shape[-1] == 4:
                    rgb = img_np[:, :, :3]
                    a_ch = img_np[:, :, 3:]
                    filtered_rgb = cv2.bilateralFilter(rgb, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
                    filtered = np.concatenate([filtered_rgb, a_ch], axis=-1)
                elif img_np.shape[-1] == 3:
                    filtered = cv2.bilateralFilter(img_np, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
                else:
                    filtered = cv2.bilateralFilter(img_np[:, :, 0], d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
                    filtered = np.expand_dims(filtered, -1)
                out_list.append(torch.from_numpy(filtered.astype(np.float32) / 255.0))
            upscaled = torch.stack(out_list, dim=0).to(image.device)

        # Step 5: Crop padding
        if pad_target > 0:
            cropped = upscaled[:, :, pad_target : pad_target + target_w, :]
        else:
            cropped = upscaled

        # Re-attach alpha channel if input was RGBA
        if has_alpha:
            alpha_trans = alpha.permute(0, 3, 1, 2)
            alpha_up = torch.nn.functional.interpolate(alpha_trans, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
            alpha_up = alpha_up.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(cropped.device, dtype=cropped.dtype)
            cropped = torch.cat([cropped, alpha_up], dim=-1)
        elif is_grayscale:
            cropped = cropped[:, :, :, :1]

        info_str = f"Seamless 360 {scale}x Upscaled: {W}x{H} -> {target_w}x{target_h} (RealESRGAN / Neural)"
        if enable_bilateral_filter:
            info_str += f" + Bilateral Filter (d={bilateral_d}, sigmaC={bilateral_sigma_color}, sigmaS={bilateral_sigma_space})"
        return (cropped, target_w, target_h, info_str)


class QwenPanoramaDownsample:
    """
    High-Fidelity 360° Equirectangular Downsampler & Supersampling Anti-Aliasing Filter.
    Takes high-resolution upscaled panoramas (e.g. 8K from 4x RealESRGAN) and smoothly
    downsamples to 4K (4096x2048) or 2K (2048x1024) to eliminate high-frequency dotted noise,
    speckle artifacts, and AI upscaler grain while preserving razor-sharp structural edges.
    Features 3 selectable reconstruction filters:
      - 'area': Box averaging filter (optimal for crushing dotted noise and single-pixel artifacts).
      - 'bicubic': Antialiased bicubic filter (smooth, natural transitions).
      - 'lanczos': 3-lobe windowed sinc filter (maximum micro-detail & line sharpness).
    Applies toroidal boundary wrapping to prevent vertical stitch lines across 0°/360°.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "High-resolution 360° panorama image (e.g. 8K from 4x RealESRGAN) to downsample."}),
                "downsample_method": ([
                    "area",
                    "bicubic",
                    "lanczos"
                ], {
                    "default": "area",
                    "tooltip": "Reconstruction algorithm: 'area' (box averaging, best for eliminating dotted/speckle artifacts), 'bicubic' (smooth antialiased bicubic), 'lanczos' (maximum crispness sinc filter)."
                }),
                "target_resolution": ([
                    "4096x2048 (4K Ultra VR 2:1)",
                    "2048x1024 (2K Production 2:1)",
                    "1536x768 (Draft 2:1)",
                    "Custom Scale",
                    "Bypass / Original (No Downsample)"
                ], {
                    "default": "4096x2048 (4K Ultra VR 2:1)",
                    "tooltip": "Target 2:1 equirectangular resolution (4096x2048 recommended for 8K->4K supersampling)."
                }),
            },
            "optional": {
                "custom_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Multiplier factor applied when 'Custom Scale' is selected (e.g. 0.5 downscales 8K to 4K)."
                }),
                "seamless_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Applies 5% toroidal boundary wrapping before filtering so left and right edges filter continuously, eliminating vertical seam artifacts."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    OUTPUT_TOOLTIPS = (
        "Downsampled seamless equirectangular 360° panorama image.",
        "Target width in pixels.",
        "Target height in pixels.",
        "Summary information of the downsampling operation."
    )
    FUNCTION = "downsample"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "360° Equirectangular downsampling node with 3 selectable filters ('area', 'bicubic', 'lanczos') and toroidal boundary wrapping. Ideal for supersampling 8K->4K to eliminate dotted artifacts."

    def downsample(self, image, downsample_method, target_resolution, custom_scale=0.5, seamless_boundary=True):
        B, H, W, C = image.shape
        if target_resolution == "Bypass / Original (No Downsample)" or (target_resolution == "Custom Scale" and custom_scale == 1.0):
            return (image, W, H, f"Original: {W}x{H} (Bypassed)")

        if "4096x2048" in target_resolution:
            target_w, target_h = 4096, 2048
        elif "2048x1024" in target_resolution:
            target_w, target_h = 2048, 1024
        elif "1536x768" in target_resolution:
            target_w, target_h = 1536, 768
        else:
            target_w = max(64, int(round(W * custom_scale / 64)) * 64)
            target_h = max(32, int(round(H * custom_scale / 32)) * 32)

        if target_w == W and target_h == H:
            return (image, W, H, f"No change: {W}x{H}")

        # Toroidal boundary wrapping to prevent vertical seam artifacts
        if seamless_boundary:
            pad_w = max(16, int(round(W * 0.05 / 32)) * 32)
            pad_target = int(round(pad_w * (target_w / W)))
            padded_target_w = target_w + 2 * pad_target
            left_pad = image[:, :, -pad_w:, :]
            right_pad = image[:, :, :pad_w, :]
            padded = torch.cat([left_pad, image, right_pad], dim=2)
        else:
            padded = image
            padded_target_w = target_w
            pad_target = 0

        if downsample_method == "lanczos":
            out_list = []
            padded_cpu = padded.cpu()
            for b in range(padded.shape[0]):
                img_slice = padded_cpu[b].numpy()
                if img_slice.shape[-1] == 4:
                    pil_mode = "RGBA"
                elif img_slice.shape[-1] == 1:
                    pil_mode = "L"
                    img_slice = img_slice[:, :, 0]
                else:
                    pil_mode = "RGB"
                    img_slice = img_slice[:, :, :3]
                img_np = (img_slice * 255.0).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode=pil_mode)
                resized = pil_img.resize((padded_target_w, target_h), Image.Resampling.LANCZOS)
                out_np = np.array(resized, dtype=np.float32) / 255.0
                if len(out_np.shape) == 2:
                    out_np = np.expand_dims(out_np, -1)
                out_list.append(torch.from_numpy(out_np))
            downsampled = torch.stack(out_list, dim=0).to(padded.device)
        elif downsample_method == "bicubic":
            x = padded.permute(0, 3, 1, 2)
            interp = torch.nn.functional.interpolate(x, size=(target_h, padded_target_w), mode="bicubic", align_corners=False, antialias=True)
            downsampled = interp.permute(0, 2, 3, 1)
        elif downsample_method == "area":
            x = padded.permute(0, 3, 1, 2)
            interp = torch.nn.functional.interpolate(x, size=(target_h, padded_target_w), mode="area")
            downsampled = interp.permute(0, 2, 3, 1)
        else:
            x = padded.permute(0, 3, 1, 2)
            interp = torch.nn.functional.interpolate(x, size=(target_h, padded_target_w), mode="bilinear", align_corners=False, antialias=True)
            downsampled = interp.permute(0, 2, 3, 1)

        if pad_target > 0:
            cropped = downsampled[:, :, pad_target : pad_target + target_w, :]
        else:
            cropped = downsampled

        out = cropped.clamp(0.0, 1.0)
        info_str = f"Seamless 360 Downsampled: {W}x{H} -> {target_w}x{target_h} ({downsample_method})"
        return (out, target_w, target_h, info_str)


class QwenPanoramaBilateralFilter:
    """
    Subtle 360° Equirectangular Edge-Preserving Bilateral Cleanup Filter.
    Removes high-frequency speckles, dotted artifacts, and neural upscaler noise while
    strictly preserving crisp structural edges and architectural geometry.
    Uses 5% toroidal boundary wrapping across 0°/360° to guarantee zero stitch lines.

    Tuning Guidelines:
      - Treat as a subtle cleanup, not a heavy denoise pass.
      - Excessive filtering causes plasticky, waxy, or painterly smoothing.
      - Mild baseline: d = 5, sigmaColor = 10 to 20, sigmaSpace = 3 to 5.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular 360° panorama image (e.g. 4K upscaled) to clean up."}),
                "d": ("INT", {
                    "default": 5, "min": 1, "max": 15, "step": 1,
                    "tooltip": "Diameter of each pixel neighborhood (default: 5). Keep small for subtle cleanup."
                }),
                "sigma_color": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Filter sigma in color space (mild recommended: 10 to 20). Controls how colors are mixed."
                }),
                "sigma_space": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Filter sigma in coordinate space (mild recommended: 3 to 5). Controls spatial distance influence."
                }),
            },
            "optional": {
                "blend_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend opacity between filtered and original image (1.0 = 100% filtered, 0.7 = subtle 70% cleanup)."
                }),
                "seamless_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enables 5% toroidal boundary wrapping before filtering so the 0°/360° edge doesn't have kernel boundary seams."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = (
        "Cleaned 360° panorama with dotted noise suppressed and sharp edges intact.",
        "Summary information of the bilateral filter operation."
    )
    FUNCTION = "filter_seamless"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Edge-preserving bilateral cleanup filter for 360° panoramas. Uses toroidal boundary wrapping to prevent vertical seam artifacts while gently eliminating ESRGAN dotted noise."

    def filter_seamless(self, image, d, sigma_color, sigma_space, blend_opacity=1.0, seamless_boundary=True):
        B, H, W, C = image.shape

        if blend_opacity <= 0.0 or d <= 0:
            return (image, f"Bypass: {W}x{H} (Opacity 0.0)")

        # Toroidal boundary wrapping to prevent seam artifacts
        if seamless_boundary:
            pad_w = max(16, int(round(W * 0.05 / 32)) * 32)
            left_pad = image[:, :, -pad_w:, :]
            right_pad = image[:, :, :pad_w, :]
            padded = torch.cat([left_pad, image, right_pad], dim=2)
        else:
            padded = image
            pad_w = 0

        out_list = []
        padded_cpu = padded.cpu()
        for b in range(padded.shape[0]):
            img_np = (padded_cpu[b].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if img_np.shape[-1] == 4:
                rgb = img_np[:, :, :3]
                alpha = img_np[:, :, 3:]
                filtered_rgb = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                filtered = np.concatenate([filtered_rgb, alpha], axis=-1)
            elif img_np.shape[-1] == 3:
                filtered = cv2.bilateralFilter(img_np, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            else:
                filtered = cv2.bilateralFilter(img_np[:, :, 0], d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                filtered = np.expand_dims(filtered, -1)

            out_list.append(torch.from_numpy(filtered.astype(np.float32) / 255.0))

        result_padded = torch.stack(out_list, dim=0).to(image.device)

        if pad_w > 0:
            result = result_padded[:, :, pad_w : pad_w + W, :]
        else:
            result = result_padded

        if blend_opacity < 1.0:
            result = blend_opacity * result + (1.0 - blend_opacity) * image

        out = result.clamp(0.0, 1.0)
        info = f"Bilateral Filter (d={d}, sigmaColor={sigma_color}, sigmaSpace={sigma_space}, opacity={blend_opacity*100:.0f}%)"
        return (out, info)




class QwenPanoramaSeamExtract:
    """
    Step 1 of Surgical Seam Healing:
    Rolls the 360 panorama horizontally by 50% so the seam is placed directly at the center,
    then crops a vertical strip (e.g. 512x1024) centered at the seam.
    The small crop can be inpainted/denoised in ~3-4 seconds with zero VRAM offloading.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Full 360° equirectangular panorama image to inspect/repair."}),
                "crop_width": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Width in pixels of the center vertical strip to crop (default: 512px). Keeps sampling fast with zero memory swapping."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "SEAM_CONTEXT")
    RETURN_NAMES = ("crop_image", "seam_context")
    OUTPUT_TOOLTIPS = (
        "Cropped vertical image centered exactly on the boundary seam for targeted inpainting.",
        "Context dictionary storing coordinates and rolled background required by Step 2 (Seam Restore)."
    )
    FUNCTION = "extract_seam"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Step 1 of Surgical Seam Healing: Rolls the panorama horizontally by 50% to place the seam at the center, then crops a narrow vertical strip for fast inpainting."

    def extract_seam(self, image, crop_width):
        B, H, W, C = image.shape
        rolled = torch.roll(image, shifts=W // 2, dims=2)
        cx = W // 2
        x1 = max(0, cx - crop_width // 2)
        x2 = min(W, cx + crop_width // 2)
        crop = rolled[:, :, x1:x2, :].clone()
        seam_context = {
            "rolled": rolled,
            "x1": x1,
            "x2": x2,
            "W": W,
            "H": H
        }
        return (crop, seam_context)


class QwenPanoramaSeamPaste:
    """
    Step 2 of Surgical Seam Healing:
    Takes the healed crop, blends it back into the rolled panorama with a smooth
    feathered alpha transition, and rolls back by -50% to original orientation.
    Outputs the final seamless 360 panorama!
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "healed_crop": ("IMAGE", {"tooltip": "The inpainted/denoised vertical crop returned from your inpainting sampler."}),
                "seam_context": ("SEAM_CONTEXT", {"tooltip": "Context dictionary from Qwen 360 Panorama Seam Extract (Step 1)."}),
                "feather_width": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Pixel width of the cosine alpha edge blend (prevents sharp boundaries between the inpainted patch and original panorama)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("healed_panorama",)
    OUTPUT_TOOLTIPS = ("Final seamless 360° equirectangular panorama with the seam fully healed and restored to original orientation.",)
    FUNCTION = "paste_seam"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Step 2 of Surgical Seam Healing: Blends the inpainted crop back with cosine feathered alpha masking and unrolls by -50% to restore original orientation."

    def paste_seam(self, healed_crop, seam_context, feather_width):
        rolled = seam_context["rolled"].clone()
        x1 = seam_context["x1"]
        x2 = seam_context["x2"]
        W = seam_context["W"]
        crop_w = x2 - x1

        # Match dimensions if needed
        if healed_crop.shape[2] != crop_w or healed_crop.shape[1] != rolled.shape[1]:
            healed_crop = torch.nn.functional.interpolate(
                healed_crop.permute(0, 3, 1, 2),
                size=(rolled.shape[1], crop_w),
                mode="bicubic",
                align_corners=False
            ).permute(0, 2, 3, 1)

        # Match color channels
        if healed_crop.shape[-1] > rolled.shape[-1]:
            healed_crop = healed_crop[:, :, :, :rolled.shape[-1]]

        feather = min(feather_width, crop_w // 4)
        device = healed_crop.device
        dtype = healed_crop.dtype

        mask_1d = torch.ones(crop_w, device=device, dtype=dtype)
        ramp_in = 0.5 * (1.0 - torch.cos(torch.linspace(0, torch.pi, feather, device=device, dtype=dtype)))
        ramp_out = 0.5 * (1.0 + torch.cos(torch.linspace(0, torch.pi, feather, device=device, dtype=dtype)))
        mask_1d[:feather] = ramp_in
        mask_1d[-feather:] = ramp_out
        alpha_mask = mask_1d.view(1, 1, crop_w, 1)

        existing = rolled[:, :, x1:x2, :]
        blended = existing * (1.0 - alpha_mask) + healed_crop * alpha_mask
        rolled[:, :, x1:x2, :] = blended

        final_pano = torch.roll(rolled, shifts=-(W // 2), dims=2)
        return (final_pano.clamp(0.0, 1.0),)


class QwenPanoramaRoll:
    """
    Horizontally rolls a 360 equirectangular panorama and optional mask:
    - 'Roll 50% (Center Seam)': shifts by +50% of image width so the boundary seam is placed directly at the center for inspection or interactive mask painting.
    - 'Unroll -50% (Restore Horizon)': shifts by -50% to restore the original 360 sphere orientation.
    - 'Custom Shift': shifts by an arbitrary percentage (-100% to +100%).
    If a mask is connected, it is rolled in exact sync with the image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "360° equirectangular panorama image to roll."}),
                "mode": ([
                    "Roll 50% (Center Seam)",
                    "Unroll -50% (Restore Horizon)",
                    "Custom Shift"
                ], {
                    "default": "Roll 50% (Center Seam)",
                    "tooltip": "Roll mode: 'Roll 50%' places the boundary seam at the center; 'Unroll -50%' restores original orientation; 'Custom Shift' allows arbitrary horizontal scrolling."
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask tensor to roll in exact lockstep with the image."}),
                "custom_shift_percent": ("FLOAT", {
                    "default": 50.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Percentage of image width to shift when 'Custom Shift' is selected (-100% to +100%)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Horizontally shifted equirectangular image.",
        "Horizontally shifted mask matching the image shift."
    )
    FUNCTION = "roll"
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Horizontally rolls a 360° equirectangular image and optional mask along the cylindrical axis to inspect or edit the boundary seam."

    def roll(self, image, mode, mask=None, custom_shift_percent=50.0):
        if image.shape[-1] > 3:
            image = image[:, :, :, :3]
        B, H, W, C = image.shape
        if mode == "Roll 50% (Center Seam)":
            shift = W // 2
        elif mode == "Unroll -50% (Restore Horizon)":
            shift = -(W // 2)
        else:
            shift = int(W * (custom_shift_percent / 100.0))

        rolled_image = torch.roll(image, shifts=shift, dims=2)

        if mask is not None:
            rolled_mask = torch.roll(mask, shifts=shift, dims=-1)
        else:
            rolled_mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

        return (rolled_image, rolled_mask)


class QwenPanoramaSeamRouter:
    """
    Intelligent 2-Phase Seam Healing Router:
    Phase 1 (No mask detected):
      - Rolls the 360 panorama by +50% so the seam is placed right down the middle.
      - Saves the centered panorama to ComfyUI's input/ directory (seam_centered_panorama.png).
      - Silently blocks downstream inpainting so no VRAM/sampler time is wasted.
      - The user then simply selects seam_centered_panorama.png in LoadImage and paints the mask!
    Phase 2 (Mask detected):
      - Passes the centered image and user's painted mask to InpaintCropImproved for ~3s Viggle Turbo healing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular panorama image to route."}),
                "mask": ("MASK", {"tooltip": "Inpainting mask tensor from MaskEditor or LoadImage."}),
                "mode": ([
                    "Auto (Center & Save if No Mask, Heal if Masked)",
                    "Force Center & Save to input/",
                    "Force Inpaint Heal"
                ], {
                    "default": "Auto (Center & Save if No Mask, Heal if Masked)",
                    "tooltip": "Routing behavior: 'Auto' saves to input/ if mask is empty and executes healing once mask is drawn; 'Force Center' always saves to input/."
                }),
                "centered_filename": ("STRING", {
                    "default": "seam_centered_panorama.png",
                    "tooltip": "Filename used when saving the centered panorama into ComfyUI's input/ directory."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Routed panorama image for downstream inpainting nodes.",
        "Routed inpainting mask tensor."
    )
    FUNCTION = "route"
    OUTPUT_NODE = True
    CATEGORY = "QwenPanorama"
    DESCRIPTION = "Two-phase workflow router: Centers the seam and saves to input/ when no mask is present; proceeds to inpainting when a mask is detected."

    def route(self, image, mask, mode, centered_filename="seam_centered_panorama.png"):
        B, H, W, C = image.shape
        is_empty_mask = (mask is None) or (torch.all(mask <= 0.001).item())

        should_center = (mode == "Force Center & Save to input/") or \
                        (mode == "Auto (Center & Save if No Mask, Heal if Masked)" and is_empty_mask)

        if should_center:
            # Roll +50% to center the boundary seam
            rolled = torch.roll(image, shifts=W // 2, dims=2)
            if rolled.shape[-1] > 3:
                rolled = rolled[:, :, :, :3]

            # Save directly to input/ folder
            input_dir = folder_paths.get_input_directory()
            save_path = os.path.join(input_dir, centered_filename)

            img_np = (rolled[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(save_path)
            print(f"[QwenPanorama] Centered seam saved to input directory: {save_path}")

            ui_item = [{"filename": centered_filename, "subfolder": "", "type": "input"}]
            try:
                from comfy_execution.graph_utils import ExecutionBlocker
                return {"ui": {"images": ui_item}, "result": (ExecutionBlocker("Seam centered! Select 'seam_centered_panorama.png' in LoadImage, open MaskEditor, paint defect, and queue prompt."), ExecutionBlocker(None))}
            except Exception:
                return {"ui": {"images": ui_item}, "result": (rolled, mask)}

        # Phase 2: Mask is present!
        return (image, mask)




