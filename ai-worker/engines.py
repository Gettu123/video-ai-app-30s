"""Motores de video.

Wan2.1 (Alibaba, Apache-2.0): https://github.com/Wan-Video/Wan2.1
HunyuanVideo / HunyuanVideo-I2V (Tencent): https://github.com/Tencent-Hunyuan/HunyuanVideo
  e https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
CogVideoX / CogVideoX1.5 (THUDM / Zhipu, zai-org): https://github.com/zai-org/CogVideo
VideoX-Fun (aigc-apps): https://github.com/aigc-apps/VideoX-Fun
  Nao embute outro modelo. Encadeia o ultimo quadro do clipe anterior
  (o mesmo esquema start-image dos predict_i2v do VideoX-Fun) ate a duracao pedida.
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

NEGATIVE = (
    "bright colors, overexposed, static, blurred details, subtitles, "
    "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, "
    "poorly drawn hands, poorly drawn face, mutation, disfigured, still frame"
)

# num_frames / fps = duracao nativa de um clipe. 30s sai da cadeia, nao de um forward so.
CATALOG: dict[str, dict] = {
    "ffmpeg": {
        "label": "Ken Burns (FFmpeg)",
        "vendor": "local",
        "repo": "",
        "model_id": "",
        "env_model": "",
        "family": "ffmpeg",
        "task": "i2v",
        "num_frames": 0,
        "fps": 12,
        "guidance": 0,
        "steps": 0,
        "vram_gb": 0,
        "license": "local",
    },
    "wan2.1-i2v": {
        "label": "Wan2.1 I2V 14B",
        "vendor": "Alibaba Cloud / Wan-Video",
        "repo": "https://github.com/Wan-Video/Wan2.1",
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "env_model": "WAN_I2V_MODEL",
        "family": "diffusion",
        "task": "i2v",
        "num_frames": 81,
        "fps": 16,
        "guidance": 5.0,
        "steps": 30,
        "vram_gb": 24,
        "license": "Apache-2.0",
    },
    "hunyuan-i2v": {
        "label": "HunyuanVideo-I2V",
        "vendor": "Tencent",
        "repo": "https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V",
        "model_id": "hunyuanvideo-community/HunyuanVideo-I2V",
        "env_model": "HUNYUAN_I2V_MODEL",
        "family": "diffusion",
        "task": "i2v",
        "num_frames": 129,
        "fps": 24,
        "guidance": 6.0,
        "steps": 30,
        "vram_gb": 24,
        "license": "Tencent Hunyuan (aceite no Hugging Face)",
    },
    "hunyuan-t2v": {
        "label": "HunyuanVideo T2V",
        "vendor": "Tencent",
        "repo": "https://github.com/Tencent-Hunyuan/HunyuanVideo",
        "model_id": "hunyuanvideo-community/HunyuanVideo",
        "env_model": "HUNYUAN_T2V_MODEL",
        "family": "diffusion",
        "task": "t2v",
        "extend_with": "hunyuan-i2v",
        "num_frames": 129,
        "fps": 24,
        "guidance": 6.0,
        "steps": 30,
        "vram_gb": 24,
        "license": "Tencent Hunyuan (aceite no Hugging Face)",
    },
    "cogvideox-5b-i2v": {
        "label": "CogVideoX-5B I2V",
        "vendor": "THUDM / Zhipu AI",
        "repo": "https://github.com/zai-org/CogVideo",
        "model_id": "THUDM/CogVideoX-5b-I2V",
        "env_model": "COGVIDEOX_MODEL",
        "family": "diffusion",
        "task": "i2v",
        "num_frames": 49,
        "fps": 8,
        "guidance": 6.0,
        "steps": 50,
        "vram_gb": 18,
        "license": "CogVideoX License",
    },
    "cogvideox1.5-i2v": {
        "label": "CogVideoX1.5-5B I2V",
        "vendor": "THUDM / Zhipu AI (zai-org)",
        "repo": "https://github.com/zai-org/CogVideo",
        "model_id": "THUDM/CogVideoX1.5-5B-I2V",
        "env_model": "COGVIDEOX15_MODEL",
        "family": "diffusion",
        "task": "i2v",
        "num_frames": 81,
        "fps": 8,
        "guidance": 6.0,
        "steps": 50,
        "vram_gb": 18,
        "license": "CogVideoX License",
    },
}

_LOCK = threading.Lock()
_LOADED: dict = {"key": None, "pipe": None}


@dataclass
class GenerationResult:
    path: Path
    engine: str
    orchestrator: str
    segments: int
    message: str
    kind: str


def native_seconds(spec: dict) -> float:
    if not spec["num_frames"] or not spec["fps"]:
        return 30.0
    return spec["num_frames"] / spec["fps"]


def runtime_probe() -> dict:
    info = {"torch": False, "cuda": False, "diffusers": False, "device": None}
    try:
        import torch

        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import diffusers  # noqa: F401

        info["diffusers"] = True
    except Exception:
        pass
    return info


def list_engines() -> list[dict]:
    probe = runtime_probe()
    gpu_ready = probe["torch"] and probe["diffusers"] and (probe["cuda"] or os.getenv("ALLOW_CPU") == "1")
    rows = []
    for engine_id, spec in CATALOG.items():
        ready = True if spec["family"] == "ffmpeg" else gpu_ready
        reason = ""
        if not ready:
            if not probe["torch"] or not probe["diffusers"]:
                reason = "Perfil GPU nao instalado (docker-compose.gpu.yml)."
            elif not probe["cuda"]:
                reason = "CUDA nao encontrada."
        rows.append({
            "id": engine_id,
            "label": spec["label"],
            "vendor": spec["vendor"],
            "repo": spec["repo"],
            "model_id": os.getenv(spec["env_model"], spec["model_id"]) if spec["env_model"] else "",
            "native_seconds": round(native_seconds(spec), 2) if spec["family"] != "ffmpeg" else 30,
            "fps": spec["fps"],
            "num_frames": spec["num_frames"],
            "vram_gb": spec["vram_gb"],
            "license": spec["license"],
            "ready": ready,
            "reason": reason,
        })
    rows.append({
        "id": "videox-fun",
        "label": "VideoX-Fun (encadeamento)",
        "vendor": "aigc-apps",
        "repo": "https://github.com/aigc-apps/VideoX-Fun",
        "model_id": "",
        "native_seconds": None,
        "fps": None,
        "num_frames": None,
        "vram_gb": None,
        "license": "Apache-2.0",
        "ready": gpu_ready,
        "reason": "" if gpu_ready else "Encadeia Wan, Hunyuan ou CogVideoX; precisa do perfil GPU.",
        "chains": [k for k, v in CATALOG.items() if v["family"] == "diffusion"],
    })
    return rows


def _run_ffmpeg(cmd: list[str], timeout: int = 300) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="FFmpeg excedio el tiempo limite") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "ffmpeg fallo")[-800:]
        raise HTTPException(status_code=500, detail=tail)


def _require_gpu() -> None:
    probe = runtime_probe()
    if not probe["torch"] or not probe["diffusers"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Motor de difusion no instalado en esta imagen. "
                "Sube el worker con docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build. "
                "Ken Burns (engine=ffmpeg) sigue disponible sin GPU."
            ),
        )
    if not probe["cuda"] and os.getenv("ALLOW_CPU") != "1":
        raise HTTPException(
            status_code=503,
            detail="CUDA no encontrada. Wan2.1, HunyuanVideo y CogVideoX necesitan GPU NVIDIA.",
        )


def _place(pipe):
    mode = os.getenv("VRAM_MODE", "offload")
    if mode == "cuda":
        pipe.to("cuda")
    elif hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
    return pipe


def _build(engine_id: str):
    import torch

    spec = CATALOG[engine_id]
    model_id = os.getenv(spec["env_model"], spec["model_id"])
    dtype = torch.bfloat16
    if engine_id == "wan2.1-i2v":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel

        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, image_encoder=image_encoder, torch_dtype=dtype
        )
    elif engine_id == "hunyuan-i2v":
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            model_id, transformer=transformer, torch_dtype=torch.float16
        )
    elif engine_id == "hunyuan-t2v":
        from diffusers import HunyuanVideoPipeline

        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    elif engine_id in ("cogvideox-5b-i2v", "cogvideox1.5-i2v"):
        from diffusers import CogVideoXImageToVideoPipeline

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        raise HTTPException(status_code=400, detail=f"Sin pipeline para {engine_id}")
    return _place(pipe)


def _pipe_for(engine_id: str):
    spec = CATALOG[engine_id]
    model_id = os.getenv(spec["env_model"], spec["model_id"]) if spec["env_model"] else engine_id
    key = f"{engine_id}|{model_id}"
    if _LOADED["key"] != key:
        _LOADED["pipe"] = None
        _LOADED["key"] = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _LOADED["pipe"] = _build(engine_id)
        _LOADED["key"] = key
    return _LOADED["pipe"]


def _generator(seed: int | None):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    value = seed if seed is not None else int(time.time()) % (2**31 - 1)
    try:
        return torch.Generator(device=device).manual_seed(value)
    except Exception:
        return torch.Generator().manual_seed(value)


def _fit_wan(pipe, image: Image.Image) -> tuple[Image.Image, int, int]:
    mod = 16
    try:
        mod = int(pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1])
    except Exception:
        mod = 16
    model_id = os.getenv("WAN_I2V_MODEL", CATALOG["wan2.1-i2v"]["model_id"])
    max_area = 720 * 1280 if "720" in model_id else 480 * 832
    aspect = image.height / max(image.width, 1)
    height = max(mod, int(round((max_area * aspect) ** 0.5)) // mod * mod)
    width = max(mod, int(round((max_area / aspect) ** 0.5)) // mod * mod)
    return image.resize((width, height)), height, width


def _export(frames, path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    export_to_video(frames, str(path), fps=fps)


def _invoke(pipe, kwargs: dict):
    local = dict(kwargs)
    while True:
        try:
            return pipe(**local).frames[0]
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise HTTPException(status_code=500, detail=message[:800]) from exc
            name = message.rsplit(" ", 1)[-1].strip("'\"")
            if name not in local:
                raise HTTPException(status_code=500, detail=message[:800]) from exc
            local.pop(name)


def _sample(engine_id: str, image: Image.Image | None, prompt: str, steps: int, seed: int | None):
    spec = CATALOG[engine_id]
    pipe = _pipe_for(engine_id)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "num_frames": spec["num_frames"],
        "guidance_scale": spec["guidance"],
        "num_inference_steps": steps,
        "generator": _generator(seed),
    }
    if spec["task"] == "t2v" and image is None:
        return _invoke(pipe, kwargs)
    if engine_id == "wan2.1-i2v":
        fitted, height, width = _fit_wan(pipe, image)
        kwargs.update({"image": fitted, "height": height, "width": width})
        return _invoke(pipe, kwargs)
    kwargs["image"] = image
    return _invoke(pipe, kwargs)


def _clip(engine_id: str, image: Image.Image | None, prompt: str, steps: int, seed: int | None, path: Path) -> None:
    spec = CATALOG[engine_id]
    frames = _sample(engine_id, image, prompt, steps, seed)
    _export(frames, path, spec["fps"])
    if not path.exists() or path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail=f"{spec['label']} no escribio el MP4")


def _last_frame(video: Path, dest: Path) -> Path:
    _run_ffmpeg(["ffmpeg", "-y", "-sseof", "-0.08", "-i", str(video), "-frames:v", "1", str(dest)])
    if not dest.exists():
        raise HTTPException(status_code=500, detail="No se pudo extraer el ultimo cuadro para encadenar")
    return dest


def _concat(clips: list[Path], dest: Path, seconds: float) -> None:
    listing = dest.with_suffix(".txt")
    listing.write_text("".join(f"file '{p}'\n" for p in clips))
    _run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dest),
    ])


def ken_burns(image_path: Path, dest: Path, duration: int, fps: int) -> None:
    render_fps = min(max(fps, 8), 12)
    frames = max(1, duration * render_fps)
    vf = (
        "scale=960:540:force_original_aspect_ratio=increase,crop=960:540,"
        f"zoompan=z='min(1.0+0.0012*on,1.35)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=960x540:fps={render_fps}"
    )
    _run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-i", str(image_path),
        "-vf", vf, "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ])


def trim_video(src: Path, dest: Path, duration: int) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src), "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", str(dest),
    ])


def first_frame(src: Path, dest: Path) -> Path:
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(src), "-frames:v", "1", str(dest)])
    return dest


def _resolve(engine: str, base_engine: str, chain: bool) -> tuple[str, str, bool]:
    engine = (engine or "ffmpeg").strip().lower()
    base_engine = (base_engine or "cogvideox1.5-i2v").strip().lower()
    if engine == "videox-fun":
        if base_engine not in CATALOG or CATALOG[base_engine]["family"] != "diffusion":
            raise HTTPException(status_code=400, detail="VideoX-Fun necesita un base_engine de difusion")
        return base_engine, "videox-fun", True
    if engine not in CATALOG:
        raise HTTPException(status_code=400, detail=f"Motor desconocido: {engine}")
    return engine, engine, chain


def run_generation(
    *,
    image_path: Path | None,
    prompt: str,
    duration: int,
    engine: str,
    base_engine: str,
    chain: bool,
    seed: int | None,
    steps: int | None,
    fps: int,
    work_dir: Path,
    dest: Path,
) -> GenerationResult:
    with _LOCK:
        return _run_locked(
            image_path=image_path,
            prompt=prompt.strip() or "Cinematic natural motion, subtle camera move, high detail",
            duration=duration,
            engine=engine,
            base_engine=base_engine,
            chain=chain,
            seed=seed,
            steps=steps,
            fps=fps,
            work_dir=work_dir,
            dest=dest,
        )


def _run_locked(
    *,
    image_path: Path | None,
    prompt: str,
    duration: int,
    engine: str,
    base_engine: str,
    chain: bool,
    seed: int | None,
    steps: int | None,
    fps: int,
    work_dir: Path,
    dest: Path,
) -> GenerationResult:
    engine_id, orchestrator, chain = _resolve(engine, base_engine, chain)
    spec = CATALOG[engine_id]

    if spec["family"] == "ffmpeg":
        if image_path is None:
            raise HTTPException(status_code=400, detail="Ken Burns necesita una imagen")
        ken_burns(image_path, dest, duration, fps)
        return GenerationResult(
            path=dest,
            engine="ffmpeg",
            orchestrator="ffmpeg",
            segments=1,
            kind="image-kenburns",
            message="Video generado con movimiento Ken Burns (FFmpeg), sin modelo de difusion.",
        )

    _require_gpu()
    native = native_seconds(spec)
    cap = max(1, int(os.getenv("MAX_CHAIN_SEGMENTS", "6")))
    segments = 1
    if chain and duration > native + 0.05:
        segments = min(cap, max(1, math.ceil(duration / native)))
    used_steps = steps or spec["steps"]
    used_steps = max(10, min(50, int(used_steps)))

    if spec["task"] == "i2v" and image_path is None:
        raise HTTPException(status_code=400, detail=f"{spec['label']} necesita una imagen")

    clips: list[Path] = []
    current = Image.open(image_path).convert("RGB") if image_path else None
    for index in range(segments):
        step_engine = engine_id
        step_image = current
        if spec["task"] == "t2v" and index == 0:
            step_image = None
        elif spec["task"] == "t2v":
            step_engine = spec["extend_with"]
        clip_path = work_dir / f"seg_{index:02d}.mp4"
        clip_seed = None if seed is None else seed + index
        _clip(step_engine, step_image, prompt, used_steps, clip_seed, clip_path)
        clips.append(clip_path)
        if index < segments - 1:
            frame_path = work_dir / f"seg_{index:02d}_last.png"
            _last_frame(clip_path, frame_path)
            current = Image.open(frame_path).convert("RGB")

    produced = min(float(duration), segments * native)
    if len(clips) == 1 and produced >= duration - 0.2:
        clips[0].replace(dest)
    else:
        _concat(clips, dest, produced)

    label = spec["label"]
    via = "VideoX-Fun" if orchestrator == "videox-fun" else label
    message = (
        f"{via}: {segments} clipe(s) de ~{native:.1f}s "
        f"({label}, {spec['num_frames']} frames @ {spec['fps']}fps) "
        f"concatenados hasta {produced:.1f}s."
    )
    return GenerationResult(
        path=dest,
        engine=engine_id,
        orchestrator=orchestrator,
        segments=segments,
        kind=f"{orchestrator}:{engine_id}",
        message=message,
    )
