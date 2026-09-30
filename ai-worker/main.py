import base64
import io
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_DURATION = int(os.environ.get("MAX_DURATION_SECONDS", "30"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
MAX_BYTES = 40 * 1024 * 1024

app = FastAPI(title="Image-to-Video AI Worker", version="1.0.0")


class VideoGenerationRequest(BaseModel):
    image_base64: str = Field(default="", description="Imagen fuente codificada en Base64")
    video_base64: str = Field(default="", description="Video de camara (webm/mp4) en Base64")
    prompt: str = Field(default="", description="Prompt o indicaciones de movimiento")
    duration_seconds: int = Field(default=30, ge=1, le=30)
    fps: int = Field(default=12, ge=8, le=30)


class VideoGenerationResponse(BaseModel):
    success: bool
    job_id: str
    duration_seconds: int
    video_url: str
    message: str


def _decode(data: str) -> bytes:
    raw = data.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Base64 invalido") from exc
    if not blob:
        raise HTTPException(status_code=400, detail="Contenido vacio")
    if len(blob) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Archivo por encima de 40MB")
    return blob


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="FFmpeg excedio el tiempo limite") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "ffmpeg fallo")[-800:]
        raise HTTPException(status_code=500, detail=tail)


def _public_url(filename: str) -> str:
    rel = f"/outputs/{filename}"
    return f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel


def _write_meta(job_id: str, payload: VideoGenerationRequest, kind: str) -> None:
    meta = {
        "job_id": job_id,
        "kind": kind,
        "prompt": payload.prompt,
        "duration_seconds": payload.duration_seconds,
        "engine": "ffmpeg",
    }
    (OUTPUT_DIR / f"{job_id}.json").write_text(json.dumps(meta, ensure_ascii=False))


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Image-to-Video Engine", "engine": "ffmpeg", "max_duration_seconds": MAX_DURATION}


@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(payload: VideoGenerationRequest):
    """
    Genera un MP4 de hasta 30s.
    Imagen -> movimiento Ken Burns (FFmpeg). El prompt se guarda para enchufar
    despues Deforum / SVD / LivePortrait / CogVideoX / ComfyUI.
    Video de camara -> recorte a la duracion pedida y reencode a MP4.
    """
    if payload.duration_seconds > MAX_DURATION:
        raise HTTPException(status_code=400, detail="La duracion maxima permitida es de 30 segundos.")

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    mp4_path = OUTPUT_DIR / f"{job_id}.mp4"
    duration = payload.duration_seconds

    if payload.image_base64:
        image_bytes = _decode(payload.image_base64)
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen") from exc
        image.thumbnail((1280, 1280))
        img_path = OUTPUT_DIR / f"{job_id}.png"
        image.save(img_path, "PNG")
        render_fps = min(payload.fps, 12)
        frames = max(1, duration * render_fps)
        vf = (
            "scale=960:540:force_original_aspect_ratio=increase,crop=960:540,"
            f"zoompan=z='min(1.0+0.0012*on,1.35)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s=960x540:fps={render_fps}"
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4_path),
        ])
        kind = "image-kenburns"
        message = "Video generado a partir de la imagen (movimiento Ken Burns, FFmpeg). El prompt quedo registrado para el modelo de IA."
    elif payload.video_base64:
        video_bytes = _decode(payload.video_base64)
        src_path = OUTPUT_DIR / f"{job_id}.src"
        src_path.write_bytes(video_bytes)
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(src_path),
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(mp4_path),
        ])
        kind = "camera-trim"
        message = "Video de camara recortado a la duracion solicitada (maximo 30s)."
    else:
        raise HTTPException(status_code=400, detail="Envia image_base64 o video_base64")

    if not mp4_path.exists() or mp4_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="El MP4 no fue escrito")

    _write_meta(job_id, payload, kind)
    return VideoGenerationResponse(
        success=True,
        job_id=job_id,
        duration_seconds=duration,
        video_url=_public_url(mp4_path.name),
        message=message,
    )


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
