import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import engines

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_DURATION = int(os.environ.get("MAX_DURATION_SECONDS", "30"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
MAX_BYTES = 40 * 1024 * 1024

app = FastAPI(title="Image-to-Video AI Worker", version="1.1.0")


class VideoGenerationRequest(BaseModel):
    image_base64: str = Field(default="", description="Imagen fuente en Base64")
    video_base64: str = Field(default="", description="Video de camara en Base64")
    prompt: str = Field(default="")
    duration_seconds: int = Field(default=30, ge=1, le=30)
    fps: int = Field(default=12, ge=8, le=30)
    engine: str = Field(default="ffmpeg", description="ffmpeg | wan2.1-i2v | hunyuan-i2v | hunyuan-t2v | cogvideox-5b-i2v | cogvideox1.5-i2v | videox-fun")
    base_engine: str = Field(default="cogvideox1.5-i2v", description="Modelo que VideoX-Fun encadena")
    chain: bool = Field(default=True, description="Encadenar clipes hasta la duracion pedida")
    seed: int | None = None
    steps: int | None = Field(default=None, ge=10, le=50)


class VideoGenerationResponse(BaseModel):
    success: bool
    job_id: str
    duration_seconds: int
    video_url: str
    message: str
    engine: str = "ffmpeg"
    segments: int = 1


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


def _public_url(filename: str) -> str:
    rel = f"/outputs/{filename}"
    return f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel


def _save_image(blob: bytes, path: Path) -> Path:
    try:
        image = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen") from exc
    image.thumbnail((1280, 1280))
    image.save(path, "PNG")
    return path


@app.get("/health")
def health_check():
    probe = engines.runtime_probe()
    return {
        "status": "ok",
        "service": "Image-to-Video Engine",
        "max_duration_seconds": MAX_DURATION,
        "cuda": probe["cuda"],
        "device": probe["device"],
    }


@app.get("/engines")
def get_engines():
    return {"max_duration_seconds": MAX_DURATION, "engines": engines.list_engines()}


@app.post("/generate-video", response_model=VideoGenerationResponse)
def generate_video(payload: VideoGenerationRequest):
    if payload.duration_seconds > MAX_DURATION:
        raise HTTPException(status_code=400, detail="La duracion maxima permitida es de 30 segundos.")

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / f"{job_id}.mp4"
    image_path = None

    diffusion = payload.engine not in ("", "ffmpeg")
    if payload.image_base64:
        image_path = _save_image(_decode(payload.image_base64), work_dir / "input.png")
    elif payload.video_base64:
        src = work_dir / "camera.src"
        src.write_bytes(_decode(payload.video_base64))
        if not diffusion:
            engines.trim_video(src, dest, payload.duration_seconds)
            _write_meta(job_id, payload, "camera-trim", "ffmpeg", 1)
            return VideoGenerationResponse(
                success=True,
                job_id=job_id,
                duration_seconds=payload.duration_seconds,
                video_url=_public_url(dest.name),
                message="Video de camara recortado a la duracion solicitada (maximo 30s).",
                engine="ffmpeg",
                segments=1,
            )
        image_path = engines.first_frame(src, work_dir / "input.png")
    elif payload.engine not in ("hunyuan-t2v",) and not (
        payload.engine == "videox-fun" and payload.base_engine == "hunyuan-t2v"
    ):
        raise HTTPException(status_code=400, detail="Envia image_base64 o video_base64")

    result = engines.run_generation(
        image_path=image_path,
        prompt=payload.prompt,
        duration=payload.duration_seconds,
        engine=payload.engine,
        base_engine=payload.base_engine,
        chain=payload.chain,
        seed=payload.seed,
        steps=payload.steps,
        fps=payload.fps,
        work_dir=work_dir,
        dest=dest,
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="El MP4 no fue escrito")

    _write_meta(job_id, payload, result.kind, result.engine, result.segments)
    return VideoGenerationResponse(
        success=True,
        job_id=job_id,
        duration_seconds=payload.duration_seconds,
        video_url=_public_url(dest.name),
        message=result.message,
        engine=result.orchestrator if result.orchestrator == "videox-fun" else result.engine,
        segments=result.segments,
    )


def _write_meta(job_id: str, payload: VideoGenerationRequest, kind: str, engine: str, segments: int) -> None:
    meta = {
        "job_id": job_id,
        "kind": kind,
        "engine": engine,
        "segments": segments,
        "prompt": payload.prompt,
        "duration_seconds": payload.duration_seconds,
        "base_engine": payload.base_engine,
    }
    (OUTPUT_DIR / f"{job_id}.json").write_text(json.dumps(meta, ensure_ascii=False))


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
