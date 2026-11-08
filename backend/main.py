"""FastAPI backend. Step 1: /check-accent accepts audio and returns a stub.
Step 2-3: the same endpoint runs the accent model and the phonetic check.

Run the stub (no models needed):   STUB=1 uvicorn backend.main:app --reload
Run the full version:              uvicorn backend.main:app --reload
(On Windows PowerShell:  $env:STUB="1"; uvicorn backend.main:app --reload)
"""
import os
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="Accent Checker")
STUB = os.getenv("STUB", "0") == "1"


def to_wav16k(src: str, folder: str) -> str:
    """Convert any uploaded audio to 16 kHz mono WAV with ffmpeg."""
    dst = os.path.join(folder, "clip16k.wav")
    cmd = ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst):
        raise HTTPException(400, "Could not read the audio file. Use WAV, MP3, M4A or OGG.")
    return dst


@app.get("/health")
def health():
    return {"status": "ok", "stub": STUB}


@app.post("/check-accent")
async def check_accent(audio: UploadFile = File(...), reference_text: str = Form("")):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    # Temporary folder is deleted automatically: recordings of students are not kept.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + suffix)
        with open(path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        size = os.path.getsize(path)
        if size == 0:
            raise HTTPException(400, "The audio file is empty.")

        if STUB:  # Step 1: placeholder answer
            return {
                "stub": True,
                "message": "Audio received. Models are not connected yet.",
                "filename": audio.filename,
                "bytes": size,
            }

        wav = to_wav16k(path, tmp)
        # imported here so the stub works without torch installed
        from .accent_model import classify_accent
        result = {"stub": False, "accents": classify_accent(wav)}

        if reference_text.strip():
            from .phonetics import analyze_pronunciation
            result["pronunciation"] = analyze_pronunciation(wav, reference_text)
        return result
