from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil
import os

app = FastAPI()

# Load model once at startup
model = WhisperModel("base", compute_type="int8")  # Try "small" for better accuracy

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    
    # Save uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Transcribe
    segments, _ = model.transcribe(temp_path)
    transcript = "".join([segment.text for segment in segments])

    # Clean up temp file
    os.remove(temp_path)

    return {
        "text": transcript
    }
