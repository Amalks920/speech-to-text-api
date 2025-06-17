from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from transformers import pipeline
import shutil
import os

unmasker = pipeline('fill-mask', model='distilbert-base-uncased')

app = FastAPI()
print(unmasker("Hello I'm a [MASK] model."))

origins = [
    "http://localhost:5173",  # or wherever your frontend runs
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allow these origins
    allow_credentials=True,
    allow_methods=["*"],             # Allow all HTTP methods
    allow_headers=["*"],             # Allow all headers
)

# Load model once at startup
model = WhisperModel("base", compute_type="int8")  # Try "small" for better accuracy

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    print(file.filename)
    temp_path = f"temp_{file.filename}"
    
    # Save uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Transcribe
    segments, _ = model.transcribe(temp_path)
    transcript = "".join([segment.text for segment in segments])
    print(transcript)
    # Clean up temp file
    os.remove(temp_path)

    return {
        "text": transcript
    }

@app.get("/v1/intent_detection")
def fill_mask(sentence: str = Query(..., description="Sentence with [MASK] token")):
    try:
        # Run the prediction
        results = unmasker(sentence)

        # Return top 5 predictions
        return [
            {
                "sequence": result["sequence"],
                "token_str": result["token_str"],
                "score": round(result["score"], 4)
            }
            for result in results
        ]

    except Exception as e:
        return {"error": str(e)}