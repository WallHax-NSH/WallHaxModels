from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import sys
import tempfile

# Add 3detr directory to sys.path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), '3detr'))
# from detector import run_detection
from 3detr import run_detection

app = FastAPI()

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Write uploaded file data to temp file
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Run detection on the temp file
        result = run_detection(tmp_path)
        # Clean up temp file
        os.unlink(tmp_path)
        return result
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)