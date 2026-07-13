import io
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from baselines import BaselineName, run_baseline

app = FastAPI(title="DSDNet Receipt Recovery API")


@app.get("/", response_class=HTMLResponse)
def homepage():
    return HTMLResponse(
        """
        <html>
            <head>
                <title>DSDNet Receipt Recovery</title>
            </head>
            <body>
                <h1>Upload a Receipt Image</h1>
                <form action="/upload" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="image/*" />
                    <button type="submit">Upload</button>
                </form>
            </body>
        </html>
        """
    )


@app.post("/upload")
async def upload_receipt(file: UploadFile = File(...), baseline: Optional[str] = Form("Proposed")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    temp_path = Path("temp_upload.png")
    image.save(temp_path)
    if baseline in BaselineName.__members__:
        baseline_name = BaselineName[baseline]
    else:
        try:
            baseline_name = BaselineName(baseline)
        except ValueError:
            baseline_name = BaselineName.PROPOSED
    result = run_baseline(str(temp_path), baseline_name)
    response = {
        "baseline": result.baseline.value,
        "fields": result.fields,
        "abstained_fields": result.abstained_fields,
        "ocr_text": result.ocr_text,
        "notes": result.notes,
    }
    return JSONResponse(response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
