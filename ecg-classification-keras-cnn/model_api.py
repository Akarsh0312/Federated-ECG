from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

MODEL_PATH = "model/global_model.h5"

@app.get("/get-model/")
def get_model():
    return FileResponse(MODEL_PATH, media_type="application/octet-stream")