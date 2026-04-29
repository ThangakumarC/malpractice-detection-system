from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
from PIL import Image
import io

from inference import LightweightMalpracticeModel, predict_clip

app = FastAPI()

device = torch.device("cpu")
model = LightweightMalpracticeModel().to(device)

checkpoint = torch.load("best_model_v4.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    # convert bytes → numpy frames (you will send npy)
    frames = np.load(io.BytesIO(data))

    prob = predict_clip(frames)
    return {"prob": float(prob)}