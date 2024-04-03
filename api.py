from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, argparse
import sys
import numpy as np
from PIL import Image
import uuid
import shutil
from scripts.gradio.i2v_test import Image2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

class DynamiCrafterAPI:
    def __init__(self):
        self.running = False
        self.upload_dir = './tmp/upload/'
        self.result_dir = './tmp/result/'
        self.image2video = None
        self.resolution = '576_1024'
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        print("Initializing model...")
        self.init_model(res=1024)
        print("Model initialized.")

    def init_model(self, res=1024):
        if res == 1024:
            self.resolution = '576_1024'
        elif res == 512:
            self.resolution = '320_512'
        elif res == 256:
            self.resolution = '256_256'
        else:
            raise NotImplementedError(f"Unsupported resolution: {res}")
        self.image2video = Image2Video(self.result_dir, resolution=self.resolution)
    
    def run(self, image_path, i2v_input_text, i2v_steps=50, i2v_cfg_scale=7.5, i2v_eta=1.0, i2v_motion=15, i2v_seed=123):
        i2v_input_image_array = np.array(Image.open(image_path))
        i2v_output_video = self.image2video.get_image(i2v_input_image_array, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed, save_name=str(uuid.uuid4()))
        return i2v_output_video

app = FastAPI()

dynamiCrafterAPI = DynamiCrafterAPI()

class GenerateRequest(BaseModel):
    image_name: str
    i2v_input_text: str
    i2v_steps: int = 50
    i2v_cfg_scale: float = 7.5
    i2v_eta: float = 1.0
    i2v_motion: int = 15
    i2v_seed: int = 123

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    file_path = os.path.join(dynamiCrafterAPI.upload_dir, filename)
    with open(file_path, "wb") as save_file:
        shutil.copyfileobj(file.file, save_file)
    print(f"Uploaded file: {file_path}")
    return {"filename": filename}
@app.post("/generate")
async def generate(request: GenerateRequest):
    print(f"Generating video for {request.image_name}")
    result = dynamiCrafterAPI.run(os.path.join(dynamiCrafterAPI.upload_dir, request.image_name), request.i2v_input_text, request.i2v_steps, request.i2v_cfg_scale, request.i2v_eta, request.i2v_motion, request.i2v_seed)
    print(f"Video generated for {request.image_name}")
    return FileResponse(result)