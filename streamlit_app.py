# 调用api.py的接口的streamlit应用

# Path: DynamiCrafter/streamlit_app.py
import streamlit as st
import os
import numpy as np
from PIL import Image
import requests
import json

if 'run_button' in st.session_state and st.session_state.run_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

class DynamiCrafterUI:
    def __init__(self):
        self.api_url = "http://127.0.0.1:8000/"
        self.upload_url = self.api_url + "upload"
        self.generate_url = self.api_url + "generate"

    def upload_image(self, bytes_data):
        files = {'file': bytes_data}
        try:
            response = requests.post(self.upload_url, files=files)
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)
            return {"filename": None}
    
    def generate_video(self, filename, i2v_input_text, i2v_steps=50, i2v_cfg_scale=7.5, i2v_eta=1.0, i2v_motion=15, i2v_seed=123, frames=None):
        data = {
            "image_name": filename,
            "i2v_input_text": i2v_input_text,
            "i2v_steps": i2v_steps,
            "i2v_cfg_scale": i2v_cfg_scale,
            "i2v_eta": i2v_eta,
            "i2v_motion": i2v_motion,
            "i2v_seed": i2v_seed,
            "frames": frames
        }
        try:
            response = requests.post(self.generate_url, json=data)
            return response.content
        except requests.exceptions.RequestException as e:
            logging.error(e)
            return None
    
    def run(self):
        st.title("DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors")
        st.markdown("## Image2Video")
        i2v_input_image = st.file_uploader("Input Image", type=['png', 'jpg', 'jpeg'])
        if i2v_input_image:
            st.image(i2v_input_image, width=512)
        i2v_input_text = st.text_input("Prompts")
        i2v_seed = st.slider("Random Seed", 0, 10000, 123)
        i2v_eta = st.slider("ETA", 0.0, 1.0, 1.0)
        i2v_cfg_scale = st.slider("CFG Scale", 1.0, 15.0, 7.5)
        i2v_steps = st.slider("Sampling steps", 1, 60, 50)
        i2v_motion = st.slider("FPS", 5, 20, 10)
        i2v_end_btn = st.button("Generate", disabled=st.session_state.running, key='run_button')
        if i2v_end_btn:
            filename = self.upload_image(i2v_input_image.read())["filename"]
            if filename:
                video = self.generate_video(filename, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed, frames=40)
                if video:
                    st.session_state.video = video
                else:
                    st.error("Failed to generate video.")
            else:
                st.error("Failed to upload image.")
        if "video" in st.session_state:
            st.video(st.session_state.video)

if __name__ == "__main__":
    DynamiCrafterUI().run()