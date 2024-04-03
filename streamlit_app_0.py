import streamlit as st
import os, argparse
import sys
import numpy as np
from PIL import Image
from scripts.gradio.i2v_test import Image2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

i2v_examples_1024 = [
    ['prompts/1024/astronaut04.png', 'a man in an astronaut suit playing a guitar', 50, 7.5, 1.0, 6, 123],
    ['prompts/1024/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 10, 123],
    ['prompts/1024/girl07.png', 'a beautiful woman with long hair and a dress blowing in the wind', 50, 7.5, 1.0, 10, 123],
    ['prompts/1024/pour_bear.png', 'pouring beer into a glass of ice and beer', 50, 7.5, 1.0, 10, 123],
    ['prompts/1024/robot01.png', 'a robot is walking through a destroyed city', 50, 7.5, 1.0, 10, 123],
    ['prompts/1024/firework03.png', 'fireworks display', 50, 7.5, 1.0, 10, 123],
]

i2v_examples_512 = [
    ['prompts/512/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/campfire.png', 'a bonfire is lit in the middle of a field', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/isometric.png', 'rotating view, small house', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/girl08.png', 'a woman looking out in the rain', 50, 7.5, 1.0, 24, 1234],
    ['prompts/512/ship02.png', 'a sailboat sailing in rough seas with a dramatic sunset', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/zreal_penguin.png', 'a group of penguins walking on a beach', 50, 7.5, 1.0, 20, 123],
]

i2v_examples_256 = [
    ['prompts/256/art.png', 'man fishing in a boat at sunset', 50, 7.5, 1.0, 3, 234],
    ['prompts/256/boy.png', 'boy walking on the street', 50, 7.5, 1.0, 3, 125],
    ['prompts/256/dance1.jpeg', 'two people dancing', 50, 7.5, 1.0, 3, 116],
    ['prompts/256/fire_and_beach.jpg', 'a campfire on the beach and the ocean waves in the background', 50, 7.5, 1.0, 3, 111],
    ['prompts/256/girl3.jpeg', 'girl talking and blinking', 50, 7.5, 1.0, 3, 111],
    ['prompts/256/guitar0.jpeg', 'bear playing guitar happily, snowing', 50, 7.5, 1.0, 3, 122]
]

if 'run_button' in st.session_state and st.session_state.run_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

# 初始化模型
@st.cache_resource
def init_model(res=1024):
    if res == 1024:
        resolution = '576_1024'
    elif res == 512:
        resolution = '320_512'
    elif res == 256:
        resolution = '256_256'
    else:
        raise NotImplementedError(f"Unsupported resolution: {res}")
    image2video = Image2Video(result_dir, resolution=resolution)
    return image2video


# 使用streamlit重写def dynamicrafter_demo
def dynamicrafter_demo(image2video, result_dir='./tmp/', res=1024):
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
        i2v_input_image_array = np.array(Image.open(i2v_input_image))
        i2v_output_video = image2video.get_image(i2v_input_image_array, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed)
        st.session_state.output_video = i2v_output_video
    if 'output_video' in st.session_state:
        st.video(st.session_state.output_video)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1024, choices=[1024,512,256], help="select the model based on the required resolution")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_dir = os.path.join('./', 'results')
    image2video = init_model(args.res)
    dynamicrafter_demo(image2video, result_dir, args.res)
