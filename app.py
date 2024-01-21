import streamlit as st
from streamlit_image_select import image_select
import torch

from wav2lip import inference 
from wav2lip.models import Wav2Lip
device='cpu'
#@st.cache_data is used to only load the model once
@st.cache_data 
def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	if device == 'cuda':
		checkpoint = torch.load(path)
	else:
		checkpoint = torch.load(path,
								map_location=lambda storage, loc: storage)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

model = load_model('wav2lip/checkpoints/wav2lip_gan.pth')

image_video_map = {
      				"avatars_images/avatar1.jpg":"avatars_videos/avatar1.mp4",
                              }
def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    st.title("Welcome to AI Lip Sync :)")
    avatar_img = image_select("Choose your Avatar", ["avatars_images/avatar1.jpg"])
    data["imge_path"]=avatar_img
    return data

def main():  
    data=streamlit_look()
    fast_animate = st.button("fast animate")
    slower_animate = st.button("slower animate")
    if fast_animate:
        inference.main('wav2lip/checkpoints/wav2lip_gan.pth',data["imge_path"],"AI4.wav",model) 
    if slower_animate:
        inference.main('wav2lip/checkpoints/wav2lip_gan.pth',"avatars/w.mp4","AI4.wav",model)  

if __name__ == "__main__":
    main()