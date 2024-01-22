import os
import streamlit as st
from streamlit_image_select import image_select
import torch
from streamlit_mic_recorder import mic_recorder
from wav2lip import inference 
from wav2lip.models import Wav2Lip

device='cpu'
#@st.cache_data is used to only load the model once
@st.cache_data 
def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
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
                    "avatars_images/avatar2.jpg":"avatars_videos/avatar2.mp4",
                    "avatars_images/avatar3.png":"avatars_videos/avatar3.mp4"
                              }
def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    st.title("Welcome to AI Lip Sync :)")
    st.write("Please choose your avatar from the following options:")
    avatar_img = image_select("", 
							  ["avatars_images/avatar1.jpg",
          						"avatars_images/avatar2.jpg",
                                "avatars_images/avatar3.png",
                                        ])
    data["imge_path"] = avatar_img
    audio=mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording", 
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None)
    if audio:
          st.audio(audio["bytes"])
          data["audio"]= audio["bytes"]
    return data

def main():  
    data=streamlit_look()
    st.write("Don't forget to save the record or there will be an error!")
    save_record = st.button("save record")
    st.write("With fast animation only the lips of the avatar will move, and it will take probably less than a minute for a record of about 30 seconds, but with fast animation choise, the full face of the avatar will move and it will take about 30 minute for a record of about 30 seconds to get ready.")
    fast_animate = st.button("fast animate")
    slower_animate = st.button("slower animate")
    if save_record:
         if os.path.exists('record.wav'):
              os.remove('record.wav') 
         with open('record.wav', mode='bx') as f:
          f.write(data["audio"])
         st.write("record saved!")
    if fast_animate:
        inference.main('wav2lip/checkpoints/wav2lip_gan.pth',data["imge_path"],"record.wav",model)
        if os.path.exists('wav2lip/results/result_voice.mp4'):
             st.video('wav2lip/results/result_voice.mp4')
    if slower_animate:
        inference.main('wav2lip/checkpoints/wav2lip_gan.pth',image_video_map[data["imge_path"]],"record.wav",model)
        if os.path.exists('wav2lip/results/result_voice.mp4'):
             st.video('wav2lip/results/result_voice.mp4') 

if __name__ == "__main__":
    main()