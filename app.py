import os
import streamlit as st
from streamlit_image_select import image_select
from st_audiorec import st_audiorec
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
                              }
def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    st.title("Welcome to AI Lip Sync :)")
    avatar_img = image_select("Choose your Avatar", 
							  ["avatars_images/avatar1.jpg"])
    data["imge_path"] = avatar_img
    #wav_audio_data = st_audiorec()
    #st.write(wav_audio_data)
    #col_playback, col_space = st.columns([0.58,0.42])
    #with col_playback:
    #        st.audio(wav_audio_data, format='audio/wav')
    #if wav_audio_data is not None:
    #      st.audio(wav_audio_data, format='audio/wav')
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
    save_record = st.button("save record")
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

if __name__ == "__main__":
    main()