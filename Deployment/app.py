import streamlit as st
from audio_recorder_streamlit import audio_recorder
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from os import path
from realwav import Realwav

@st.cache_resource
def decoi():  
  return Realwav()

rw = decoi()

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=16_000)
    return speech_array
        
def eva_speech(sound,ref):
    filesound = speech_file_to_array_fn(sound)
    output = rw.sound_and_sentence(filesound,ref)
    output = round(output*100,2)
    return output

st.title("How to know percent your sound?")
option = st.selectbox(
    'เลือกศัพท์ภาษาจีน?',
    ('你好', '吃饭', '好的'))

st.write('You selected:', option)

transtr = f"./{option}.wav"
# path_sound =  open(transtr, 'rb')
# audio = path_sound.read()
with open(transtr, 'rb') as f:
  audio = f.read()

col1, col2 = st.columns(2)
with col1:
  with st.expander("",True):
    st.subheader('เสียงตัวอย่างจากคน Native Speaker')
    st.audio(audio, format='audio/wav')
    if st.button('Predict native'):
              test2 = eva_speech(transtr,option)
              st.write("เปอร์เซ็นการออกเสียงของคน Native Speaker!!")    
              st.write(test2)

with col2:
  with st.expander("",True):
    st.subheader('เสียงพูดของคุณ')
    audio_bytes = audio_recorder(sample_rate=16_000)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # wav_file = open("./audio.wav", "wb")
        # wav_file.write(audio_bytes)
        with open("./audio.wav", "wb") as f:
          f.write(audio_bytes)
        
        if st.button('Predict'):
            test = eva_speech("./audio.wav",option)
            
            st.write("เปอร์เซ็นการออกเสียงของคุณ!!")
            st.write(test)

      
