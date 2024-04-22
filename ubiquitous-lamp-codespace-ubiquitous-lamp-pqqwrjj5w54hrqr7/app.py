from transformers import WhisperProcessor, WhisperForConditionalGeneration
from CFG import CFG
import streamlit as st
from helper import *
import torch

cuda = torch.cuda.is_available()

processor = WhisperProcessor.from_pretrained(CFG.model_path)
model = WhisperForConditionalGeneration.from_pretrained(CFG.model_path)

audio_file = st.file_uploader(
    'Upload an audio file' , 
    type = ['wav' , 'mp3' , 'mp4'])

if audio_file : 

    if audio_file.name.endswith('wav') : transcription = get_transcript_from_wav(audio_file , model , processor , cuda)
    elif audio_file.name.endswith('mp3') : transcription = get_transcript_from_mp3(audio_file , model , processor , cuda)
    elif audio_file.name.endswith('mp4') : transcription = get_transcript_from_mp4(audio_file , model , processor , cuda)

    st.write(transcription)
