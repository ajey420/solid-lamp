import pydub
import os 
from base_utilities import format_path
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import streamlit as st
import torchaudio
from tqdm import tqdm
import torch

import streamlit as st
import time
import torchaudio

def get_transcription(model , processor , waveform , sample_rate , cuda = False) : 

    if sample_rate != 16000 : 

        resampler = torchaudio.transforms.Resample(sample_rate , 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    if waveform.shape[0] > 1 : waveform = waveform.mean(dim = 0 , keepdim = True)

    chunk_size = 70000
    total_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size  
    text = ''

    start_time = time.time()
    my_bar = st.progress(0.0)

    for prog_index , index in enumerate(range(0 , waveform.shape[1] , chunk_size)) : 

        wave = waveform[: , index : index + chunk_size]
        input_features = processor(
            wave.numpy() , 
            sampling_rate = 16000 , 
            return_tensors = 'pt'
        ).input_features

        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        text += transcription[0]

        progress = (prog_index + 1) / total_chunks
        my_bar.progress(progress)

        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / progress) * (1 - progress)
        remaining_minutes = int(remaining_time // 60)
        remaining_seconds = int(remaining_time % 60)
        st.write(f'Estimated time remaining: {remaining_minutes}m {remaining_seconds}s')

    return text

# def get_transcription(model , processor , waveform , sample_rate , cuda = False) : 

#     if sample_rate != 16000 :

#         resampler = torchaudio.transforms.Resample(sample_rate , 16000)
#         waveform = resampler(waveform)
#         sample_rate = 16000
#     if waveform.shape[0] > 1 : waveform = waveform.mean(dim = 0 , keepdim = True)

#     chunk_size = 70000
#     length = waveform.shape[1] // 70000
#     text = ''

#     for index in tqdm(range(0 , waveform.shape[1] , chunk_size) , total = length) :

#         wave = waveform[index : , index + 70000]

#         input_features = processor(
#             wave.numpy() ,
#             sampling_rate = 16000 ,
#             return_tensors = 'pt'
#         ).input_features

#         predicted_ids = model.generate(input_features)

#         transcription = processor.batch_decode(
#             predicted_ids ,
#             skip_special_tokens = True
#         )

#         text += transcription[0]

#     return text

def get_transcript_from_mp3(audio_file , model , processor , cuda = False) : 

    with st.spinner('Converting MP3 to WAV') : 

        sound = pydub.AudioSegment.from_mp3(audio_file)
        wav_data = sound.export(format = 'wav')
        wav_bytes = wav_data.read()

        waveform , sample_rate = torchaudio.load(wav_bytes)

    return get_transcription(model , processor , waveform , sample_rate , cuda)
def get_transcript_from_mp4(audio_file , model , processor , cuda = False) : 

    with st.spinner('Converting MP4 to WAV') : 
        video = AudioSegment.from_file(audio_file , format = 'mp4')
        audio_track = video.audio
        wav_data = audio_track.export(format = 'wav')
        wav_bytes = wav_data.read()

        waveform , sample_rate = torchaudio.load(wav_bytes)

    return 'Done'

    return get_transcription(model , processor , waveform , sample_rate , cuda)
def get_transcript_from_wav(audio_file , model , processor , cuda = False) : 

    waveform , sample_rate = torchaudio.load(audio_file)

    return get_transcription(model , processor , waveform , sample_rate , cuda)