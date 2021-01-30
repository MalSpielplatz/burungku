from librosa.core import audio
from matplotlib.pyplot import plot
import streamlit as st
import matplotlib.pylab as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf

import io

def plot_mel_scale(sampling_rate=48000):
    fig, ax = plt.subplots(figsize=(6, 6))

    # How many FFT components are kept?
    N_FFT = 2048

    mel_scale_signal = librosa.filters.mel(sampling_rate, N_FFT)

    im = librosa.display.specshow(mel_scale_signal, x_axis='linear', ax=ax)

    ax.set_ylabel('Mel filter')

    ax.set_title('Mel filter bank')

    fig.colorbar(im)

    fig.tight_layout()

    st.pyplot(fig)

def plot_mel_spectogram(audio_signal, sampling_rate=48000):
    fig, ax = plt.subplots(figsize=(15, 5))

    S = librosa.feature.melspectrogram(y=audio_signal, sr=sampling_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)


    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sampling_rate,
                                fmax=16000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Mel-frequency spectrogram')

    st.pyplot(fig)

def plot_mfccs(audio_signal, sampling_rate=48000):
    S = librosa.feature.melspectrogram(y=audio_signal, sr=sampling_rate, n_mels=128,
                                    fmax=16000) # 16k is better?

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    fig, ax = plt.subplots(figsize=(15, 5))


    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))

    im = librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', ax=ax, sr=48000,
                                fmax=24000, fmin=40)


    fig.colorbar(im, format='%+2.0f dB')


    ax.set_title(f'MFCC')
    st.pyplot(fig)

uploaded_audio_file = st.sidebar.file_uploader("Upload an audio file")
if uploaded_audio_file is not None:
    audio_bytes = uploaded_audio_file.read()
    st.sidebar.audio(audio_bytes, format='audio/flac')

    audio_signal, sampling_rate =  sf.read(io.BytesIO(audio_bytes))


    plot_mel_spectogram(audio_signal)
    plot_mfccs(audio_signal)
    show = st.sidebar.selectbox("Show Mel scale?", ["yes", "no"])
    if show == "yes":
        plot_mel_scale()

