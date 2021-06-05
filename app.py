# import sys
# from pathlib import  Path
# # Since streamlit doesn't allow a python setup.py install, we have to add the rcfx folder by hand.
# module_path = str(Path().absolute() / "rcfx")
# sys.path.append(module_path)
# sys.path.append("../rcfx")


from librosa.core import audio
from matplotlib.pyplot import plot
import streamlit as st
import matplotlib.pylab as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf

import io
from model import (
    read_audio_fast,
    get_model_predictions_for_clip,
    BIRDS,
    load_pretrained_model,
)


from utils import top_bird_bar_plot
import pandas as pd


@st.cache
def taxonomy_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/gaborfodor/wave-bird-recognition/main/data/taxonomy.csv"
    ).set_index("speciesCode")


tx_df = taxonomy_data()

def plot_mel_scale(sampling_rate=48000):
    fig, ax = plt.subplots(figsize=(6, 6))

    # How many FFT components are kept?
    N_FFT = 2048

    mel_scale_signal = librosa.filters.mel(sampling_rate, N_FFT)

    im = librosa.display.specshow(mel_scale_signal, x_axis="linear", ax=ax)

    ax.set_ylabel("Mel filter")

    ax.set_title("Mel filter bank")

    fig.colorbar(im)

    fig.tight_layout()

    st.pyplot(fig)


def plot_mel_spectogram(audio_signal, sampling_rate=48000):
    fig, ax = plt.subplots(figsize=(6, 8))

    S = librosa.feature.melspectrogram(y=audio_signal, sr=sampling_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=16000, ax=ax
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    ax.set(title="Mel-frequency spectrogram")

    st.pyplot(fig)


def plot_mfccs(audio_signal, sampling_rate=48000):
    S = librosa.feature.melspectrogram(
        y=audio_signal, sr=sampling_rate, n_mels=128, fmax=16000
    )  # 16k is better?

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
    fig, ax = plt.subplots(figsize=(6, 8))

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))

    im = librosa.display.specshow(
        mfccs, x_axis="time", y_axis="mel", ax=ax, sr=48000, fmax=24000, fmin=40
    )

    fig.colorbar(im, format="%+2.0f dB")

    ax.set_title(f"MFCC")
    st.pyplot(fig)


# From here mostly: https://github.com/gaborfodor/wave-bird-recognition/blob/main/birds/run.py
@st.cache(allow_output_mutation=True)
def get_top_birds_classification(y):
    """Predict the 5 most likely bird's species using the PANN model."""
    # TODO: Add a plot.
    model = load_pretrained_model()
    predictions = get_model_predictions_for_clip(y, model)
    class_probs = predictions[BIRDS].sum().reset_index()
    class_probs.columns = ["ebird", "p"]
    class_probs = class_probs.sort_values(by="p")

    top_birds = class_probs.tail(5).reset_index(drop=True)
    return top_birds


uploaded_audio_file = st.sidebar.file_uploader("Upload an audio file")


if uploaded_audio_file is not None:

    st.title("Birds Audio Streamlit App")
    audio_bytes = uploaded_audio_file.read()
    st.sidebar.audio(audio_bytes, format="audio/flac")
    # show = st.sidebar.selectbox("Show Mel scale?", ["yes", "no"])
    predict = st.sidebar.selectbox("Predict the bird's specie?", ["yes", "no"])
    markdown = """

    A [streamlit](https://streamlit.io/) app by [Yassine](https://www.kaggle.com/yassinealouini).

    The source code could be found [here](https://github.com/yassineAlouini/rfcx-species-audio-detection-streamlit).

    Some parts of the app are inspired from this [repo](https://github.com/gaborfodor/wave-bird-recognition).

    """
    st.sidebar.markdown(markdown)



    audio_signal, sampling_rate = sf.read(io.BytesIO(audio_bytes))

    with st.beta_container():
        st.subheader("Some frequency features", )
        col1, col2 = st.beta_columns(2)
        with col1:
            plot_mel_spectogram(audio_signal)
        with col2:
            plot_mfccs(audio_signal)
    
    


    if predict == "yes":
        y = read_audio_fast(audio_signal, sampling_rate)
        # Since this step takes some time (downloading the weights the first time),
        # we add a spinner.
        with st.spinner("The model is computing..."):
            df = get_top_birds_classification(y)

            with st.beta_container():
                top_ebird_code = df.tail(1)["ebird"]
                bird_info = tx_df.loc[top_ebird_code].to_dict()
                st.subheader("Some information for the top bird")
                st.write(bird_info)
            with st.beta_container():
                # Plasma scale 5 colors.
                df["color"] = [
                    "#0d0887",
                    "#46039f",
                    "#7201a8",
                    "#9c179e",
                    "#bd3786",
                ]
                st.subheader("Top 5 predicted birds from the audio clip", )
                st.plotly_chart(top_bird_bar_plot(df, 3, 5))





    # if show == "yes":
    #     with st.beta_container():
    #         plot_mel_scale()
