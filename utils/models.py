import streamlit as st
from huggingface_hub import hf_hub_download

@st.cache_resource(show_spinner="Chargement du modèle Inception V3...")
def load_inception_model():
    from tensorflow.keras.models import load_model
    model_path = hf_hub_download(repo_id="AsmaSima/inception_v3_covid",
                                 filename="inceptionV3_covid.keras")
    return load_model(model_path, compile=False)

@st.cache_resource(show_spinner="Chargement du modèle de masquage...")
def load_mask_model(device="cpu"):
    import torch
    model_path = hf_hub_download(repo_id="AsmaSima/mask_auto",
                                 filename="mask_auto.pt")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

