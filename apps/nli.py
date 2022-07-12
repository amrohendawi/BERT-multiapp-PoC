from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import streamlit as st
from streamlit_tags import st_tags_sidebar

def app():

    # create streamlit app
    st.title("Natural Language Inference")
    st.markdown("This app demonstrates how to use the Natural Language Inference")
