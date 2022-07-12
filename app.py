import streamlit as st

# Custom imports 
from multiapp import Multiapp
from apps import sts, nli

app = Multiapp()

# Title of the main app
st.title("BERT downstream tasks demo")

# Add all your application here
app.add_app("Semantic Textual Similarity", sts.app)
app.add_app("Natural Language Inference", nli.app)

# The main app
app.run()
