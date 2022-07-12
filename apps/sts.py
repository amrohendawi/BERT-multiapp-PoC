from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import streamlit as st
from streamlit_tags import st_tags_sidebar

def app():

    # create streamlit app
    st.title("Semantic Textual Similarity demo")
    st.markdown("This app demonstrates how to use the SentenceTransformer model to compute the semantic textual similarity between two sentences.")  # Markdown

    # Load model from HuggingFace Hub
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: green;
    }
    </style>
    """, unsafe_allow_html=True)

    # Read multiple sentences from the user
    # define empty list
    sentences = []
    # A box for the source sentence
    st.sidebar.title("Source sentence")
    source_sentence = st.sidebar.text_input(
        "Enter a source sentence:", value="Dogs are cute.")
    # A box for the sentences to compare to the source sentence and a button to add more sentences
    st.sidebar.title("Sentences to compare to the source sentence")

    sentences = st_tags_sidebar(
        label='Enter sentences for comparison:',
        text='Press enter to add',
        value=['Dogs are funny', 'Dogs are sweet', 'Nothing is cuter than dogs',
            'All animals are cute', 'All animals deserve to be loved', 'All children are born innocent', 'I like tuna sandwiches'],
        suggestions=['Cats don\'t like to be compared',
                    'Humans like to be compared'],
        maxtags=10,
        key='1')

    # A button to compute the semantic textual similarity and output the results
    if st.sidebar.button("Compute semantic textual similarity"):
        # Append the source sentence to the beginning of the list
        sentences.insert(0, source_sentence)
        # print sentences and source sentence
        st.write("Sentences to compare to the source sentence:", sentences)
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True,
                                truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # Compute cosine similarity between source sentence and other sentences
        cosine_similarity = torch.nn.functional.cosine_similarity(
            sentence_embeddings[0], sentence_embeddings[1:])
        # output every sentence with a percentage bar of the similarity as progress bar
        for i in range(len(sentences)-1):
            st.write(sentences[i+1], ": ", cosine_similarity[i].item())
            st.progress(
                int(100 * cosine_similarity[i].item()) if cosine_similarity[i].item() > 0 else 0)