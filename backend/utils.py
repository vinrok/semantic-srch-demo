import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    # Lazy downloading
    model = SentenceTransformer(model_name)
    return model


def load_embeddings():
    # embedding pre-generated
    embedding_data = torch.load(
        "./embeddings/art-descr-embeddings-all-mpnet-base-v2.pkl",
        map_location=torch.device("cpu"),
    )

    corpus_sentences = embedding_data['sentences']
    corpus_embeddings = embedding_data['embeddings']
    
    return corpus_embeddings


@st.cache(allow_output_mutation=True)
def load_texts():
    # texts database pre-generated
    corpus_texts = pd.read_csv("./data/artwork_detail_prepcd_cleaned.csv")
    return corpus_texts
