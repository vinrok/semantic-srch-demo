import streamlit as st
from PIL import Image

from backend import inference

st.title("Semantic search for artwork descriptions based on query via SBert Models for Semantic Similarity.")



anchor = st.text_input(
    "Please enter the query and based on that relevant descriptions will be procured from the data.",
    value="Enter the query",
)

n_desc = st.number_input(
    f"""How many similar descriptions to be displayed?""", value=2, min_value=1
)

MODEL_OPTIONS = ['all-mpnet-base-v2', 'distilbert-base-nli-mean-tokens', 
                'msmarco-distilbert-base-v4', 'all-MiniLM-L6-v2']

model_choice = st.sidebar.selectbox("Select models from below: ", options=MODEL_OPTIONS)

if st.button("Find them....."):

    st.table(
        inference.query_search(
            anchor, n_desc, f"sentence-transformers/{model_choice}"
        )
    )

