import torch
from backend.utils import load_sentences_and_embeddings, load_model
import streamlit as st

# Search
def query_search(query: str, n_desc: int, model_name: str):
    model = load_model(model_name)

    # Creating embeddings
    # query_emb = model.encode(query, convert_to_tensor=True)[None, :]
    query_emb = model.encode(query, convert_to_tensor=True)

    embedding_cache_path = f"./embeddings/art-descr-embeddings-{model_name.replace('/', '_')}.pkl"
    dataset_path = "./data/artwork_detail_prepcd_cleaned.csv"

    print("loading embeddings")
    with st.spinner("Encoding the corpus. This might take a while"):
        corpus_texts, corpus_emb = load_sentences_and_embeddings(embedding_cache_path, dataset_path, model, model_name)
    

    # Getting hits
    hits = torch.nn.functional.cosine_similarity(
        query_emb[None, :], corpus_emb, dim=1, eps=1e-8
    )

    corpus_texts["Similarity"] = hits.tolist()

    print(corpus_texts)

    return corpus_texts.sort_values(by="Similarity", ascending=False).head(n_desc)[
        ["artist_name", "artwork_name", "artwork_full_desc"]
    ]
