import torch

from backend.utils import load_embeddings, load_model, load_texts

# Search
def query_search(query: str, n_desc: int, model_name: str):
    model = load_model(model_name)

    # Creating embeddings
    # query_emb = model.encode(query, convert_to_tensor=True)[None, :]
    query_emb = model.encode(query, convert_to_tensor=True)

    print("loading embedding")
    corpus_emb = load_embeddings()
    corpus_texts = load_texts()

    # Getting hits
    hits = torch.nn.functional.cosine_similarity(
        query_emb[None, :], corpus_emb, dim=1, eps=1e-8
    )

    corpus_texts["Similarity"] = hits.tolist()

    print(corpus_texts)

    return corpus_texts.sort_values(by="Similarity", ascending=False).head(n_desc)[
        ["artist_name", "artwork_name", "artwork_full_desc"]
    ]
