import pandas as pd
import streamlit as st
import torch
import csv
import pickle
import itertools
import os
from sentence_transformers import SentenceTransformer


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    # Lazy downloading
    model = SentenceTransformer(model_name)
    return model


def lower_case(iterator):
    return itertools.chain([next(iterator).lower()], iterator)


def load_sentences_and_embeddings(embedding_cache_path, dataset_path, model):
    max_corpus_size = 100000

    if not os.path.exists(embedding_cache_path):
        # Check if the dataset exists.
        if not os.path.exists(dataset_path):
            print("Path doesn't exist")
        else:
            # Get all unique sentences from the file
            corpus_sentences = set()
            with open(dataset_path, encoding='utf8') as fIn:
                reader = csv.DictReader(lower_case(fIn))
                for row in reader:
                # print('row: ', row)
                    corpus_sentences.add(row['artwork_full_desc'])
                    if len(corpus_sentences) >= max_corpus_size:
                        break


            corpus_sentences = list(corpus_sentences)
            print("Encode the corpus. This might take a while")
            corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

            print("Store file on disc")
            with open(embedding_cache_path, "wb") as fOut:
                pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    else:
        print("Load pre-computed embeddings from disc")
            # embedding pre-generated
        embedding_data = torch.load(
            embedding_cache_path,
            map_location=torch.device("cpu"),
        )

        corpus_sentences = embedding_data['sentences']
        corpus_embeddings = embedding_data['embeddings']
   
    
    return corpus_sentences, corpus_embeddings


# @st.cache(allow_output_mutation=True)
# def load_texts(data_path):
#     # texts database pre-generated
#     corpus_texts = pd.read_csv("./data/artwork_detail_prepcd_cleaned.csv")
#     return corpus_texts
