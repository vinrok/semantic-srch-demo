import pandas as pd
from zmq import device
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


def load_sentences_and_embeddings(embedding_cache_path, dataset_path, model, model_name):
    max_corpus_size = 100000
    splitted_path='-'.join(str(embedding_cache_path).split("/")[2].split(".")[0].split("-")[3:])
    corpus_df = pd.DataFrame()

    # Check if the dataset exists.
    if os.path.exists(dataset_path):
        corpus_df = pd.read_csv(dataset_path)
        
        #Check if embedding cache path exists
        if not os.path.exists(embedding_cache_path):
        # if splitted_path != model_name:
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
            corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

            print("Store file on disc")
            with open(embedding_cache_path, "wb") as fOut:
                pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
              
        else:
            print("Load pre-computed embeddings from disc")
            with open(embedding_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_sentences = cache_data['sentences']
                corpus_embeddings = cache_data['embeddings']

    else:
        print("Path doesn't exist")
   
    
    return corpus_df, corpus_embeddings

