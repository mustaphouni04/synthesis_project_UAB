# IMPORTING NECESSARY LIBRARIES
#_________________________________________________________________________________________________________
import pandas as pd
import numpy as np
import nltk
import re
import sqlite3
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from string import punctuation
from keras.preprocessing import text
from urllib.parse import unquote
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader
from gensim.models import Word2Vec
import multiprocessing
import os

# HELPER FUNCTIONS
#_________________________________________________________________________________________________________
def preprocess_referer(ref):
    ref = re.sub(r'[/.-:]', ' ', ref)
    tokens = word_tokenize(ref)
    return tokens

def preprocess_url(url):
    url = re.sub(r'[/.-]', ' ', url)
    tokens = word_tokenize(url)
    return tokens
    
def embeddings(df: pd.DataFrame, preprocess, feature, emb_dim = 300, epochs = 10, window = 5) -> pd.DataFrame:
    # load data
    conn = sqlite3.connect('generic.db')
    df.to_sql('generic', conn, index=False, if_exists='replace')

    # tokenize feature and create corpus
    corpus_path = "corpus"
    gen_tokens = []

    chunk_size = 10000
    for chunk in pd.read_sql_query(f'SELECT {feature} FROM generic', conn, chunksize=chunk_size):
        feats = chunk[feature]
        for feat in feats:
            decoded_feat = unquote(feat)
            tokens = preprocess(decoded_feat)
            gen_tokens.append(tokens)

    conn.close()

    db_file = 'generic.db'
    os.remove(db_file)

    # getting the embeddings
    EMB_DIM = emb_dim
    w2v = Word2Vec(sentences=gen_tokens, vector_size=EMB_DIM, window=window, min_count=5, negative=15,
               epochs=epochs, workers=multiprocessing.cpu_count())

    word_vectors = w2v.wv

    # get the vocabulary
    vocab = list(word_vectors.key_to_index.keys())

    # get the word vectors
    word_vecs = [word_vectors[word] for word in vocab]

    # convert word_vecs to a numpy array
    word_vecs_np = np.array(word_vecs)

    # create a DataFrame for the embeddings
    embeddings_df = pd.DataFrame(word_vecs_np, index=vocab)

    # merge embeddings with the original DataFrame
    df_with_embeddings = pd.merge(df, embeddings_df, left_on=feature, right_index=True, how='left')

    return df_with_embeddings

# PREPROCESSING FUNCTIONS
#_________________________________________________________________________________________________________
def preprocessing_TS(df: pd.DataFrame) -> pd.DataFrame:
    """
    TS: Cyclic encoding of timestamp.
    """
    # convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S %z')

    # extract hour, minute, and second components
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second

    # encode hour, minute, and second with sine and cosine functions
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)

    # drop original timestamp and hour, minute, and second columns
    df.drop(['timestamp', 'hour', 'minute', 'second'], axis=1, inplace=True)

    return df

def preprocessing_SC(df: pd.DataFrame) -> pd.DataFrame:
    """
    SC: One-Hot encoding status_code.
    """
    one_hot_encoded = pd.get_dummies(df['status_code'], prefix='status_code')
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    df_encoded.drop('status_code', axis=1, inplace=True)

    return df_encoded

def preprocessing_RM(df: pd.DataFrame) -> pd.DataFrame:
    """
    RM: One-Hot encoding request_method.
    """
    one_hot_encoded = pd.get_dummies(df['request_method'], prefix='request_method')
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    df_encoded.drop('request_method', axis=1, inplace=True)

    return df_encoded

def preprocessing_RH(df: pd.DataFrame) -> pd.DataFrame:
    """
    RH: Normalizing remote_host and splitting into octets.
    """
    # split the IP address into four separate columns
    df[['octet1', 'octet2', 'octet3', 'octet4']] = df['remote_host'].str.split('.', expand=True)
    df[['octet1', 'octet2', 'octet3', 'octet4']] = df[['octet1', 'octet2', 'octet3', 'octet4']].apply(pd.to_numeric)
    df[['octet1', 'octet2', 'octet3', 'octet4']] /= 255

    return df

def preprocessing_R(df: pd.DataFrame) -> pd.DataFrame:
    """
    R: Generating embeddings for referer.
    """
    # drop all - 
    df = df[df['referer'] != '-']
    
    df_with_embeddings = embeddings(df, preprocess_referer, feature = 'referer', emb_dim = 300, epochs = 10, window = 5)  

    return df_with_embeddings

def preprocessing_RU(df: pd.DataFrame) -> pd.DataFrame:
    """
    RU: Generating embeddings for requested_url.
    """   
    df_with_embeddings = embeddings(df, preprocess_url, feature = 'requested_url', emb_dim = 300, epochs = 10, window = 5)  

    return df_with_embeddings

def preprocessing_BS(df: pd.DataFrame) -> pd.DataFrame:
    """
    BS: Normalizing bytes_sent.
    """
    raise NotImplementedError

def preprocessing_UA(df: pd.DataFrame) -> pd.DataFrame:
    """
    UA: Generating embeddings for user_agent.
    """
    raise NotImplementedError

def preprocessing_CLEAN(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLEAN: Drop all unnecessary columns.
    """
    raise NotImplementedError