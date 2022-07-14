import csv
import pandas as pd
import numpy as np

import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import plotly.express as px
import json

import spacy
import string 

nlp = spacy.load("en_core_web_sm")

from gensim import corpora
from gensim import models
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

def read_network(file_path):
    f = open(file_path, encoding='utf-8-sig') # delete descripcion:
    network = json.load(f)["network"]
    network_items = pd.DataFrame(network["items"])
    network_items["DOI"] = network_items["url"].apply(lambda x: x.replace("https://doi.org/", "").lower() if str(x) != "nan" else np.nan)
    network_items["Citations"] = network_items["weights"].apply(lambda x: x["Citations"])
    return network_items

def read_bibliography(file_path):
    df_scopus = pd.read_csv(file_path)
    df_scopus = df_scopus.reset_index().rename(columns={"index": "id"})
    return df_scopus

def get_docs(df_net, df_biblio):
    relevant_columns = {
        "LSA": "Title  Abstract  Author Keywords  Index Keywords".split("  "),
        "others": "Year  Authors".split("  ")
    }
    df_docs = df_biblio[["id"]+relevant_columns["LSA"]+relevant_columns["others"]]
    df_docs = df_docs.merge(df_net, how="inner", on="id")
    df_docs["Abstract"] = df_docs["Abstract"].replace({"[No abstract available]": np.nan})
    df_docs = df_docs.fillna("")
    df_docs["doc"] = df_docs.apply(lambda x: ". ".join(x[relevant_columns["LSA"]].values), axis=1)
    return df_docs

def plot_frec_clusters(df_docs):
    num_clusters = df_docs["cluster"].unique().shape[0]
    table_hist = pd.DataFrame()
    table_hist["Frequency"] = df_docs["cluster"].value_counts().values
    table_hist["Cluster"] = [str(i) for i in range(1, num_clusters+1)]
    color_discrete_sequence = ['#d63f4b', 
                               "#58bd5b", 
                               "#5599c3", 
                               "#bcbe4f", 
                               "#996fc0", 
                               "#5cccd9"][:num_clusters]
    fig = px.bar(table_hist, 
                 x="Cluster", 
                 y="Frequency", 
                 color="Cluster", 
                 title="Frecuency of clusters (LinLog/mod.)", 
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()

def preprocess(text):
    tokens = []
    for token in nlp(text):
        val = token.text
        if val not in string.punctuation+"'":
            if not token.is_stop:
                if "x" in token.shape_.lower():
                    tokens.append(token.lemma_.lower())
    bi_tokens = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    return tokens+bi_tokens

def add_preprocess(df_docs):
    if "doc_clean" not in df_docs:
        df_docs['doc_clean'] = df_docs['doc'].apply(lambda x: preprocess(x))
    pass

def LSA(df_docs, cluster_id, num_topics):
    df_cluster = df_docs[df_docs["cluster"] == cluster_id]
    corpus = df_cluster['doc_clean']
    dictionary = corpora.Dictionary(corpus)

    bow = [dictionary.doc2bow(text) for text in corpus]

    tfidf = models.TfidfModel(bow)
    corpus_tfidf = tfidf[bow]

    lsi = LsiModel(
        corpus_tfidf, 
        num_topics=num_topics, 
        id2word=dictionary, 
        random_seed=2022
    )
    return {
        "lsi": lsi, 
        "bow": bow,
        "num_topics": num_topics, 
        "df_cluster": df_cluster, 
        "cluster_id": cluster_id
    }

def get_topics(lsa):
    lsi = lsa["lsi"]
    bow = lsa["bow"]
    num_topics = lsa["num_topics"]
    df_cluster = lsa["df_cluster"]
    
    corpus_lsi = lsi[bow]
    scores = [[] for _ in range(num_topics)]
    for doc in corpus_lsi:
        for k in range(num_topics):
            scores[k].append(round(doc[k][1],2))

    df_topic = df_cluster.copy()
    for k in range(num_topics):
        df_topic[f'score_topic_{k}'] = scores[k]

    df_topic['Topic']= df_topic[
        [f'score_topic_{k}' for k in range(num_topics)]
    ].apply(lambda x: str(x.argmax()), axis=1)
    return df_topic

def plot_frec_topics(df_topic, cluster_id):
    table_hist_per_topic = pd.DataFrame(df_topic["Topic"].value_counts()).reset_index().rename(columns={"index": "Topic", "Topic": "Frequency"})
    table_hist_per_topic["Topic"] = table_hist_per_topic["Topic"].astype(str)
    table_hist_per_topic = table_hist_per_topic.sort_values("Frequency", ascending=False)

    relevant_topics = table_hist_per_topic["Topic"].unique()

    fig = px.bar(
        table_hist_per_topic, 
        color="Topic", 
        x="Topic", 
        y="Frequency", 
        title=f"Frecuency of topics (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()
    
def get_strengths(lsa):
    lsi = lsa["lsi"]
    num_topics = lsa["num_topics"]
    
    strengths_per_topic = []
    data_strengths_per_topic = lsi.print_topics(num_topics)
    for topic_k in range(len(data_strengths_per_topic)):
        id_k, strengths_k = data_strengths_per_topic[topic_k]
        strengths_k = strengths_k.split(" + ")
        dic_rev = {}
        for a_w in strengths_k:
            a, w = a_w.split("*")
            a = float(a)
            w=w.replace('"', "")
            if w:
                do = True
                for k, v in dic_rev.copy().items():
                    if str(v) == str(a):
                        if w in k: do = False
                        elif k in w:
                            dic_rev.pop(k)
                            do = False
                if do:
                    dic_rev[w] = a
        for w, a in dic_rev.items():
            o = {
                "Topic": str(id_k), 
                "Term": w, 
                "Strength": a
            }
            strengths_per_topic.append(o)
    return strengths_per_topic

def plot_strength(strengths, relevant_topics, cluster_id):
    table_strength = pd.DataFrame(strengths)
    table_strength = table_strength[table_strength["Topic"].isin(relevant_topics)]
    table_strength = table_strength.sort_values("Strength", ascending=True)

    fig = px.bar(
        table_strength, 
        color="Topic", 
        y="Term", 
        x="Strength", 
        title=f"Strength of topics terms (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=8
        ),
        height=800,
        width=500
    )
    fig.show()
    
def get_terms_table(df_topics, strengths):
    df_terms = pd.DataFrame(strengths).drop(columns="Strength").groupby("Topic").aggregate(lambda x: ", ".join(x).replace("_", " ")).rename(columns={"Term": "#Term"}).reset_index()
    df_frec_topics = pd.DataFrame(df_topics["Topic"].value_counts()).reset_index().rename(columns={"index": "Topic", "Topic": "#Docs"})
    df_table_terms = df_terms.merge(df_frec_topics, on="Topic", how="outer").set_index("Topic")
    df_table_terms["#Docs"] = df_table_terms["#Docs"].fillna(0).astype(int)
    return df_table_terms