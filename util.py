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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from gensim.matutils import corpus2csc

pd.set_option('max_colwidth', 200)

def read_network(file_path):
    f = open(file_path, encoding='utf-8-sig') # delete descripcion:
    network = json.load(f)["network"]
    network_items = pd.DataFrame(network["items"])
    network_items["DOI"] = network_items["url"].apply(lambda x: x.replace("https://doi.org/", "").lower() if str(x) != "nan" else np.nan)
    network_items["Citations"] = network_items["weights"].apply(lambda x: x["Citations"])
    network_items["Links"] = network_items["weights"].apply(lambda x: x["Links"])
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
    table_hist["#Docs"] = df_docs["cluster"].value_counts().values
    table_hist["Cluster"] = [str(i) for i in range(1, num_clusters+1)]
    color_discrete_sequence = ['#d63f4b', 
                               "#58bd5b", 
                               "#5599c3", 
                               "#bcbe4f", 
                               "#996fc0", 
                               "#5cccd9"][:num_clusters]
    fig = px.bar(table_hist, 
                 x="Cluster", 
                 y="#Docs", 
                 color="Cluster", 
                 title="#Docs per cluster (LinLog/mod.)", 
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()

def plot_docs_per_year(df_biblio):
    fig = px.histogram(df_biblio, x="Year", text_auto=True)
    fig.update_layout(
        title="Histogram of #Docs per year"
    )
    fig.show(renderer="notebook")
    
def plot_docs_per_source(df_biblio):
    frec_sources = pd.DataFrame(df_biblio["Source title"].value_counts()).reset_index().iloc[:25,:].rename(columns={"index": "Source title", "Source title": "#Docs"})
    fig = px.bar(frec_sources, x="Source title", y="#Docs", text_auto=True)
    fig.update_layout(
        title="Histogram of #Docs per source"
    )
    fig.show(renderer="notebook")
    
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
        "corpus_tfidf": corpus_tfidf,
        "num_topics": num_topics, 
        "df_cluster": df_cluster, 
        "cluster_id": cluster_id
    }

def plot_top_eigenvalues(lsa, cluster_id=1, k=25):
    corpus_csc = corpus2csc(lsa["corpus_tfidf"])
    vals, _ = eigs(corpus_csc @ corpus_csc.T, k=k)
    vals = np.real(vals)

    top_eigs_vals = pd.DataFrame(vals).reset_index()
    top_eigs_vals["index"] += 1
    top_eigs_vals["index"] = top_eigs_vals["index"].astype(str)
    top_eigs_vals = top_eigs_vals.rename(columns={0: "Value", "index": "N-th eigenvalue"})

    fig = px.bar(
        top_eigs_vals, 
        x="N-th eigenvalue", 
        y="Value", 
        title=f"Top {k} eigenvalues (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show(renderer="notebook")
    
def get_topics(lsa, num_std_u=1, num_std_v=1):
    abs_u = abs(lsa["lsi"].projection.u)
    tresh_u = np.mean(abs_u, axis=0) + num_std_u * np.std(abs_u, axis=0)

    index_u_per_factor = np.where(abs_u>tresh_u)
    term_per_factor = csr_matrix(
        (
            abs_u[index_u_per_factor[0], index_u_per_factor[1]], 
            (index_u_per_factor[0], index_u_per_factor[1])
        ), 
        shape=abs_u.shape
    )

    num_terms = int(lsa["lsi"].num_terms)
    num_docs = int(lsa["lsi"].docs_processed)
    corpus_csc = corpus2csc(lsa["corpus_tfidf"], num_terms=num_terms, num_docs=num_docs)

    doc_per_factor = corpus_csc.T @ term_per_factor
    doc_per_factor = doc_per_factor / np.sum(doc_per_factor, axis=1)
    doc_per_factor = np.asarray(doc_per_factor)

    tresh_v = np.mean(doc_per_factor, axis=0) + num_std_v * np.std(doc_per_factor, axis=0)
    index_v_per_factor = np.where(doc_per_factor>tresh_v)

    index_per_factor = {
        "u": index_u_per_factor, 
        "v": index_v_per_factor
    }
    return {
        "term_per_factor": term_per_factor, 
        "doc_per_factor": doc_per_factor, 
        "index_per_factor": index_per_factor
    }

def get_table_hist_per_topic(index_per_factor):
    index_v_per_factor, index_u_per_factor = index_per_factor["v"], index_per_factor["u"]
    
    table_hist_per_topic = pd.DataFrame(pd.Series(index_v_per_factor[1].astype(str)).value_counts())
    table_hist_per_topic = table_hist_per_topic.reset_index().rename(columns={"index": "Topic", 0: "#Docs"})
    
    aux = pd.DataFrame(pd.Series(index_u_per_factor[1].astype(str)).value_counts())
    aux = aux.reset_index().rename(columns={"index": "Topic", 0: "#Terms"})
    table_hist_per_topic = table_hist_per_topic.merge(aux, on="Topic", how="left")
    
    table_hist_per_topic = table_hist_per_topic.sort_values("Topic")
    return table_hist_per_topic

def plot_topics(table_hist_per_topic, cluster_id, by):
    fig = px.bar(
        table_hist_per_topic, 
        color="Topic", 
        x="Topic", 
        y=f"#{by}", 
        title=f"#{by} per topic (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()
    
def get_strengths(lsa, term_per_factor, num_terms=10):
    strengths = [] 
    for factor in range(term_per_factor.shape[1]):
        id_terms_f = np.argsort(term_per_factor[:, factor].toarray().flatten())[::-1][:num_terms]
        strenght_f = term_per_factor[id_terms_f, factor].toarray().flatten()
        terms_f = [lsa["lsi"].id2word[ix] for ix in id_terms_f]
        for t, s in zip(terms_f, strenght_f):
            o = {
                'Topic': str(factor), 
                'Term': t, 
                'Strength': s
            }
            strengths.append(o)
    return strengths

def plot_strength_per_topic(strengths, cluster_id):
    table_strength = pd.DataFrame(strengths)
    table_strength = table_strength.sort_values(["Topic", "Strength"], ascending=[True, False])

    fig = px.bar(
        table_strength, 
        color="Topic", 
        y="Term", 
        x="Strength", 
        title=f"Strength of topics terms (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=10
        ),
        height=700,
        width=600
    )
    fig.show()
    
def get_terms_table(table_hist_per_topic, strengths):
    df_terms = pd.DataFrame(strengths).drop(columns="Strength").groupby("Topic").aggregate(
        lambda x: ", ".join(x).replace("_", " ")
    )
    df_terms = df_terms.reset_index().rename(columns={"Term": "Top 10 terms"})
    df_terms = df_terms.merge(table_hist_per_topic, on="Topic", how="left")
    df_terms = df_terms["Topic  Top 10 terms  #Terms  #Docs".split("  ")]
    return df_terms.set_index("Topic")

def get_group_table(network_items):
    table_summary = pd.DataFrame(
        {
            "#Docs": network_items["cluster"].fillna(0).value_counts().values,
            "Group": ["Others", "Dense component"]
        }
    ).set_index("Group")
    return table_summary

def get_links_stats_table(network_items):
    stats = [
        ("sum", sum), 
        ("mean", np.mean), 
        ("std", np.std), 
        ("min", min), 
        ("25%", lambda x: np.quantile(x, q=0.25)), 
        ("50%", lambda x: np.quantile(x, q=0.50)), 
        ("75%", lambda x: np.quantile(x, q=0.75)),
        ("max", max)
    ]
    table_stats = network_items["cluster Links".split()].fillna({"cluster": 0}).groupby("cluster").aggregate(stats)
    table_stats.index = ["Others", "Dense component"]
    stats_overall = network_items["cluster Links".split()].fillna({"cluster": 1}).groupby("cluster").aggregate(stats)
    stats_overall.index = ["All"]
    table_stats = pd.concat([stats_overall, table_stats], axis=0)
    table_stats.index.name = "Docs"
    return table_stats

def get_top_cited_table(df_docs, network_items, min_citations=31):
    id_top_cited = network_items[network_items["Citations"] >= 31]["id"]
    top_cited = df_docs[df_docs["id"].isin(id_top_cited)]
    table_top_cited = top_cited[top_cited["cluster"] == 1].sort_values("Year")
    table_top_cited = pd.concat([table_top_cited, top_cited[top_cited["cluster"] == 2].sort_values("Year")])
    table_top_cited = pd.concat([table_top_cited, top_cited[top_cited["cluster"] == 3].sort_values("Year")])
    table_top_cited = table_top_cited["cluster Year Authors Title Citations".split()].rename(columns={"cluster": "Cluster"})
    table_top_cited["Citations"] = table_top_cited["Citations"].astype(int)
    return table_top_cited