# informationRetrieval_LSA.py

from util import *
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class InformationRetrieval():

    def __init__(self):
        self.index = {"tfidf": [], "idf": {}, "docIDs": []}
        self.vectorizer = TfidfVectorizer(stop_words='english',sublinear_tf=True,max_df=0.7,min_df=2,ngram_range=(1,2))
        self.svd        = TruncatedSVD(n_components=150,random_state=42)
        self.doc_matrix = np.zeros((1,100))

    def buildIndex(self, docs, docIDs):

        # Flatten docs
        flat_docs = [
            " ".join(token for sentence in doc for token in sentence)
            for doc in docs
        ]

        # TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix    = self.vectorizer.fit_transform(flat_docs)

        # LSA — reduce to 100 dimensions
        self.svd        = TruncatedSVD(n_components=100, random_state=42)
        lsa_matrix      = self.svd.fit_transform(tfidf_matrix)

        # Normalize for cosine similarity
        self.doc_matrix = normalize(lsa_matrix)
        self.index      = {"docIDs": docIDs}

        print(f"  Index built: {len(docIDs)} docs, "
              f"{tfidf_matrix.shape[1]} terms → 100 LSA dims")

    def rank(self, queries):
        doc_IDs_ordered = []

        for query in queries:
            # Flatten query
            flat_query  = " ".join(token for sentence in query for token in sentence)

            # Project into LSA space
            query_tfidf = self.vectorizer.transform([flat_query])
            query_lsa   = self.svd.transform(query_tfidf)
            query_norm  = normalize(query_lsa)

            # Cosine similarity = dot product (since both normalized)
            scores      = self.doc_matrix.dot(query_norm.T).flatten()

            # Rank descending
            ranked_idx  = np.argsort(scores)[::-1]
            ranked_ids  = [self.index["docIDs"][i] for i in ranked_idx]
            doc_IDs_ordered.append(ranked_ids)

        return doc_IDs_ordered