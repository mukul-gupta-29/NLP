# ==========================================
# File: informationRetrieval_Hybrid.py
# ==========================================

from util import *

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class InformationRetrieval():

    def __init__(self):

        self.index = {"docIDs": []}

        
        # Word-level TF-IDF
        
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            max_df=0.8,
            min_df=2,
            ngram_range=(1, 2)
        )

        
        # Character-level TF-IDF
        
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2
        )

        
        # LSA
        
        self.svd = TruncatedSVD(
            n_components=300,
            random_state=42
        )

        self.word_matrix = None
        self.char_matrix = None
        self.lsa_matrix = None

    # ======================================
    # Build Index
    # ======================================

    def buildIndex(self, docs, docIDs):

        flat_docs = [
            " ".join(token for sentence in doc for token in sentence)
            for doc in docs
        ]

        # Word TF-IDF
        word_tfidf = self.word_vectorizer.fit_transform(flat_docs)

        # Character TF-IDF
        self.char_matrix = self.char_vectorizer.fit_transform(flat_docs)

        # LSA
        lsa = self.svd.fit_transform(word_tfidf)

        self.lsa_matrix = normalize(lsa)

        self.word_matrix = normalize(word_tfidf)

        self.index["docIDs"] = docIDs

        print("\n========== HYBRID IR ==========")
        print(f"Documents Indexed : {len(docIDs)}")
        print(f"TF-IDF Features   : {word_tfidf.shape[1]}")
        print(f"LSA Dimensions    : {self.lsa_matrix.shape[1]}")
        print("================================\n")

    # ======================================
    # Rank
    # ======================================

    def rank(self, queries):

        doc_IDs_ordered = []

        for query in queries:

            flat_query = " ".join(
                token
                for sentence in query
                for token in sentence
            )

            
            # TF-IDF similarity
            
            query_word = self.word_vectorizer.transform(
                [flat_query]
            )

            tfidf_scores = cosine_similarity(
                self.word_matrix, # pyright: ignore[reportArgumentType]
                normalize(query_word)
            ).flatten()

            
            # LSA similarity
          
            query_lsa = self.svd.transform(query_word)

            lsa_scores = cosine_similarity(
                self.lsa_matrix, # pyright: ignore[reportArgumentType]
                normalize(query_lsa)
            ).flatten()

           
            # Character similarity
          
            query_char = self.char_vectorizer.transform(
                [flat_query]
            )

            char_scores = cosine_similarity(
                self.char_matrix, # pyright: ignore[reportArgumentType]
                query_char
            ).flatten()

            
            # Hybrid weighted score
           
            final_scores = (
                0.4 * tfidf_scores +
                0.5 * lsa_scores +
                0.1 * char_scores
            )

            ranked_idx = np.argsort(final_scores)[::-1]

            ranked_ids = [
                self.index["docIDs"][i]
                for i in ranked_idx
            ]

            doc_IDs_ordered.append(ranked_ids)

        return doc_IDs_ordered