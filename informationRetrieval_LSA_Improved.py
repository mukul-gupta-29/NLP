# ================================
# File: informationRetrieval_LSA_Improved.py
# ================================

from util import *

import numpy as np
from collections import Counter
from difflib import get_close_matches

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class InformationRetrieval():

    def __init__(self):

        self.index = {"docIDs": []}

        # Improved TF-IDF
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            max_df=0.8,
            min_df=2,
            ngram_range=(1, 2)
        )

        # Character-level features
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2
        )

        # Improved LSA dimensions
        self.svd = TruncatedSVD(
            n_components=300,
            random_state=42
        )

        self.doc_matrix = None
        self.char_matrix = None

        self.vocab = set()
        self.term_freq = Counter()

    
    # Build Index
    

    def buildIndex(self, docs, docIDs):

        flat_docs = [
            " ".join(token for sentence in doc for token in sentence)
            for doc in docs
        ]

        # Build vocabulary
        all_words = []

        for doc in flat_docs:
            words = doc.lower().split()
            all_words.extend(words)

        self.vocab = set(all_words)
        self.term_freq = Counter(all_words)

        # Word TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(flat_docs)

        # Character TF-IDF
        self.char_matrix = self.char_vectorizer.fit_transform(flat_docs)

        # LSA reduction
        lsa_matrix = self.svd.fit_transform(tfidf_matrix)

        # Normalize
        self.doc_matrix = normalize(lsa_matrix)

        self.index["docIDs"] = docIDs

        print("\n========== IMPROVED LSA INDEX ==========")
        print(f"Documents Indexed      : {len(docIDs)}")
        print(f"Vocabulary Size        : {len(self.vocab)}")
        print(f"Original TF-IDF Dims   : {tfidf_matrix.shape[1]}")
        print(f"Reduced LSA Dimensions : {self.doc_matrix.shape[1]}")
        print("========================================\n")

    
    # Spell Correction
    

    def correct_query(self, query):

        corrected_words = []

        for word in query.lower().split():

            if word in self.vocab:
                corrected_words.append(word)
                continue

            matches = get_close_matches(
                word,
                self.vocab,
                n=1,
                cutoff=0.75
            )

            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    
    # Auto-complete
    

    def autocomplete(self, prefix, top_k=5):

        suggestions = []

        for word in self.vocab:
            if word.startswith(prefix.lower()):
                suggestions.append(
                    (word, self.term_freq[word])
                )

        suggestions = sorted(
            suggestions,
            key=lambda x: x[1],
            reverse=True
        )

        return [word for word, _ in suggestions[:top_k]]

    
    # Rank Documents
    

    def rank(self, queries):

        doc_IDs_ordered = []

        for query in queries:

            flat_query = " ".join(
                token
                for sentence in query
                for token in sentence
            )

            corrected_query = self.correct_query(flat_query)

            words = corrected_query.split()
            if len(words) <= 2:
                expanded = [
                    self.autocomplete(w, top_k=1)
                    for w in words
                ]
                expanded_words = [
                    alts[0] if alts and alts[0] != w else w
                    for w, alts in zip(words, expanded)
                ]
                corrected_query = " ".join(expanded_words)

            query_tfidf = self.vectorizer.transform(
                [corrected_query]
            )

            query_lsa = self.svd.transform(query_tfidf)

            query_norm = normalize(query_lsa)

            lsa_scores = cosine_similarity(
                self.doc_matrix, # pyright: ignore[reportArgumentType]
                query_norm
            ).flatten()

            query_char = self.char_vectorizer.transform(
                [corrected_query]
            )

            char_scores = cosine_similarity(
                self.char_matrix, # pyright: ignore[reportArgumentType]
                query_char
            ).flatten()

            scores = 0.8 * lsa_scores + 0.2 * char_scores

            ranked_idx = np.argsort(scores)[::-1]

            ranked_ids = [
                self.index["docIDs"][i]
                for i in ranked_idx
            ]

            doc_IDs_ordered.append(ranked_ids)

        return doc_IDs_ordered
