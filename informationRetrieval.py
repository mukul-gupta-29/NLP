from util import *
import math
from collections import defaultdict

class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        N = len(docs)  # total number of documents

        # --- Step 1: Flatten each doc from list-of-sentences to list-of-tokens ---
        flat_docs = []
        for doc in docs:
            tokens = []
            for sentence in doc:
                tokens.extend(sentence)
            flat_docs.append(tokens)

        # --- Step 2: Compute TF for each document ---
        # tf[doc_index][term] = raw count / total terms in doc
        tf = []
        for tokens in flat_docs:
            tf_doc = defaultdict(float)
            total = len(tokens)
            if total == 0:
                tf.append(tf_doc)
                continue
            for token in tokens:
                tf_doc[token.lower()] += 1
            for term in tf_doc:
                tf_doc[term] /= total
            tf.append(tf_doc)

        # --- Step 3: Compute DF then IDF for each term ---
        # df[term] = number of documents containing the term
        df = defaultdict(int)
        for tf_doc in tf:
            for term in tf_doc:
                df[term] += 1

        # idf[term] = log(N / df(term))
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log(N / freq)

        # --- Step 4: Compute TF-IDF vectors for each document ---
        # tfidf[doc_index][term] = tf * idf
        tfidf = []
        for tf_doc in tf:
            tfidf_doc = {}
            for term, tf_val in tf_doc.items():
                tfidf_doc[term] = tf_val * idf[term]
            tfidf.append(tfidf_doc)

        # --- Store everything needed for ranking ---
        self.index = {
            "tfidf": tfidf,       # list of dicts, one per document
            "idf": idf,           # idf values for query terms
            "docIDs": docIDs      # to map index → actual doc ID
        }


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        tfidf  = self.index["tfidf"] # type: ignore
        idf    = self.index["idf"] # type: ignore
        docIDs = self.index["docIDs"] # pyright: ignore[reportOptionalSubscript]

        for query in queries:

            # --- Step 1: Flatten query sentences → tokens ---
            query_tokens = []
            for sentence in query:
                query_tokens.extend(sentence)

            # --- Step 2: Build TF vector for query ---
            query_tf = defaultdict(float)
            total = len(query_tokens)
            if total == 0:
                doc_IDs_ordered.append(docIDs)
                continue
            for token in query_tokens:
                query_tf[token.lower()] += 1
            for term in query_tf:
                query_tf[term] /= total

            # --- Step 3: Build TF-IDF vector for query ---
            query_tfidf = {}
            for term, tf_val in query_tf.items():
                if term in idf:
                    query_tfidf[term] = tf_val * idf[term]
                # terms not in index get weight 0 (ignored)

            # --- Step 4: Cosine similarity with each document ---
            def cosine_similarity(doc_vec, query_vec):
                # dot product
                dot = sum(doc_vec.get(t, 0) * w for t, w in query_vec.items())

                # magnitudes
                doc_mag   = math.sqrt(sum(v**2 for v in doc_vec.values()))
                query_mag = math.sqrt(sum(v**2 for v in query_vec.values()))

                if doc_mag == 0 or query_mag == 0:
                    return 0.0
                return dot / (doc_mag * query_mag)

            scores = []
            for i, doc_vec in enumerate(tfidf):
                score = cosine_similarity(doc_vec, query_tfidf)
                scores.append((docIDs[i], score))

            # --- Step 5: Sort by score descending ---
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, score in scores]

            doc_IDs_ordered.append(ranked_ids)

        return doc_IDs_ordered