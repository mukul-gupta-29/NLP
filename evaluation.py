from util import *
import math

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Precision@k for a single query
        = (relevant docs in top-k) / k
        """
        top_k = query_doc_IDs_ordered[:k]
        true_set = set(true_doc_IDs)
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in true_set)
        precision = relevant_retrieved / k
        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Mean Precision@k over all queries
        """
        precisions = []
        for i, query_id in enumerate(query_ids):
            # Get ground truth doc IDs for this query
            true_doc_IDs = [int(qrel["id"]) for qrel in qrels
                            if int(qrel["query_num"]) == int(query_id)]
            p = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            precisions.append(p)
        meanPrecision = sum(precisions) / len(precisions)
        return meanPrecision


    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Recall@k for a single query
        = (relevant docs in top-k) / (total relevant docs)
        """
        top_k = query_doc_IDs_ordered[:k]
        true_set = set(true_doc_IDs)
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in true_set)
        if len(true_set) == 0:
            return 0.0
        recall = relevant_retrieved / len(true_set)
        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Mean Recall@k over all queries
        """
        recalls = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = [int(qrel["id"]) for qrel in qrels
                            if int(qrel["query_num"]) == int(query_id)]
            r = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            recalls.append(r)
        meanRecall = sum(recalls) / len(recalls)
        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        F-score@k for a single query
        = 2 * (Precision * Recall) / (Precision + Recall)
        """
        p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        beta = 0.5
        if p + r == 0:
            return 0.0
    # F-beta formula
        fscore = (1 + beta**2) * p * r / (beta**2 * p + r)
        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Mean F-score@k over all queries
        """
        fscores = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = [int(qrel["id"]) for qrel in qrels
                            if int(qrel["query_num"]) == int(query_id)]
            f = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            fscores.append(f)
        meanFscore = sum(fscores) / len(fscores)
        return meanFscore


    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        nDCG@k for a single query

        Relevance score = 1 if doc is relevant, 0 otherwise
        DCG  = sum( rel_i / log2(i+1) ) for i = 1..k
        IDCG = DCG of the ideal (perfect) ranking
        nDCG = DCG / IDCG
        """
        top_k = query_doc_IDs_ordered[:k]
        true_set = set(true_doc_IDs)

        # DCG — actual ranking
        dcg = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            rel = 1 if doc_id in true_set else 0
            dcg += rel / math.log2(rank + 1)

        # IDCG — ideal ranking (all relevant docs at the top)
        ideal_rels = [1] * min(len(true_set), k)  # best case: k relevant docs
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels, start=1):
            idcg += rel / math.log2(rank + 1)

        if idcg == 0:
            return 0.0
        nDCG = dcg / idcg
        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Mean nDCG@k over all queries
        """
        ndcgs = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = [int(qrel["id"]) for qrel in qrels
                            if int(qrel["query_num"]) == int(query_id)]
            n = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            ndcgs.append(n)
        meanNDCG = sum(ndcgs) / len(ndcgs)
        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Average Precision@k for a single query
        = average of Precision@i for each position i (1..k)
          where the document at position i is relevant
        """
        top_k = query_doc_IDs_ordered[:k]
        true_set = set(true_doc_IDs)

        total_precision = 0.0
        relevant_count  = 0

        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in true_set:
                relevant_count += 1
                total_precision += relevant_count / rank  # Precision@rank

        if relevant_count == 0:
            return 0.0
        avgPrecision = total_precision / len(true_set)
        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        MAP@k over all queries
        """
        avg_precisions = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = [int(qrel["id"]) for qrel in q_rels
                            if int(qrel["query_num"]) == int(query_id)]
            ap = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            avg_precisions.append(ap)
        meanAveragePrecision = sum(avg_precisions) / len(avg_precisions)
        return meanAveragePrecision


    def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Reciprocal Rank@k for a single query
        = 1 / rank of the first relevant document in top-k
        = 0 if no relevant document found in top-k
        """
        top_k = query_doc_IDs_ordered[:k]
        true_set = set(true_doc_IDs)

        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in true_set:
                return 1 / rank  # first hit

        return 0.0  # no relevant doc found in top-k


    def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        MRR@k over all queries
        """
        rr_scores = []
        for i, query_id in enumerate(query_ids):
            true_doc_IDs = [int(qrel["id"]) for qrel in qrels
                            if int(qrel["query_num"]) == int(query_id)]
            rr = self.queryReciprocalRank(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
            rr_scores.append(rr)
        meanReciprocalRank = sum(rr_scores) / len(rr_scores)
        return meanReciprocalRank