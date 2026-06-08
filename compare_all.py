# ==========================================
# File: compare_all.py
# ==========================================

import os
import json
import time
import matplotlib.pyplot as plt

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from evaluation import Evaluation

# ==========================================
# IMPORT ALL IR SYSTEMS
# ==========================================

from informationRetrieval import InformationRetrieval as IR_TFIDF

from informationRetrieval_LSA import InformationRetrieval as IR_LSA

from informationRetrieval_LSA_Improved import InformationRetrieval as IR_LSA_IMPROVED

from informationRetrieval_Hybrid import InformationRetrieval as IR_HYBRID


# ==========================================
# TOTAL TIMER START
# ==========================================

total_start = time.time()


# ==========================================
# DATASET
# ==========================================

dataset = "cranfield/"


# ==========================================
# PREPROCESSING MODULES
# ==========================================

segmenter = SentenceSegmentation()

tokenizer = Tokenization()

reducer = InflectionReduction()

stopper = StopwordRemoval()

evaluator = Evaluation()


# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess(texts):

    seg = [segmenter.punkt(t) for t in texts]

    tok = [tokenizer.pennTreeBank(s) for s in seg]

    red = [reducer.reduce(t) for t in tok]

    stop = [stopper.fromList(r) for r in red]

    return stop


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Cranfield dataset...\n")

queries_json = json.load(
    open(os.path.join(dataset, "cran_queries.json"))
)

docs_json = json.load(
    open(os.path.join(dataset, "cran_docs.json"))
)

qrels = json.load(
    open(os.path.join(dataset, "cran_qrels.json"))
)

query_ids = [
    item["query number"]
    for item in queries_json
]

queries = [
    item["query"]
    for item in queries_json
]

doc_ids = [
    item["id"]
    for item in docs_json
]

docs = [
    item["body"]
    for item in docs_json
]



print("Preprocessing queries and documents (shared copy for analysis)...\n")

proc_queries = preprocess(queries)

proc_docs = preprocess(docs)



systems_classes = {
    "TF-IDF":        IR_TFIDF,
    "LSA":           IR_LSA,
    "Improved-LSA":  IR_LSA_IMPROVED,
    "Hybrid":        IR_HYBRID
}


# ==========================================
# RUN ALL SYSTEMS — END-TO-END TIMING
# Each system independently preprocesses + builds index + ranks
# ==========================================

rankings = {}

times = {}

for system_name, IRClass in systems_classes.items():

    print(f"\nRunning {system_name} (end-to-end timing)...\n")

    start = time.time()

    # --- Preprocessing (counted in this system's time) ---
    proc_queries_local = preprocess(queries)

    proc_docs_local = preprocess(docs)

    # --- Build index ---
    ir_system = IRClass()

    ir_system.buildIndex(proc_docs_local, doc_ids)

    # --- Rank ---
    ranked_docs = ir_system.rank(proc_queries_local)

    end = time.time()

    rankings[system_name] = ranked_docs

    times[system_name] = end - start

# ==========================================
# PER-SYSTEM TIME SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("PER-SYSTEM END-TO-END TIME SUMMARY")
print("=" * 60)
print(f"{'System':<18} {'Time (s)':>12}")
print("-" * 60)

for system_name in systems_classes.keys():
    print(f"{system_name:<18} {times[system_name]:>12.4f}")

print("=" * 60)
# ==========================================
# METRICS STORAGE
# ==========================================

metrics = {}

for system_name in systems_classes.keys():

    metrics[system_name] = {

        "P":   [],
        "R":   [],
        "F":   [],
        "MAP": [],
        "nDCG": [],
        "MRR": []
    }


# ==========================================
# EVALUATION (k = 1 to 10)
# ==========================================

print("\n")
print("=" * 120)
print("RUNNING EVALUATION")
print("=" * 120)

for k in range(1, 11):

    print(f"\nEvaluating at k = {k}")

    for system_name in systems_classes.keys():

        ranked = rankings[system_name]

        metrics[system_name]["P"].append(

            evaluator.meanPrecision(
                ranked,
                query_ids,
                qrels,
                k
            )
        )

        metrics[system_name]["R"].append(

            evaluator.meanRecall(
                ranked,
                query_ids,
                qrels,
                k
            )
        )

        metrics[system_name]["F"].append(

            evaluator.meanFscore(
                ranked,
                query_ids,
                qrels,
                k
            )
        )

        metrics[system_name]["MAP"].append(

            evaluator.meanAveragePrecision(
                ranked,
                query_ids,
                qrels,
                k
            )
        )

        metrics[system_name]["nDCG"].append(

            evaluator.meanNDCG(
                ranked,
                query_ids,
                qrels,
                k
            )
        )

        metrics[system_name]["MRR"].append(

            evaluator.meanReciprocalRank(
                ranked,
                query_ids,
                qrels,
                k
            )
        )


# ==========================================
# PRINT RESULTS FOR ALL k
# ==========================================

print("\n")
print("=" * 140)
print("DETAILED EVALUATION RESULTS (k = 1 to 10)")
print("=" * 140)

for k in range(10):

    current_k = k + 1

    print("\n")
    print("-" * 140)

    print(f"RESULTS @ k = {current_k}")

    print("-" * 140)

    header = (
        f"{'System':<18}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'F':>10}"
        f"{'MAP':>12}"
        f"{'nDCG':>12}"
        f"{'MRR':>12}"
        f"{'Time(s)':>12}"
    )

    print(header)

    print("-" * 140)

    for system_name in systems_classes.keys():

        print(

            f"{system_name:<18}"

            f"{metrics[system_name]['P'][k]:>10.4f}"

            f"{metrics[system_name]['R'][k]:>10.4f}"

            f"{metrics[system_name]['F'][k]:>10.4f}"

            f"{metrics[system_name]['MAP'][k]:>12.4f}"

            f"{metrics[system_name]['nDCG'][k]:>12.4f}"

            f"{metrics[system_name]['MRR'][k]:>12.4f}"

            f"{times[system_name]:>12.2f}"
        )

print("\n" + "=" * 140)


# ==========================================
# SAVE PLOTS
# ==========================================

os.makedirs("output_compare_all", exist_ok=True)

k_vals = list(range(1, 11))

plot_metrics = [

    ("P",    "Precision"),

    ("R",    "Recall"),

    ("F",    "F-score"),

    ("MAP",  "MAP"),

    ("nDCG", "nDCG"),

    ("MRR",  "MRR")
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle(
    "TF-IDF vs LSA vs Improved-LSA vs Hybrid",
    fontsize=18
)

for ax, (metric, title) in zip(
    axes.flatten(),
    plot_metrics
):

    for system_name in systems_classes.keys():

        ax.plot(
            k_vals,
            metrics[system_name][metric],
            marker='o',
            label=system_name
        )

    ax.set_title(title)

    ax.set_xlabel("k")

    ax.grid(True)

    ax.legend()

plt.tight_layout()

plt.savefig(
    "output_compare_all/all_systems_comparison.png"
)

print("\nPlot saved:")
print("output_compare_all/all_systems_comparison.png")


# ==========================================
# FAILURE ANALYSIS
# ==========================================

print("\n")
print("=" * 140)
print("FAILURE ANALYSIS")
print("=" * 140)

# Failure queries from previous analysis
failure_queries = [5, 8, 9, 12, 14, 18, 19]

# Query lookup
query_lookup = {

    q["query number"]: q["query"]

    for q in queries_json
}

# Document title lookup
doc_lookup = {

    int(d["id"]): d["title"]

    for d in docs_json
}

# Relevant docs lookup
relevant_lookup = {}

for item in qrels:

    qid = int(item["query_num"])

    did = int(item["id"])

    if qid not in relevant_lookup:
        relevant_lookup[qid] = set()

    relevant_lookup[qid].add(did)




for qid in failure_queries:

    print("\n" + "-" * 140)

    print(f"\nQuery ID : {qid}")

    print(f"Query    : {query_lookup[qid]}")

    print("\nRelevant Documents:")

    rel_docs = list(relevant_lookup[qid])[:5]

    for rid in rel_docs:

        title = doc_lookup.get(rid, "Unknown")

        print(f"  {rid} -> {title}")

    print("\nTop Retrievals:\n")

    query_index = qid - 1

    for system_name in systems_classes.keys():

        ranked = rankings[system_name]

        top_doc = int(ranked[query_index][0])

        title = doc_lookup.get(top_doc, "Unknown")

        correct = (

            "CORRECT ✅"

            if top_doc in relevant_lookup[qid]

            else "WRONG ❌"
        )

        print(
            f"{system_name:<18}"
            f" -> Doc {top_doc:<6}"
            f" -> {correct}"
        )

        print(f"   Title: {title}")

    print("\n")




print("\n")
print("=" * 140)
print("OOV QUERY TEST")
print("=" * 140)

oov_query = "supersonic metamaterial wing design"

print(f"\nOOV Query: {oov_query}\n")

proc_oov = preprocess([oov_query])

oov_systems = {
    "TF-IDF":       IR_TFIDF(),
    "LSA":          IR_LSA(),
    "Improved-LSA": IR_LSA_IMPROVED(),
    "Hybrid":       IR_HYBRID()
}

for system_name, ir_system in oov_systems.items():

    ir_system.buildIndex(proc_docs, doc_ids)

    ranked = ir_system.rank(proc_oov)

    top_doc = int(ranked[0][0])

    title = doc_lookup.get(top_doc, "Unknown")

    print(f"{system_name:<18} -> Doc {top_doc}")

    print(f"Title: {title}\n")


