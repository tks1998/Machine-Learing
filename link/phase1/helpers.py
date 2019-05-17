import math
from collections import defaultdict
import copy

def compute_idf(corpus):
    num_docs = len(corpus)
    idf = defaultdict(lambda: 0)
    for doc in corpus:
        for word in doc.keys():
            idf[word] += 1

    for word, value in idf.items():
        idf[word] = 1 + math.log(num_docs / value)
    return idf


def compute_tf(corpus):
    tf = copy.deepcopy(corpus)
    for doc in tf:
        for word, value in doc.items():
            doc[word] = value / len(doc)

    return tf


def compute_weight(corpus):
    idf = compute_idf(corpus)
    tf = compute_tf(corpus)
    weight = list()
  
    for doc in tf:
        weight_ = list()
        for term in idf.keys():
            weight_.append(doc[term] * idf[term] if term in doc.keys() else 0)
        weight.append(weight_)
        print(len(weight))

    return weight
